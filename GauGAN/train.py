import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, serializers
import numpy as np
import argparse
import os
import pylab
from model import Generator, Discriminator, Prior
from prepare import prepare_dataset

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta=0.0, beta2=0.999):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta, beta2=beta2)
    optimizer.setup(model)

    return optimizer

def calc_loss(fake,real):
    loss = 0
    for f,r in zip(fake,real):
        _,c,h,w=f.shape
        loss+=F.mean_absolute_error(f,r) / (c*h*w)
        
        return loss

def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    
    return loss

parser = argparse.ArgumentParser(description='GauGAN')
parser.add_argument('--epoch', default=1000, type=int, help="the number of epochs")
parser.add_argument('--batchsize', default=16, type=int, help="batch size")
parser.add_argument('--testsize', default=4, type=int, help="test size")
parser.add_argument('--weight', default=10.0, type=float, help="the weight of content loss")
parser.add_argument('--iter', default=2000, type=int, help="the number of iterations")
parser.add_argument('--N', default=19000, type=int, help="the number of training images")
parser.add_argument('--beta', default=0.05, type=float, help="the weight of kl divergence")
args = parser.parse_args()

epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
weight = args.weight
itertions = args.iter
Ntrain = args.N
beta = args.beta

outdir = './output'
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator, alpha=0.0001)

discriminator_0 = Discriminator()
discriminator_0.to_gpu()
dis0_opt = set_optimizer(discriminator_0, alpha=0.0004)

discriminator_2 = Discriminator()
discriminator_2.to_gpu()
dis2_opt = set_optimizer(discriminator_2, alpha=0.0004)

discriminator_4 = Discriminator()
discriminator_4.to_gpu()
dis4_opt = set_optimizer(discriminator_4, alpha=0.0004)

prior = Prior()

line_path="./Dataset/line/train_lin_256/"
color_path="./Dataset/trim/train_trim_color/"

line_box = []
hint_box = []
color_box = []
for index in range(testsize):
    rnd = np.random.randint(Ntrain+1, Ntrain+ 400)
    filename = "trim_free_" + str(rnd) + ".png"
    color, line, image_mask = prepare_dataset(line_path + filename, color_path + filename)
    hint_box.append(image_mask)
    line_box.append(line)
    color_box.append(color)

line_test = chainer.as_variable(xp.array(line_box).astype(xp.float32))
hint_test = chainer.as_variable(xp.array(hint_box).astype(xp.float32))
color_test = chainer.as_variable(xp.array(color_box).astype(xp.float32))

#lh_test = F.concat([line_test, hint_test], axis=1)

ztest = xp.random.uniform(-1,1,(testsize,256),dtype=np.float32)
ztest = chainer.as_variable(ztest)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, itertions, batchsize):
        line_box = []
        hint_box = []
        color_box = []
        for _ in range(batchsize):
            rnd = np.random.randint(1,Ntrain)
            filename = "trim_free_" + str(rnd) + ".png"
            color, line, image_mask = prepare_dataset(line_path+filename, color_path + filename)
            hint_box.append(image_mask)
            line_box.append(line)
            color_box.append(color)

        l = chainer.as_variable(xp.array(line_box).astype(xp.float32))
        h = chainer.as_variable(xp.array(hint_box).astype(xp.float32))
        l_2 = F.average_pooling_2d(l,3,2,1)
        l_4 = F.average_pooling_2d(l_2,3,2,1)
        c = chainer.as_variable(xp.array(color_box).astype(xp.float32))
        c_2 = F.average_pooling_2d(c, 3,2,1)
        c_4 = F.average_pooling_2d(c_2, 3,2,1)
        #lh = F.concat([l, h], axis=1)

        y, _ = generator(c, l)
        y_2 = F.average_pooling_2d(y,3,2,1)
        y_4 = F.average_pooling_2d(y_2,3,2,1)

        dis_fake, _ = discriminator_0(F.concat([y,l]))
        dis_real, _ = discriminator_0(F.concat([c,l]))
        dis2_fake, _ = discriminator_2(F.concat([y_2, l_2]))
        dis2_real, _ = discriminator_2(F.concat([c_2, l_2]))
        dis4_fake, _ = discriminator_4(F.concat([y_4, l_4]))
        dis4_real, _ = discriminator_4(F.concat([c_4, l_4]))

        y.unchain_backward()
        y_2.unchain_backward()
        y_4.unchain_backward()

        dis_loss = loss_hinge_dis(dis_fake, dis_real)
        dis_loss += loss_hinge_dis(dis2_fake, dis2_real)
        dis_loss += loss_hinge_dis(dis4_fake, dis4_real)

        discriminator_0.cleargrads()
        discriminator_2.cleargrads()
        discriminator_4.cleargrads()
        dis_loss.backward()
        dis0_opt.update()
        dis2_opt.update()
        dis4_opt.update()
        dis_loss.unchain_backward()

        y, y_z = generator(c, l)
        y_2 = F.average_pooling_2d(y,3,2,1)
        y_4 = F.average_pooling_2d(y_2,3,2,1)

        dis_fake, fake_feat = discriminator_0(F.concat([y,l]))
        dis_real, real_feat = discriminator_0(F.concat([c,l]))
        dis2_fake, fake_feat2 = discriminator_2(F.concat([y_2, l_2]))
        dis2_real, real_feat2 = discriminator_2(F.concat([c_2, l_2]))
        dis4_fake, fake_feat4 = discriminator_4(F.concat([y_4, l_4]))
        dis4_real, real_feat4 = discriminator_4(F.concat([c_4, l_4]))

        gen_loss = loss_hinge_gen(dis_fake)
        gen_loss += loss_hinge_gen(dis2_fake)
        gen_loss += loss_hinge_gen(dis4_fake)

        ms_loss = calc_loss(fake_feat, real_feat)
        ms_loss += calc_loss(fake_feat2, real_feat2)
        ms_loss += calc_loss(fake_feat4, real_feat4)

        recon_loss = F.mean_absolute_error(y, c)
        recon_loss += F.mean_absolute_error(y_2, c_2)
        recon_loss += F.mean_absolute_error(y_4, c_4)

        p_z = prior()
        kl_divergence = F.mean(F.sum(chainer.kl_divergence(y_z, p_z), axis=-1))

        gen_loss = gen_loss + weight * (ms_loss + recon_loss) + beta * kl_divergence

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if batch == 0:
            serializers.save_npz('generator.model', generator)
            serializers.save_npz('discriminator_0.model', discriminator_0)
            serializers.save_npz('discriminator_2.model', discriminator_2)
            serializers.save_npz('discriminator_4.model', discriminator_4)

            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()

            with chainer.using_config("train", False):
                y, _ = generator(color_test, line_test)
            y = y.data.get()
            sr = line_test.data.get()
            cr = color_test.data.get()
            hr = hint_test.data.get()

            for i_ in range(testsize):
                tmp = (np.clip((sr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((hr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+2)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((cr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+3)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((y[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+4)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
    
    print("epoch : {}".format(epoch))
    print("Generator loss : {}".format(sum_gen_loss/Ntrain))
    print("Discriminator loss : {}".format(sum_dis_loss/Ntrain))