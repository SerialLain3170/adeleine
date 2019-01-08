import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, initializers, serializers
import numpy as np
import os
import argparse
import pylab
from prepare import prepare_refine
from model import VGG,Generator,Discriminator
import i2v

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta1=0.5, beta2 = 0.99):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta1, beta2 = beta2)
    optimizer.setup(model)

    return optimizer

def calc_content_loss(fake, real):
    _, c,w,h = fake.shape
    return F.mean_squared_error(fake, real) / (c*w*h)

parser = argparse.ArgumentParser(description="Colorization")
parser.add_argument("--epoch",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize", default=16, type= int, help="batchsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--testsize",default=2,type=int,help="testsize")
parser.add_argument("--Ntrain",default=19000,type=int,help="the number of train images")
parser.add_argument("--aw",default=0.01,type=float,help="the weight of adversarial loss")
parser.add_argument("--pw",default=10.0,type=float,help="the weight of penalty loss")
parser.add_argument("--dw",default=0.001,type=float,help="epsilon drift")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
testsize = args.testsize
Ntrain = args.Ntrain
adver_weight = args.aw
penalty_weight = args.pw
epsilon_drift = args.dw

output_dir = "./output_refine/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

line_path = "/line/"
color_path = "/color/"
line_box = []
color_box = []
i_mask_box = []
mask_box = []
for index in range(testsize):
    rnd = np.random.randint(Ntrain+1, Ntrain+ 400)
    filename = "trim_free_" + str(rnd) + ".png"
    color,line, image_mask, mask = prepare_refine(line_path + filename, color_path + filename)
    line_box.append(line)
    color_box.append(color)
    i_mask_box.append(image_mask)
    mask_box.append(mask)

line_test = chainer.as_variable(xp.array(line_box).astype(xp.float32))
color_test = chainer.as_variable(xp.array(color_box).astype(xp.float32))
image_mask_test = chainer.as_variable(xp.array(i_mask_box).astype(xp.float32))
#mask_test = chainer.as_variable(xp.array(mask_box).astype(xp.float32))

mask_test = image_mask_test

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

extractor_test = vgg(line_test, extract=True)
extractor_test = F.average_pooling_2d(extractor_test, 3,2,1)
extractor_test.unchain_backward()

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0
    for batch in range(0,2000,batchsize):
        line_box = []
        color_box = []
        i_mask_box = []
        mask_box = []
        for index in range(batchsize):
            rnd = np.random.randint(1,Ntrain)
            filename = "trim_free_" + str(rnd) + ".png"
            color,line, image_mask, mask = prepare_refine(line_path+filename, color_path + filename)
            line_box.append(line)
            color_box.append(color)
            i_mask_box.append(image_mask)
            mask_box.append(mask)

        line = chainer.as_variable(xp.array(line_box).astype(xp.float32))
        color = chainer.as_variable(xp.array(color_box).astype(xp.float32))
        image_mask = chainer.as_variable(xp.array(i_mask_box).astype(xp.float32))
        #mask = chainer.as_variable(xp.array(mask_box).astype(xp.float32))
        mask = image_mask

        extractor = vgg(line, extract=True)
        extractor = F.average_pooling_2d(extractor,3,2,1)
        extractor.unchain_backward()

        fake = generator(line,mask,extractor)
        y_dis = discriminator(fake, extractor)
        t_dis = discriminator(color, extractor)
        dis_loss = F.mean(F.softplus(y_dis)) + F.mean(F.softplus(-t_dis))

        #rnd_x = xp.random.uniform(0,1,fake.shape).astype(xp.float32)
        #x_perturbed = chainer.as_variable(rnd_x*fake + (1.0-rnd_x)*color)

        fake.unchain_backward()

        #y_perturbed  = discriminator(x_perturbed, extractor)
        #grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
        #grad = F.sqrt(F.batch_l2_norm_squared(grad))
        #loss_grad = penalty_weight * F.mean_squared_error(grad, xp.ones_like(grad.data))

        #dis_loss += loss_grad

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        fake = generator(line, mask, extractor)
        y_dis = discriminator(fake, extractor)

        gen_loss = adver_weight * F.mean(F.softplus(-y_dis))
        #fake_feat = vgg(fake)
        #real_feat = vgg(color)
        #vgg_loss = calc_content_loss(fake_feat, real_feat)
        vgg_loss = F.mean_absolute_error(fake,color)
        gen_loss += vgg_loss
        
        vgg.cleargrads()
        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        vgg_opt.update()
        gen_loss.unchain_backward()
        
        sum_gen_loss += gen_loss.data.get()
        sum_dis_loss += dis_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("generator_test.model",generator)
            with chainer.using_config("train", False):
                y = generator(line_test,mask_test, extractor_test)
            y = y.data.get()
            sr = line_test.data.get()
            cr = color_test.data.get()
            im = image_mask_test.data.get()
            for i_ in range(testsize):
                tmp = (np.clip((sr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                tmp = (np.clip((im[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+2)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(output_dir, epoch))
                tmp = (np.clip((cr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+3)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(output_dir, epoch))
                tmp = (np.clip((y[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i_+4)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(output_dir, epoch))
    
    print("epoch : {}".format(epoch))
    print("Generator loss : {}".format(sum_gen_loss/Ntrain))
    print("Discriminator loss : {}".format(sum_dis_loss/Ntrain))