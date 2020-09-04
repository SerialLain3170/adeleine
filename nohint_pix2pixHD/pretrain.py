import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,initializers,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Global_Generator,Local_Enhancer,Discriminator
from prepare import prepare_dataset_line,prepare_dataset_color

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def calc_loss(fake,real):
    loss = 0
    for f,r in zip(fake,real):
        _,c,h,w=f.shape
        loss+=F.mean_absolute_error(f,r) / (c*h*w)

    return loss

def calc_content_loss(fake,real):
    _,c,h,w=fake.shape
    loss=F.mean_absolute_error(fake,real) / (c*h*w)

    return loss

parser=argparse.ArgumentParser(description="pix2pixHD")
parser.add_argument("--Ntrain",default=19000,type=int,help="the number of training images")
parser.add_argument("--epochs",default=10000,type=int,help="the numbef of epochs")
parser.add_argument("--batchsize",default=16,type=int,help="batch size")
parser.add_argument("--testsize",default=2,type=int,help="test size")
parser.add_argument("--weight",default=10.0,type=float,help="the weight of feature mathcing loss")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iteration")

args=parser.parse_args()
Ntrain=args.Ntrain
epochs=args.epochs
batchsize=args.batchsize
testsize=args.testsize
weight=args.weight
iterations=args.iterations

outdir="./output_pretrain/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

line_path="/line/"
color_path="/color/"
line_box = []
color_box = []
for index in range(testsize):
    rnd = np.random.randint(Ntrain+1, Ntrain+ 400)
    filename = "trim_free_" + str(rnd) + ".png"
    line,rnd1,rnd2 = prepare_dataset_line(line_path + filename)
    color = prepare_dataset_color(color_path + filename,rnd1,rnd2)
    line_box.append(line)
    color_box.append(color)

line_test = xp.array(line_box).astype(xp.float32)
line_test = chainer.as_variable(line_test)
color_test = xp.array(color_box).astype(xp.float32)
color_test = chainer.as_variable(color_test)

global_generator=Global_Generator()
global_generator.to_gpu()
gg_opt=set_optimizer(global_generator)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

discriminator_2=Discriminator()
discriminator_2.to_gpu()
dis2_opt=set_optimizer(discriminator_2)

discriminator_4=Discriminator()
discriminator_4.to_gpu()
dis4_opt=set_optimizer(discriminator_4)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,batchsize):
        line_box=[]
        color_box=[]
        for index in range(batchsize):
            rnd = np.random.randint(1,Ntrain)
            filename = "trim_free_" + str(rnd) + ".png"
            line,rnd1,rnd2 = prepare_dataset_line(line_path + filename)
            color = prepare_dataset_color(color_path + filename,rnd1,rnd2)
            line_box.append(line)
            color_box.append(color)

        line=chainer.as_variable(xp.array(line_box).astype(xp.float32))
        color=chainer.as_variable(xp.array(color_box).astype(xp.float32))
        line=F.average_pooling_2d(line,3,stride=2,pad=1)
        color=F.average_pooling_2d(color,3,stride=2,pad=1)

        color_2=F.average_pooling_2d(color,3,2,1)
        color_4=F.average_pooling_2d(color_2,3,2,1)

        line_2=F.average_pooling_2d(line,3,2,1)
        line_4=F.average_pooling_2d(line_2,3,2,1)

        fake,_=global_generator(line)
        fake_2=F.average_pooling_2d(fake,3,2,1)
        fake_4=F.average_pooling_2d(fake_2,3,2,1)

        dis_fake,_=discriminator(F.concat([line,fake]))
        dis2_fake,_=discriminator_2(F.concat([line_2,fake_2]))
        dis4_fake,_=discriminator_4(F.concat([line_4,fake_4]))
        dis_color,_=discriminator(F.concat([line,color]))
        dis2_color,_=discriminator_2(F.concat([line_2,color_2]))
        dis4_color,_=discriminator_4(F.concat([line_4,color_4]))

        fake.unchain_backward()
        fake_2.unchain_backward()
        fake_4.unchain_backward()

        # LSGAN
        #adver_loss=0.5*(F.sum((dis_color-1.0)**2)+F.sum(dis_fake**2))/batchsize
        #adver_loss+=0.5*(F.sum((dis2_color-1.0)**2)+F.sum(dis2_fake**2))/batchsize
        #adver_loss+=0.5*(F.sum((dis4_color-1.0)**2)+F.sum(dis4_fake**2))/batchsize

        # DCGAN
        adver_loss = F.mean(F.softplus(-dis_color)) + F.mean(F.softplus(dis_fake))
        adver_loss+=F.mean(F.softplus(-dis2_color)) + F.mean(F.softplus(dis2_fake))
        adver_loss+=F.mean(F.softplus(-dis4_color)) + F.mean(F.softplus(dis4_fake))

        discriminator.cleargrads()
        discriminator_2.cleargrads()
        discriminator_4.cleargrads()
        adver_loss.backward()
        dis_opt.update()
        dis2_opt.update()
        dis4_opt.update()
        adver_loss.unchain_backward()

        fake,_=global_generator(line)
        fake_2=F.average_pooling_2d(fake,3,2,1)
        fake_4=F.average_pooling_2d(fake_2,3,2,1)

        dis_fake,fake_feat=discriminator(F.concat([line,fake]))
        dis2_fake,fake_feat2=discriminator_2(F.concat([line_2,fake_2]))
        dis4_fake,fake_feat3=discriminator_4(F.concat([line_4,fake_4]))
        _,real_feat=discriminator(F.concat([line,color]))
        _,real_feat2=discriminator_2(F.concat([line_2,color_2]))
        _,real_feat3=discriminator_4(F.concat([line_4,color_4]))

        # LSGAN
        #gen_loss=0.5*(F.sum((dis_fake-1.0)**2))/batchsize
        #gen_loss+=0.5*(F.sum((dis2_fake-1.0)**2))/batchsize
        #gen_loss+=0.5*(F.sum((dis4_fake-1.0)**2))/batchsize

        # DCGAN
        gen_loss=F.mean(F.softplus(-dis_fake))
        gen_loss+=F.mean(F.softplus(-dis2_fake))
        gen_loss+=F.mean(F.softplus(-dis4_fake))

        content_loss=calc_content_loss(fake,color)
        content_loss+=calc_content_loss(fake_2,color_2)
        content_loss+=calc_content_loss(fake_4,color_4)

        feat_loss=calc_loss(real_feat,fake_feat)
        feat_loss+=calc_loss(real_feat2,fake_feat2)
        feat_loss+=calc_loss(real_feat3,fake_feat3)

        gen_loss+=weight * (content_loss+feat_loss)

        global_generator.cleargrads()
        gen_loss.backward()
        gg_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=adver_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if batch==0:
            serializers.save_npz("global_generator_pretrain.model",global_generator)
            with chainer.using_config("train", False):
                line_test_2=F.average_pooling_2d(line_test,3,stride=2,pad=1)
                y,_=global_generator(line_test_2)
            y = y.data.get()
            sr = line_test.data.get()
            cr = color_test.data.get()
            for i_ in range(testsize):
                tmp = (np.clip((sr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((cr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+2)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((y[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+3)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("Generator loss:{}".format(sum_gen_loss/iterations))