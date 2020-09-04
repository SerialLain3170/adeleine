import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,initializers,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
import requests
from model import Global_Generator,Local_Enhancer,Discriminator,VGG
from prepare import prepare_dataset_line,prepare_dataset_color
import matplotlib
matplotlib.use('Agg')

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

outdir="./output/"
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
serializers.load_npz("./global_generator_pretrain.model",global_generator)

local_enhancer=Local_Enhancer()
local_enhancer.to_gpu()
le_opt=set_optimizer(local_enhancer)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

discriminator_2=Discriminator()
discriminator_2.to_gpu()
dis2_opt=set_optimizer(discriminator_2)

discriminator_4=Discriminator()
discriminator_4.to_gpu()
dis4_opt=set_optimizer(discriminator_4)

#vgg=VGG()
#vgg.to_gpu()
#vgg_opt=set_optimizer(vgg)
#vgg.base.disable_update()

for epoch in range(epochs):
    if epoch <= 25:
        global_generator.disable_update()
    else:
        global_generator.enable_update()
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

        color_2=F.average_pooling_2d(color,3,2,1)
        color_4=F.average_pooling_2d(color_2,3,2,1)

        line_2=F.average_pooling_2d(line,3,2,1)
        line_4=F.average_pooling_2d(line_2,3,2,1)

        _,gg=global_generator(line_2)
        fake=local_enhancer(line,gg)

        fake_2=F.average_pooling_2d(fake,3,2,1)
        fake_4=F.average_pooling_2d(fake_2,3,2,1)

        dis_fake,_=discriminator(F.concat([fake,line]))
        dis2_fake,_=discriminator_2(F.concat([fake_2,line_2]))
        dis4_fake,_=discriminator_4(F.concat([fake_4,line_4]))
        dis_color,_=discriminator(F.concat([color,line]))
        dis2_color,_=discriminator_2(F.concat([color_2,line_2]))
        dis4_color,_=discriminator_4(F.concat([color_4,line_4]))

        fake.unchain_backward()
        fake_2.unchain_backward()
        fake_4.unchain_backward()

        adver_loss=F.mean(F.softplus(-dis_color)) + F.mean(F.softplus(dis_fake))
        adver_loss+=F.mean(F.softplus(-dis2_color)) + F.mean(F.softplus(dis2_fake))
        adver_loss+=F.mean(F.softplus(-dis4_color)) + F.mean(F.softplus(dis4_fake))

        discriminator.cleargrads()
        discriminator_2.cleargrads()
        discriminator_4.cleargrads()
        discriminator.to_gpu()
        adver_loss.backward()
        dis_opt.update()
        dis2_opt.update()
        dis4_opt.update()
        adver_loss.unchain_backward()

        _,gg=global_generator(line_2)
        fake=local_enhancer(line,gg)
        fake_2=F.average_pooling_2d(fake,3,2,1)
        fake_4=F.average_pooling_2d(fake_2,3,2,1)

        dis_fake,fake_feat=discriminator(F.concat([fake,line]))
        dis2_fake,fake_feat2=discriminator_2(F.concat([fake_2,line_2]))
        dis4_fake,fake_feat3=discriminator_4(F.concat([fake_4,line_4]))
        dis_color,real_feat=discriminator(F.concat([color,line]))
        dis2_color,real_feat2=discriminator_2(F.concat([color_2,line_2]))
        dis4_color,real_feat3=discriminator_4(F.concat([color_4,line_4]))

        gen_loss=F.mean(F.softplus(-dis_fake))
        gen_loss+=F.mean(F.softplus(-dis2_fake))
        gen_loss+=F.mean(F.softplus(-dis4_fake))

        feat_loss=calc_loss(real_feat,fake_feat)
        feat_loss+=calc_loss(real_feat2,fake_feat2)
        feat_loss+=calc_loss(real_feat3,fake_feat3)

        #perc_fake=vgg(fake)
        #perc_color=vgg(color)
        #percep_loss=calc_loss(perc_fake,perc_color)

        content_loss=F.mean_absolute_error(color,fake)
        content_loss+=F.mean_absolute_error(color_2,fake_2)
        content_loss+=F.mean_absolute_error(color_4,fake_4)

        gen_loss+=weight * (feat_loss+content_loss)

        global_generator.cleargrads()
        local_enhancer.cleargrads()
        #vgg.cleargrads()
        gen_loss.backward()
        gg_opt.update()
        le_opt.update()
        #vgg_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=adver_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if batch==0:
            serializers.save_npz("global_generator",global_generator)
            serializers.save_npz("local_enhancer",local_enhancer)
            with chainer.using_config("train", False):
                line_test_2=F.average_pooling_2d(line_test,(2,2))
                _,gg=global_generator(line_test_2)
                y = local_enhancer(line_test,gg)
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