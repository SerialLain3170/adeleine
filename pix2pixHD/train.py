import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,initializers,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Global_Generator,Local_Enhancer,Discriminator

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha1=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="pix2pixHD")
parser.add_argument("--Ntrain",default=20000,type=int,help="the number of training images")
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

global_generator=Global_Generator()
global_generator.to_gpu()
gg_opt=set_optimizer(global_generator)

local_enhancer=Local_Enhancer()
local_enhancer.to_gpu()
le_opt=set_optimizer(local_enhancer)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,iterations,batchsize):
        line_box=[]
        color_box=[]
        for index in range(batchsize):


        line=chainer.as_variable(xp.array(line_box).astype(xp.float32))
        color=chainer.as_variable(xp.array(color_box).astype(xp.float32))

        color_2=F.average_pooling_2d(color,(2,2))
        color_4=F.average_pooling_2d(color_2,(2,2))

        line_2=F.average_pooling_2d(line,(2,2))
        line_4=F.average_pooling_2d(line_2,(2,2))

        gg=global_generator(line_2)
        fake=local_enhancer(line,gg)

        fake_2=F.average_pooling_2d(fake,(2,2))
        fake_4=F.average_pooling_2d(fake_2,(2,2))

        dis_fake,_=discriminator(F.concat([line,fake]))
        dis2_fake,_=discriminator(F.concat([line_2,fake_2]))
        dis4_fake,_=discriminator(F.concat([line_4,fake_4)))
        dis_color,_=discriminator(F.concat([color,line]))
        dis2_color,_=discriminator(F.concat([color_2,line_2]))
        dis4_color,_=discriminator(F.concat([color_4,line_4]))

        adver_loss=0.5*(F.sum((dis_color-1.0)**2)+F.sum(dis_fake**2))/batchsize
        adver_loss+=0.5*(F.sum((dis2_color-1.0)**2)+F.sum(dis2_color**2))/batchsize
        adver_loss+=0.5*(F.sum((dis4_color-1.0)**2)+F.sum(dis4_color**2))/batchsize

        discriminator.to_gpu()
        adver_loss.backward()
        dis_opt.update()
        adver_loss.unchain_backward()

        gg=global_generator(line_2)
        fake=local_enhancer(line,gg)
        fake_2=F.average_pooling_2d(fake,(2,2))
        fake_4=F.average_pooling_2d(fake_2,(2,2))

        dis_fake,fake_feat=discriminator(F.concat([line,fake]))
        dis2_fake,fake_feat2=discriminator(F.concat([line_2,fake_2]))
        dis4_fake,fake_feat3=discriminator(F.concat([line_4,fake_4)))
        dis_color,real_feat=discriminator(F.concat([color,line]))
        dis2_color,real_feat2=discriminator(F.concat([color_2,line_2]))
        dis4_color,real_feat3=discriminator(F.concat([color_4,line_4]))

        gen_loss=0.5*(F.sum((dis_fake-1.0)**2))/batchsize
        gen_loss+=0.5*(F.sum((dis2_fake-1.0)**2))/batchsize
        gen_loss+=0.5*(F.sum((dis4_fake-1.0)**2))/batchsize

        feat_loss=F.mean_absolute_error(real_feat,fake_feat)
        feat_loss+=F.mean_absolute_error(real_feat2,fake_feat2)
        feat_loss+=F.mean_absolute_error(real_feat3,fake_feat3)

        gen_loss+=feat_loss

        global_generator.cleargrads()
        local_enhancer.cleargrads()
        gen_loss.backward()
        gg_opt.update()
        le_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=adver_loss
        sum_gen_loss+=gen_loss

        if batch==0:
            serializers.save_npz("global_generator",global_generator)
            serializers.save_npz("local_enhancer",local_enhancer)

    print("epoch:{}".format(epoch))
    print("Discriminator loss".format(sum_dis_loss/iterations))
    print("Generator loss".format(sum_gen_loss/iterations))
