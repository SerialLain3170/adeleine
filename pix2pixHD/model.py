import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda,Chain,initializers
import numpy as np
from instance_normalization import InstanceNormalization

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self,in_ch,out_ch):
        w=initializers.Normal(0.02)
        super(CBR,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.in0=InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        h=F.relu(self.in0(self.c0(x)))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(in_ch,out_ch)
            self.cbr1=CBR(out_ch,out_ch)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)

        return h+x

class Down(Chain):
    def __init__(self,in_ch,out_ch):
        w=initializers.Normal(0.02)
        super(Down,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(in_ch,out_ch,3,2,1,initialW=w)
            self.in0=InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        h=F.relu(self.in0(self.c0(x)))

        return h

class Up(Chain):
    def __init__(self,in_ch,out_ch):
        w=initializers.Normal(0.02)
        super(Up,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.in0=InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        h=F.unpooling_2d(x,2,2,0,cover_all=False)
        h=F.relu(self.in0(self.c0(h)))

        return h

class Global_Generator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Global_Generator,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(3,base,7,1,3,initialW=w)
            self.down0=Down(base,base*2)
            self.down1=Down(base*2,base*4)
            self.down2=Down(base*4,base*8)
            self.down3=Down(base*8,base*16)
            self.res0=ResBlock(base*16,base*16)
            self.res1=ResBlock(base*16,base*16)
            self.res2=ResBlock(base*16,base*16)
            self.res3=ResBlock(base*16,base*16)
            self.res4=ResBlock(base*16,base*16)
            self.res5=ResBlock(base*16,base*16)
            self.up0=Up(base*16,base*8)
            self.up1=Up(base*8,base*4)
            self.up2=Up(base*4,base*2)
            self.up3=Up(base*2,base)
            self.c1=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.in0=InstanceNormalization(base)
            #self.in0=L.BatchNormalization(base)

    def __call__(self,x):
        h=F.relu(self.in0(self.c0(x)))
        h=self.down0(h)
        h=self.down1(h)
        h=self.down2(h)
        h=self.down3(h)
        h=self.res0(h)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(h)
        h=self.up0(h)
        h=self.up1(h)
        h=self.up2(h)
        hout=self.up3(h)
        h=self.c1(hout)
        h=F.tanh(h)

        return h,hout

class Local_Enhancer(Chain):
    def __init__(self,base=32):
        w=initializers.Normal(0.02)
        super(Local_Enhancer,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(3,base,3,1,1,initialW=w)
            self.down0=Down(base,base*2)
            self.res0=ResBlock(base*2,base*2)
            self.res1=ResBlock(base*2,base*2)
            self.res2=ResBlock(base*2,base*2)
            self.res3=ResBlock(base*2,base*2)
            self.up0=Up(base*2,base)
            self.c1=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.in0=InstanceNormalization(base)
            #self.in0=L.BatchNormalization(base)

    def __call__(self,x,gn):
        h=F.relu(self.in0(self.c0(x)))
        h=self.down0(h)
        h=self.res0(h+gn)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.up0(h)
        h=self.c1(h)

        return h

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.c0=L.Convolution2D(6,base,4,2,1,initialW=w)
            self.c1=L.Convolution2D(base,base*2,4,2,1,initialW=w)
            self.c2=L.Convolution2D(base*2,base*4,4,2,1,initialW=w)
            self.c3=L.Convolution2D(base*4,base*8,4,2,1,initialW=w)
            self.c4=L.Linear(None,1,initialW=w)

            self.in1=InstanceNormalization(base*2)
            #self.in1=L.BatchNormalization(base*2)
            self.in2=InstanceNormalization(base*4)
            #self.in2=L.BatchNormalization(base*4)
            self.in3=InstanceNormalization(base*8)
            #self.in3=L.BatchNormalization(base*8)

    def __call__(self,x):
        h1=F.leaky_relu(self.c0(x))
        h2=F.leaky_relu(self.in1(self.c1(h1)))
        h3=F.leaky_relu(self.in2(self.c2(h2)))
        h4=F.leaky_relu(self.in3(self.c3(h3)))
        h=self.c4(h4)

        return h,[h1,h2,h3,h4]

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()

    def __call__(self,x):
        h = self.base(x,layers=["conv4_3"])["conv4_3"]

        return [h]