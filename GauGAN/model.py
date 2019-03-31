import chainer
import chainer.links as L
import chainer.functions as F
import chainer.distributions as D
from chainer import cuda, Chain, initializers
import numpy as np
from instance_normalization import InstanceNormalization
from sn import SNConvolution2D, SNLinear

xp = cuda.cupy
cuda.get_device(0).use()

class SPADE(Chain):
    def __init__(self, out_ch):
        super(SPADE, self).__init__()
        w = initializers.GlorotUniform()
        self.eps = 1e-5
        with self.init_scope():
            self.c0 = SNConvolution2D(3, 128, 3,1,1,initialW=w)
            self.cw = SNConvolution2D(128, out_ch, 3,1,1,initialW=w)
            self.cb = SNConvolution2D(128, out_ch, 3,1,1,initialW=w)

    def __call__(self, x, c):
        mu = F.average(x, axis=0).reshape(1, x.shape[1], x.shape[2], x.shape[3])
        sigma = F.average((x-F.tile(mu ,(x.shape[0],1,1,1)))**2, axis=0)
        x_hat = (x-F.tile(mu,(x.shape[0],1,1,1)))/F.sqrt(F.tile(sigma+self.eps,(x.shape[0],1,1,1)))

        h = F.relu(self.c0(c))
        w = self.cw(h)
        b = self.cb(h)
        h = w * x_hat + b

        return h

class SPADEResblk(Chain):
    def __init__(self, in_ch, out_ch):
        super(SPADEResblk, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.spade0 = SPADE(in_ch)
            self.c0 = SNConvolution2D(in_ch, in_ch, 3,1,1,initialW=w)
            self.spade1 = SPADE(in_ch)
            self.c1 = SNConvolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.spade_sc = SPADE(in_ch)
            self.c_sc = SNConvolution2D(in_ch, out_ch, 3,1,1,initialW=w)

    def __call__(self, x, c):
        h = self.c0(F.relu(self.spade0(x, c)))
        h = self.c1(F.relu(self.spade1(h, c)))
        h_sc = self.c_sc(F.relu(self.spade_sc(x,c)))
        h = h + h_sc
        h = F.unpooling_2d(h,2,2,0,cover_all=False)

        return h

class CIL(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(CIL, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(in_ch, out_ch, 4,2,1,initialW=w)
            self.in0 = InstanceNormalization(out_ch)

    def __call__(self, x):
        h = F.leaky_relu(self.in0(self.c0(x)))

        return h

class Encoder(Chain):
    def __init__(self, base=64):
        w = initializers.GlorotUniform()
        super(Encoder, self).__init__()
        with self.init_scope():
            self.cil0 = CIL(3, base)
            self.cil1 = CIL(base, base*2)
            self.cil2 = CIL(base*2, base*4)
            self.cil3 = CIL(base*4, base*8)
            self.cil4 = CIL(base*8, base*8)
            self.l0 = L.Linear(None, 256)
            self.l1 = L.Linear(None, 256)

    def __call__(self, x):
        h = self.cil0(x)
        h = self.cil1(h)
        h = self.cil2(h)
        h = self.cil3(h)
        h = self.cil4(h)
        mu = self.l0(h)
        sigma = self.l1(h)

        return D.Normal(mu, log_scale=sigma)

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.encoder = Encoder()
            self.l0 = L.Linear(256, 1024*7*7)
            self.res0 = SPADEResblk(1024, 512)
            self.res1 = SPADEResblk(512, 512)
            self.res2 = SPADEResblk(512, 256)
            self.res3 = SPADEResblk(256, 128)
            self.res4 = SPADEResblk(128, 64)
            self.c0 = L.Convolution2D(64, 3,3,1,1,initialW=w)

    def __call__(self, x, c):
        h_z = self.encoder(x)
        h = h_z.sample(1)
        h = self.l0(h, n_batch_axes=2)
        h = F.reshape(h, (x.shape[0], 1024, 7, 7))
        h = self.res0(h, F.average_pooling_2d(c, 32, 32))
        h = self.res1(h, F.average_pooling_2d(c, 16, 16))
        h = self.res2(h, F.average_pooling_2d(c, 8, 8))
        h = self.res3(h, F.average_pooling_2d(c, 4, 4))
        h = self.res4(h, F.average_pooling_2d(c, 2, 2))
        h = self.c0(h)

        return F.tanh(h), h_z

class Discriminator(Chain):
    def __init__(self,base=64):
        w=initializers.GlorotUniform()
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.c0=SNConvolution2D(6,base,4,2,1,initialW=w)
            self.c1=SNConvolution2D(base,base*2,4,2,1,initialW=w)
            self.c2=SNConvolution2D(base*2,base*4,4,2,1,initialW=w)
            self.c3=SNConvolution2D(base*4,base*8,4,2,1,initialW=w)
            self.c4=L.Convolution2D(base*8 ,1,3,1,1,initialW=w)

            self.in1=InstanceNormalization(base*2)
            #self.in1=L.BatchNormalization(base*2)
            self.in2=InstanceNormalization(base*4)
            #self.in2=L.BatchNormalization(base*4)
            self.in3=InstanceNormalization(base*8)
            self.in4 = InstanceNormalization(base*8)
            #self.in3=L.BatchNormalization(base*8)

    def __call__(self,x):
        h1=F.leaky_relu(self.c0(x))
        h2=F.leaky_relu(self.in1(self.c1(h1)))
        h3=F.leaky_relu(self.in2(self.c2(h2)))
        h4=F.leaky_relu(self.in3(self.c3(h3)))
        h = F.leaky_relu(self.in4(h4))
        h=self.c4(h4)

        return h, [h1,h2,h3,h4]

class Prior(chainer.Link):

    def __init__(self):
        super(Prior, self).__init__()

        self.loc = xp.zeros(256, xp.float32)
        self.scale = xp.ones(256, xp.float32)

    def __call__(self):
        return D.Normal(self.loc, scale=self.scale)