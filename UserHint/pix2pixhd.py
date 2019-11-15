import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import cuda, Chain, initializers
from instance_normalization import InstanceNormalization

xp = cuda.cupy
cuda.get_device(0).use()


class CBR(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(CBR, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.in0 = InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.in0(self.c0(x)))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(in_ch, out_ch)
            self.cbr1 = CBR(out_ch, out_ch)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)

        return h+x


class Down(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(Down, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 2, 1, initialW=w)
            self.in0 = InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.in0(self.c0(x)))

        return h


class Up(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(Up, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.in0 = InstanceNormalization(out_ch)
            #self.in0=L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
        h = F.relu(self.in0(self.c0(h)))

        return h


class SACat(Chain):
    def __init__(self, in_ch, out_ch):
        super(SACat, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch*2, out_ch, 1, 1, 0, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 1, 1, 0, initialW=w)

    def __call__(self, x, extractor):
        h = F.relu(self.c0(F.concat([x, extractor])))
        h = F.sigmoid(self.c1(h))

        return h


class SACatResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(SACatResBlock, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)
            self.sa = SACat(out_ch, out_ch)

    def __call__(self, x, extractor):
        h = F.relu(self.bn0(self.c0(x)))
        h = h * self.sa(h, extractor)

        return h + x


class Global_Generator(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(Global_Generator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(6, base, 7, 1, 3, initialW=w)
            self.down0 = Down(base, base*2)
            self.down1 = Down(base*2, base*4)
            self.down2 = Down(base*4, base*8)
            self.down3 = Down(base*8, base*16)
            self.res0 = SACatResBlock(base*16, base*16)
            self.res1 = SACatResBlock(base*16, base*16)
            self.res2 = SACatResBlock(base*16, base*16)
            self.res3 = SACatResBlock(base*16, base*16)
            self.res4 = SACatResBlock(base*16, base*16)
            self.res5 = SACatResBlock(base*16, base*16)
            self.up0 = Up(base*16, base*8)
            self.up1 = Up(base*8, base*4)
            self.up2 = Up(base*4, base*2)
            self.up3 = Up(base*2, base)
            self.c1 = L.Convolution2D(base, 3, 7, 1, 3, initialW=w)

            self.in0 = InstanceNormalization(base)
            #self.in0=L.BatchNormalization(base)

    def __call__(self, x, ext):
        h = F.relu(self.in0(self.c0(x)))
        h = self.down0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.res0(h, ext)
        h = self.res1(h, ext)
        h = self.res2(h, ext)
        h = self.res3(h, ext)
        h = self.res4(h, ext)
        h = self.res5(h, ext)
        h = self.up0(h)
        h = self.up1(h)
        h = self.up2(h)
        hout = self.up3(h)
        h = self.c1(hout)
        h = F.tanh(h)

        return h, hout


class Local_Enhancer(Chain):
    def __init__(self, base=32):
        w = initializers.Normal(0.02)
        super(Local_Enhancer, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(6, base, 3, 1, 1, initialW=w)
            self.down0 = Down(base, base*2)
            self.res0 = SACatResBlock(base*2, base*2)
            self.res1 = SACatResBlock(base*2, base*2)
            self.res2 = SACatResBlock(base*2, base*2)
            self.res3 = SACatResBlock(base*2, base*2)
            self.up0 = Up(base*2, base)
            self.c1 = L.Convolution2D(base, 3, 7, 1, 3, initialW=w)

            self.cord = L.Convolution2D(base*16, base*2, 1, 1, 0, initialW=w)

            self.in0 = InstanceNormalization(base)
            #self.in0=L.BatchNormalization(base)

    def __call__(self, x, gn, ext):
        ext = self.cord(ext)

        h = F.relu(self.in0(self.c0(x)))
        h = self.down0(h)
        h = self.res0(h+gn, ext)
        h = self.res1(h, ext)
        h = self.res2(h, ext)
        h = self.res3(h, ext)
        h = self.up0(h)
        h = self.c1(h)

        return h


class CIL(Chain):
    def __init__(self, in_ch, out_ch):
        super(CIL, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
            self.in0 = InstanceNormalization(out_ch)

    def __call__(self, x):
        return F.leaky_relu(self.in0(self.c0(x)))


class DiscriminatorBlock(Chain):
    def __init__(self, base=64):
        super(DiscriminatorBlock, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = CIL(6, base)
            self.c1 = CIL(base, base*2)
            self.c2 = CIL(base*2, base*4)
            self.c3 = CIL(base*4, base*8)
            self.c4 = L.Convolution2D(base*8, 1, 1, 1, 0, initialW=w)

    def __call__(self, x):
        h1 = self.c0(x)
        h2 = self.c1(h1)
        h3 = self.c2(h2)
        h4 = self.c3(h3)
        h = self.c4(h4)

        return h, [h1, h2, h3, h4]


class MSDiscriminator(Chain):
    def __init__(self, base=64):
        super(MSDiscriminator, self).__init__()
        discriminators = chainer.ChainList()
        for _ in range(3):
            discriminators.add_link(DiscriminatorBlock())
        with self.init_scope():
            self.dis = discriminators

    def __call__(self, x):
        adv_list = []
        con_list = []
        for index in range(3):
            h, h_list = self.dis[index](x)
            adv_list.append(h)
            con_list.append(h_list)
            x = F.average_pooling_2d(3, 2, 1)

        return adv_list, con_list


class VGG(Chain):
    def __init__(self):
        super(VGG, self).__init__()

        with self.init_scope():
            self.base = L.VGG19Layers()

    def __call__(self, x):
        h = self.base(x, layers=["conv4_4"])["conv4_4"]

        return [h]
