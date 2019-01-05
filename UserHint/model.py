import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np
from instance_normalization import InstanceNormalization

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()

    def __call__(self,x, extract=False):
        if extract:
            h = self.base(x,layers=['conv4_3'])['conv4_3']
        else:
            h = self.base(x,layers=["conv2_2"])["conv2_2"]

        return h

class CBR(Chain):
    def __init__(self, in_ch,out_ch):
        super(CBR,self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch,out_ch,4,2,1,initialW = w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))

        return h

def pixel_shuffler(out_ch, x, r = 2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0,3,4,1,5,2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))
    
    return out_map

class ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(ResBlock,self).__init__()
        w = initializers.Normal(0.02)
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch,3,1,1,initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW = w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))

        return h + x

class Upsamp(Chain):
    def __init__(self, in_ch, out_ch, shuffler = None):
        super(Upsamp, self).__init__()
        w = initializers.Normal(0.02)
        self.shuffler = shuffler
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)
            self.bshuffe = L.BatchNormalization(int(out_ch/4))

    def __call__(self,x):
        if self.shuffler:
            h = self.c0(x)
            h = pixel_shuffler(self.out_ch, h)
            h = F.relu(self.bshuffe(h))

        else:
            h = F.unpooling_2d(x,2,2,0,cover_all=False)
            h = F.relu(self.bn0(self.c0(h)))

        return h

class Generator(Chain):
    def __init__(self, base=32):
        super(Generator, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(6,base,3,1,1,initialW = w)
            self.cbr0 = CBR(base, base*2)
            self.cbr1 = CBR(base*4, base*4)
            self.cbr2 = CBR(base*4, base*8)
            self.cbr3 = CBR(base*8, base*16)
            self.res0 = ResBlock(base*32, base*32)
            self.res1 = ResBlock(base*32, base*32)
            #self.res2 = ResBlock(base*16, base*16)
            self.up0 = Upsamp(base*32, base*8)
            self.up1 = Upsamp(base*16, base*4)
            self.up2 = Upsamp(base*8, base*2)
            self.up3 = Upsamp(base*4, base)
            self.c1 = L.Convolution2D(base*2, 3, 3,1,1,initialW = w)
            self.bn0 = L.BatchNormalization(base)

            self.cmask = L.Convolution2D(3,base*2,3,1,1,initialW=w)
            self.bmask = L.BatchNormalization(base*2)

            self.cext1 = L.Convolution2D(base*16, base*16, 3,1,1,initialW=w)
            self.cext2 = L.Convolution2D(base*16, base*16, 3,1,1,initialW=w)
            self.bext1 = L.BatchNormalization(base*16)
            self.bext2 = L.BatchNormalization(base*16)

    def __call__(self,x,mask,extractor):
        hmask = F.relu(self.bmask(self.cmask(mask)))
        hextract = F.relu(self.bext1(self.cext1(extractor)))
        hextract = F.relu(self.bext2(self.cext2(hextract)))

        h1 = F.relu(self.c0(x))
        h2 = self.cbr0(h1)
        h3 = self.cbr1(F.concat([h2,hmask]))
        h4 = self.cbr2(h3)
        h5 = self.cbr3(h4)
        h = self.res0(F.concat([h5, hextract], axis=1))
        h = self.res1(h)
        #h = self.res2(h)
        h = self.up0(h)
        inp = F.concat([h,h4],axis = 1)
        h = self.up1(inp)
        inp = F.concat([h,h3],axis = 1)
        h = self.up2(inp)
        inp = F.concat([h,h2],axis = 1)
        h = self.up3(inp)
        inp = F.concat([h,h1],axis = 1)
        h = F.tanh(self.c1(inp))

        return h

class CBR_Dis(Chain):
    def __init__(self,in_ch, out_ch,down=True):
        super(CBR_Dis, self).__init__()
        w = initializers.Normal(0.02)
        self.down = down
        with self.init_scope():
            self.cdown = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW = w)
            self.cpara = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW = w)

            #self.bn0 = L.BatchNormalization(out_ch)
            #self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.down:
            h = F.leaky_relu(self.cdown(x))
        else:
            h = F.leaky_relu(self.cpara(x))

        return h

class Discriminator(Chain):
    def __init__(self, base=64):
        super(Discriminator,self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3,1,1,initialW = w)
            self.cbr0 = CBR_Dis(base, base*2)
            self.cbr1 = CBR_Dis(base*2, base*2)
            self.cbr2 = CBR_Dis(base*2, base*4)
            self.cbr3 = CBR_Dis(base*4, base*4, down=False)
            self.cbr4 = CBR_Dis(base*4, base*8)
            self.cbr5 = CBR_Dis(base*16, base*8, down=False)
            self.cbr6 = CBR_Dis(base*8, base*8)
            self.l0 = L.Linear(None, 1, initialW = w)

            #self.bn0 = L.BatchNormalization(base)

    def __call__(self,x, extractor):
        h = F.leaky_relu(self.c0(x))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        h = self.cbr5(F.concat([h, extractor]))
        h = self.cbr6(h)
        h = self.l0(h)

        return h