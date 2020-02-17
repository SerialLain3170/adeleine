import chainer
import chainer.links as L
import chainer.functions as F

from chainer import cuda, Chain, initializers
from spectral_norm import SNConvolution2D


class VGG(Chain):
    def __init__(self):
        super(VGG, self).__init__()

        with self.init_scope():
            self.base = L.VGG19Layers()

    def __call__(self, x, extract=False):
        if extract:
            h = self.base(x, layers=['conv4_4'])['conv4_4']
        else:
            h = self.base(x, layers=["conv2_2"])["conv2_2"]

        return h


class CBR(Chain):
    def __init__(self, in_ch, out_ch, bn=True, activ=F.relu):
        super(CBR, self).__init__()
        w = initializers.GlorotUniform()
        self.bn = bn
        self.activ = activ
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
            if bn:
                self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.bn:
            h = self.activ(self.bn0(self.c0(x)))
        else:
            h = F.relu(self.c0(x))

        return h


def pixel_shuffler(out_ch, x, r=2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0, 3, 4, 1, 5, 2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))

    return out_map


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


class SECat(Chain):
    def __init__(self, in_ch, out_ch):
        super(SECat, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.l0 = L.Linear(in_ch*2, out_ch, initialW=w, nobias=True)
            self.l1 = L.Linear(out_ch, in_ch, initialW=w, nobias=True)

    def __call__(self, x, extractor):
        batch, ch, height, width = x.shape
        x_pool = F.average_pooling_2d(x, (height, width)).reshape(batch, ch)
        extractor = F.average_pooling_2d(extractor, (height, width)).reshape(batch, ch)

        h = F.relu(self.l0(F.concat([x_pool, extractor])))
        h = F.sigmoid(self.l1(h)).reshape(batch, ch, 1, 1)
        h = F.tile(h, (1, 1, height, width))

        return h


class SECatResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(SECatResBlock, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)
            self.se = SECat(out_ch, int(out_ch/16))

    def __call__(self, x, extractor):
        h = F.relu(self.bn0(self.c0(x)))
        h = h * self.se(h, extractor)

        return h + x


class SACatResBlock(Chain):
    def __init__(self, in_ch, out_ch, bn=True, activ=F.relu):
        super(SACatResBlock, self).__init__()
        w = initializers.GlorotUniform()
        self.bn = bn
        self.activ = activ
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            if bn:
                self.bn0 = L.BatchNormalization(out_ch)
            self.sa = SACat(out_ch, out_ch)

    def __call__(self, x, extractor):
        if self.bn:
            h = self.activ(self.bn0(self.c0(x)))
        else:
            h = self.activ(self.c0(x))
        h = h * self.sa(h, extractor)

        return h + x


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        w = initializers.GlorotUniform()
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))

        return h + x


class Upsamp(Chain):
    def __init__(self, in_ch, out_ch, bn=True, activ=F.relu):
        super(Upsamp, self).__init__()
        w = initializers.GlorotUniform()
        self.out_ch = out_ch
        self.bn = bn
        self.activ = activ
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            if bn:
                self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
        if self.bn:
            h = self.activ(self.bn0(self.c0(h)))
        else:
            h = self.activ(self.c0(h))

        return h


class AttentionBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(AttentionBlock, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 1, 1, 0)
            self.c1 = L.Convolution2D(in_ch, out_ch, 1, 1, 0)
            self.c2 = L.Convolution2D(in_ch, out_ch, 1, 1, 0)

    def __call__(self, img, opt):
        batch, channel, width, height = img.shape
        query = self.c0(img).reshape(batch, channel, width * height).transpose(0, 2, 1)
        key = self.c1(opt).reshape(batch, channel, width * height)
        energy = F.matmul(query, key)
        attention = F.softmax(energy)
        value = self.c2(img).reshape(batch, channel, width * height)

        h = F.matmul(value, attention.transpose(0, 2, 1))
        h = h.reshape(batch, channel, width, height)

        return h


class GuideDecoder(Chain):
    def __init__(self, base=64, bn=True, activ=F.relu):
        super(GuideDecoder, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            self.up0 = Upsamp(base*16, base*8, bn=bn, activ=activ)
            self.up1 = Upsamp(base*8, base*4, bn=bn, activ=activ)
            self.up2 = Upsamp(base*4, base*2, bn=bn, activ=activ)
            self.up3 = Upsamp(base*2, base, bn=bn, activ=activ)
            self.cout = L.Convolution2D(base, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = self.up0(x)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.cout(h)

        return F.tanh(h)


class Generator(Chain):
    def __init__(self, base=32, layer=8):
        super(Generator, self).__init__()
        w = initializers.GlorotUniform()
        res = chainer.ChainList()
        for _ in range(layer):
            res.add_link(ResBlock(base*16, base*16))
        with self.init_scope():
            # Input layer
            self.c0 = L.Convolution2D(6, base, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(base)

            # UNet
            self.cbr0 = CBR(base, base*2)
            self.cbr1 = CBR(base*4, base*4)
            self.cbr2 = CBR(base*4, base*8)
            self.cbr3 = CBR(base*8, base*16)
            self.res = res
            self.up0 = Upsamp(base*16, base*8)
            self.up1 = Upsamp(base*16, base*4)
            self.up2 = Upsamp(base*8, base*2)
            self.up3 = Upsamp(base*4, base)

            # Output layer
            self.c1 = L.Convolution2D(base*2, 3, 3, 1, 1, initialW=w)

            # Attention Block
            self.attn = AttentionBlock(base*16, base*16)

            # Mask Feature Extractor
            self.cmask = L.Convolution2D(3, base*2, 3, 1, 1, initialW=w)
            self.bmask = L.BatchNormalization(base*2)

            self.cext1 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)
            self.cext2 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)
            self.bext1 = L.BatchNormalization(base*16)
            self.bext2 = L.BatchNormalization(base*16)

    def __call__(self, x, mask, extractor):
        # Mask Extractor
        hmask = F.relu(self.bmask(self.cmask(mask)))

        # Line Art Extractor
        hextract = F.relu(self.bext1(self.cext1(extractor)))
        hextract = F.relu(self.bext2(self.cext2(hextract)))

        # Main Stream
        h1 = F.relu(self.bn0(self.c0(x)))
        h2 = self.cbr0(h1)
        h3 = self.cbr1(F.concat([h2, hmask]))
        h4 = self.cbr2(h3)
        h5 = self.cbr3(h4)
        h = self.attn(h5, hextract)
        for i, res in enumerate(self.res.children()):
            h = res(h)
        h = self.up0(h)
        inp = F.concat([h, h4], axis=1)
        h = self.up1(inp)
        inp = F.concat([h, h3], axis=1)
        h = self.up2(inp)
        inp = F.concat([h, h2], axis=1)
        h = self.up3(inp)
        inp = F.concat([h, h1], axis=1)
        h = F.tanh(self.c1(inp))

        return h


class SAGenerator(Chain):
    def __init__(self, base=64, layer=8, attn_type="sa"):
        super(SAGenerator, self).__init__()
        w = initializers.GlorotUniform()
        res = chainer.ChainList()
        for _ in range(layer):
            if attn_type == "sa":
                res.add_link(SACatResBlock(base*16, base*16))
            elif attn_type == "se":
                res.add_link(SECatResBlock(base*16, base*16))
        with self.init_scope():
            self.c0 = L.Convolution2D(6, base, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(base)
            self.cbr0 = CBR(base, base*2)
            self.cbr1 = CBR(base*4, base*4)
            self.cbr2 = CBR(base*4, base*8)
            self.cbr3 = CBR(base*8, base*16)
            self.res = res
            self.up0 = Upsamp(base*16, base*8)
            self.up1 = Upsamp(base*16, base*4)
            self.up2 = Upsamp(base*8, base*2)
            self.up3 = Upsamp(base*4, base)
            self.c1 = L.Convolution2D(base*2, 3, 3, 1, 1, initialW=w)

            self.cmask = L.Convolution2D(3, base*2, 3, 1, 1, initialW=w)
            self.bmask = L.BatchNormalization(base*2)

            if base == 64:
                self.cext1 = L.Convolution2D(base*8, base*16, 3, 1, 1, initialW=w)
            elif base == 32:
                self.cext1 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)
            elif base == 96:
                self.cext1 = L.Convolution2D(512, base*16, 3, 1, 1, initialW=w)
            self.cext2 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)
            self.bext1 = L.BatchNormalization(base*16)
            self.bext2 = L.BatchNormalization(base*16)

    def __call__(self, x, mask, extractor):
        # Mask Extractor
        hmask = F.relu(self.bmask(self.cmask(mask)))

        # Line Art Extractor
        hextract = F.relu(self.bext1(self.cext1(extractor)))
        hextract = F.relu(self.bext2(self.cext2(hextract)))

        # Main Stream
        h1 = F.relu(self.bn0(self.c0(x)))
        h2 = self.cbr0(h1)
        h3 = self.cbr1(F.concat([h2, hmask]))
        h4 = self.cbr2(h3)
        h = self.cbr3(h4)
        for i, res in enumerate(self.res.children()):
            h = res(h, hextract)
        h = self.up0(h)
        inp = F.concat([h, h4], axis=1)
        h = self.up1(inp)
        inp = F.concat([h, h3], axis=1)
        h = self.up2(inp)
        inp = F.concat([h, h2], axis=1)
        h = self.up3(inp)
        inp = F.concat([h, h1], axis=1)
        h = F.tanh(self.c1(inp))

        return h


class SAGeneratorWithGuide(Chain):
    def __init__(self, base=64, layer=8, attn_type="sa", bn=True, activ=F.relu):
        super(SAGeneratorWithGuide, self).__init__()
        w = initializers.GlorotUniform()
        self.bn = bn
        self.activ = activ
        res = chainer.ChainList()
        for _ in range(layer):
            if attn_type == "sa":
                res.add_link(SACatResBlock(base*16, base*16, bn=bn, activ=activ))
            elif attn_type == "se":
                res.add_link(SECatResBlock(base*16, base*16, bn=bn, activ=activ))
        with self.init_scope():
            self.c0 = L.Convolution2D(6, base, 3, 1, 1, initialW=w)
            if bn:
                self.bn0 = L.BatchNormalization(base)
            self.cbr0 = CBR(base, base*2, bn=bn, activ=activ)
            self.cbr1 = CBR(base*4, base*4, bn=bn, activ=activ)
            self.cbr2 = CBR(base*4, base*8, bn=bn, activ=activ)
            self.cbr3 = CBR(base*8, base*16, bn=bn, activ=activ)
            self.res = res
            self.up0 = Upsamp(base*16, base*8, bn=bn, activ=activ)
            self.up1 = Upsamp(base*16, base*4, bn=bn, activ=activ)
            self.up2 = Upsamp(base*8, base*2, bn=bn, activ=activ)
            self.up3 = Upsamp(base*4, base, bn=bn, activ=activ)
            self.c1 = L.Convolution2D(base*2, 3, 3, 1, 1, initialW=w)

            self.cmask = L.Convolution2D(3, base*2, 3, 1, 1, initialW=w)
            if bn:
                self.bmask = L.BatchNormalization(base*2)

            self.guide = GuideDecoder()

            if base == 64:
                self.cext1 = L.Convolution2D(base*8, base*16, 3, 1, 1, initialW=w)
            elif base == 32:
                self.cext1 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)
            elif base == 96:
                self.cext1 = L.Convolution2D(512, base*16, 3, 1, 1, initialW=w)
            self.cext2 = L.Convolution2D(base*16, base*16, 3, 1, 1, initialW=w)

            if bn:
                self.bext1 = L.BatchNormalization(base*16)
                self.bext2 = L.BatchNormalization(base*16)

    def __call__(self, x, mask, extractor):
        # Mask Extractor
        if self.bn:
            hmask = self.activ(self.bmask(self.cmask(mask)))

            # Line Art Extractor
            hextract = self.activ(self.bext1(self.cext1(extractor)))
            hextract = self.activ(self.bext2(self.cext2(hextract)))

            h1 = self.activ(self.bn0(self.c0(x)))
        else:
            hmask = self.activ(self.cmask(mask))

            # Line Art Extractor
            hextract = self.activ(self.cext1(extractor))
            hextract = self.activ(self.cext2(hextract))

            h1 = self.activ(self.c0(x))

        # Main Stream
        h2 = self.cbr0(h1)
        h3 = self.cbr1(F.concat([h2, hmask]))
        h4 = self.cbr2(h3)
        h = self.cbr3(h4)
        for i, res in enumerate(self.res.children()):
            h = res(h, hextract)
        hguide = self.guide(h)
        h = self.up0(h)
        inp = F.concat([h, h4], axis=1)
        h = self.up1(inp)
        inp = F.concat([h, h3], axis=1)
        h = self.up2(inp)
        inp = F.concat([h, h2], axis=1)
        h = self.up3(inp)
        inp = F.concat([h, h1], axis=1)
        h = F.tanh(self.c1(inp))

        return h, hguide


class CBR_Dis(Chain):
    def __init__(self, in_ch, out_ch, down=True, sn=False):
        super(CBR_Dis, self).__init__()
        w = initializers.GlorotUniform()
        self.down = down
        with self.init_scope():
            if sn:
                self.cdown = SNConvolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
                self.cpara = SNConvolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)

            else:
                self.cdown = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)
                self.cpara = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        if self.down:
            h = F.leaky_relu(self.cdown(x))
        else:
            h = F.leaky_relu(self.cpara(x))

        return h


class DiscriminatorBlock(Chain):
    def __init__(self, base=64, sn=False):
        super(DiscriminatorBlock, self).__init__()
        w = initializers.GlorotUniform()
        with self.init_scope():
            if sn:
                self.c0 = SNConvolution2D(3, base, 3, 1, 1, initialW=w)
            else:
                self.c0 = L.Convolution2D(3, base, 3, 1, 1, initialW=w)

            self.cbr0 = CBR_Dis(base, base*2, sn=sn)
            self.cbr1 = CBR_Dis(base*2, base*2, sn=sn)
            self.cbr2 = CBR_Dis(base*2, base*4, sn=sn)
            self.cbr3 = CBR_Dis(base*4, base*4, down=False, sn=sn)
            self.cbr4 = CBR_Dis(base*4, base*8, sn=sn)
            self.cbr5 = CBR_Dis(base*16, base*8, down=False, sn=sn)
            self.cbr6 = CBR_Dis(base*8, base*8, down=False, sn=sn)

            if sn:
                self.cout = SNConvolution2D(base*8, 1, 1, 1, 0, initialW=w)
            else:
                self.cout = L.Convolution2D(base*8, 1, 1, 1, 0, initialW=w)

    def __call__(self, x, extractor):
        h = F.leaky_relu(self.c0(x))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        h = self.cbr5(F.concat([h, extractor]))
        h = self.cbr6(h)
        h = self.cout(h)

        return h


class Discriminator(Chain):
    def __init__(self, base=64, sn=False):
        super(Discriminator, self).__init__()
        discriminators = chainer.ChainList()
        for _ in range(3):
            discriminators.add_link(DiscriminatorBlock(sn=sn))
        with self.init_scope():
            self.dis = discriminators

    def __call__(self, x, extractor):
        adv_list = []
        for index in range(3):
            h = self.dis[index](x, extractor)
            adv_list.append(h)
            x = F.average_pooling_2d(x, 3, 2, 1)
            extractor = F.average_pooling_2d(extractor, 3, 2, 1)

        return adv_list
