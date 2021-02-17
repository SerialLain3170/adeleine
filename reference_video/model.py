import torch
import torch.nn as nn

from typing import List
from blocks import GatedConv, GatedDeconv
from torch.nn import init
from torch.nn.utils import spectral_norm
from torchvision import models


# Initialization of model
def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net: nn.Module):
    net.apply(weights_init_normal)


def calc_mean_std(feat: torch.Tensor, eps=1e-5) -> (torch.Tensor, torch.Tensor):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adain(content_feat: torch.Tensor,
          style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = style_feat[:, :256], style_feat[:, 256:]
    style_mean = style_mean.unsqueeze(2).unsqueeze(3)
    style_std = style_std.unsqueeze(2).unsqueeze(3)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, layer=None):
        super(Vgg19, self).__init__()
        self.layer = layer

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        if layer == 'four':
            self.slice = nn.Sequential()
            for x in range(21):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        elif layer == 'five':
            self.slice = nn.Sequential()
            for x in range(30):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        else:
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(2):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(7, 12):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 21):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(21, 36):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer == 'four':
            h = self.slice(x)

        elif self.layer == 'five':
            h = self.slice(x)

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# Basic components of generator and discriminator
class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 up=False,
                 norm="in",
                 activ="lrelu",
                 sn=False):

        super(CBR, self).__init__()

        modules = []
        modules = self._preprocess(modules, up)
        modules = self._conv(modules, in_ch, out_ch, kernel, stride, pad, sn)
        modules = self._norm(modules, norm, out_ch)
        modules = self._activ(modules, activ)

        self.cbr = nn.ModuleList(modules)

    @staticmethod
    def _preprocess(modules: List, up: bool) -> List:
        if up:
            modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        return modules

    @staticmethod
    def _conv(modules: List,
              in_ch: int,
              out_ch: int,
              kernel: int,
              stride: int,
              pad: int,
              sn: bool) -> List:
        if sn:
            modules.append(spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, pad)))
        else:
            modules.append(nn.Conv2d(in_ch, out_ch, kernel, stride, pad))

        return modules

    @staticmethod
    def _norm(modules: List,
              norm: str,
              out_ch: int) -> List:

        if norm == "bn":
            modules.append(nn.BatchNorm2d(out_ch))
        elif norm == "in":
            modules.append(nn.InstanceNorm2d(out_ch))

        return modules

    @staticmethod
    def _activ(modules: List, activ: str) -> List:
        if activ == "relu":
            modules.append(nn.ReLU())
        elif activ == "lrelu":
            modules.append(nn.LeakyReLU())

        return modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.cbr:
            x = layer(x)

        return x


class AdaINResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(AdaINResBlock, self).__init__()
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        h = self.c0(x)
        h = self.relu(adain(h, z))
        h = self.c1(h)
        h = adain(h, z)

        return h + x


class Encoder(nn.Module):
    def __init__(self, base=64):
        super(Encoder, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, 3, 1, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.dec = nn.Sequential(
            CBR(base*4, base*4, 3, 1, 1),
            CBR(base*4, base*2, 3, 1, 1, up=True),
            CBR(base*2, base, 3, 1, 1, up=True),
            nn.Conv2d(base, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(x)


class LatentDecoder(nn.Module):
    def __init__(self, base=256):
        super(LatentDecoder, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(base, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(x)


class Embedder(nn.Module):
    def __init__(self, base=64):
        super(Embedder, self).__init__()

        self.embed0 = nn.Sequential(
            CBR(6, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
        )

        self.embed1 = nn.Sequential(
            CBR(6, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(base*8, base*8),
            nn.ReLU(),
            nn.Linear(base*8, base*8),
            nn.ReLU()
        )

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor) -> torch.Tensor:
        h0 = self.embed0(x0)
        h0 = self.avg(h0).squeeze(3).squeeze(2)
        h1 = self.embed0(x1)
        h1 = self.avg(h1).squeeze(3).squeeze(2)
        h = (h0 + h1) / 2
        h = self.fc(h)

        return h


class ColorTransformLayer(nn.Module):
    def __init__(self, base=256):
        super(ColorTransformLayer, self).__init__()

        self.encd = Encoder()
        self.encd0 = Encoder()
        self.encd1 = Encoder()
        self.encc0 = Encoder()
        self.encc1 = Encoder()

        self.simdis0_c0 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1),
            nn.Sigmoid()
        )

        self.simdis0_c1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1),
            nn.Sigmoid()
        )

        self.simdis1_c0 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1),
            nn.Sigmoid()
        )
        self.simdis1_c1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, 1, 1),
            nn.Sigmoid()
        )

        self.pos0_c = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.posd_c0 = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.pos1_c = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.posd_c1 = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)

        self.ch0_c = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.ch1_c = nn.Conv2d(base, int(base/8), 1, 1, 0)

        self.cout = nn.Conv2d(int(base/8), base, 1, 1, 0)

    def _all_enc(self, d, d0, d1, r0, r1):
        fd = self.encd(d)
        fd0 = self.encd0(d0)
        fd1 = self.encd1(d1)
        fc0 = self.encc0(r0)
        fc1 = self.encc1(r1)

        return fd, fd0, fd1, fc0, fc1

    def _sim0(self, fd0, fd):
        h = torch.cat([fd0, fd], dim=1)
        m = self.simdis0_c0(h)
        n = self.simdis0_c1(h)

        return m, n

    def _sim1(self, fd1, fd):
        h = torch.cat([fd1, fd], dim=1)
        m = self.simdis1_c0(h)
        n = self.simdis1_c1(h)

        return m, n

    def _positional_attention0(self, fd0, fd):
        batch, ch, height, width = fd0.size()
        hd = self.pos0_c(fd).view(batch, int(ch/8), height * width)
        hd0 = self.posd_c0(fd0).view(batch, int(ch/8), height * width).permute(0, 2, 1)

        return torch.bmm(hd0, hd)

    def _positional_attention1(self, fd1, fd):
        batch, ch, height, width = fd1.size()
        hd = self.pos0_c(fd).view(batch, int(ch/8), height * width)
        hd1 = self.posd_c0(fd1).view(batch, int(ch/8), height * width).permute(0, 2, 1)

        return torch.bmm(hd1, hd)

    def _channel_attention0(self, fc0, m0):
        batch, ch, height, width = fc0.size()
        fmy0 = fc0 * m0
        h = self.ch0_c(fmy0).view(batch, int(ch/8), height * width)

        return h, fmy0

    def _channel_attention1(self, fc1, m1):
        batch, ch, height, width = fc1.size()
        fmy1 = fc1 * m1
        h = self.ch1_c(fmy1).view(batch, int(ch/8), height * width)

        return h, fmy1

    def _trans0(self, fm, n0, fmy0):
        hmy0 = fmy0 * n0
        hm0 = fm * (1 - n0)

        return hm0 + hmy0

    def _trans1(self, fm, n1, fmy1):
        hmy1 = fmy1 * n1
        hm1 = fm * (1 - n1)

        return hm1 + hmy1

    def forward(self, d, d0, d1, r0, r1):
        fd, fd0, fd1, fc0, fc1 = self._all_enc(d, d0, d1, r0, r1)
        batch, ch, height, width = fd.size()

        m0, n0 = self._sim0(fd0, fd)
        m1, n1 = self._sim1(fd1, fd)

        pa0 = self._positional_attention0(fd0, fd)
        pa1 = self._positional_attention0(fd1, fd)
        pa = torch.cat([pa0, pa1], dim=1)
        pa = self.softmax(pa)

        ca0, fmy0 = self._channel_attention0(fc0, m0)
        ca1, fmy1 = self._channel_attention1(fc1, m1)
        ca = torch.cat([ca0, ca1], dim=2)

        fm = torch.bmm(ca, pa).view(batch, int(ch/8), height, width)
        fm = self.cout(fm)

        fm0 = self._trans0(fm, n0, fmy0)
        fm1 = self._trans1(fm, n1, fmy1)

        return (fm0 + fm1) / 2


class ColorTransformNetwork(nn.Module):
    def __init__(self, layers=8, base=256):
        super(ColorTransformNetwork, self).__init__()

        self.enc = Encoder()
        self.embed = Embedder()
        self.ctl = ColorTransformLayer()
        self.ld0 = LatentDecoder()
        self.ld1 = LatentDecoder()
        self.dec = Decoder()

        res = [AdaINResBlock(base, base) for _ in range(layers)]
        self.adain_res = nn.ModuleList(res)

        init_weights(self.enc)
        init_weights(self.embed)
        init_weights(self.ctl)
        init_weights(self.ld0)
        init_weights(self.ld1)
        init_weights(self.dec)
        init_weights(self.adain_res)

    def forward(self, x, xr0, xr1, d, d0, d1, r0, r1):
        h = self.enc(x)
        hsim = self.ctl(d, d0, d1, r0, r1)
        x0 = torch.cat([xr0, r0], dim=1)
        x1 = torch.cat([xr1, r1], dim=1)
        hembed = self.embed(x0, x1)

        h = h + hsim
        ysim = self.ld0(h)

        for res in self.adain_res:
            h = res(h, hembed)

        ymid = self.ld1(h)
        y = self.dec(h)

        return y, ysim, ymid


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(3, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
            nn.Conv2d(base*8, 1, 1, 1, 0)
        )

        init_weights(self.dis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dis(x)


class TemporalConstraintNetwork(nn.Module):
    def __init__(self, base=64):
        super(TemporalConstraintNetwork, self).__init__()

        self.tcs = nn.Sequential(
            GatedConv(6, base, 3, 1, 1),
            GatedConv(base, base*2, 4, 2, 1),
            GatedConv(base*2, base*4, 4, 2, 1),
            GatedConv(base*4, base*4, 3, 1, 1, dilation=1),
            GatedConv(base*4, base*4, 3, 1, 2, dilation=2),
            GatedConv(base*4, base*4, 3, 1, 4, dilation=4),
            GatedConv(base*4, base*4, 3, 1, 8, dilation=8),
            GatedDeconv(base*4, base*2, 3, 1, 1, scale_factor=2),
            GatedDeconv(base*2, base, 3, 1, 1, scale_factor=2),
            GatedConv(base, 6, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcs(x)

        return h


class TemporalDiscriminator(nn.Module):
    def __init__(self, base=64):
        super(TemporalDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            GatedConv(6, base, 4, 2, 1),
            GatedConv(base, base*2, 4, 2, 1),
            GatedConv(base*2, base*4, 4, 2, 1),
            GatedConv(base*4, base*8, 4, 2, 1),
            GatedConv(base*8, base*8, 4, 2, 1),
            GatedConv(base*8, 1, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dis(x)
