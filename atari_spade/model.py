import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.nn import init
from torch.nn.utils import spectral_norm
from torchvision import models


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
            h = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return h


# Basic components of encoder or decoder
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


class SPADE(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):

        super(SPADE, self).__init__()

        self.norm = nn.InstanceNorm2d(out_ch, affine=False)

        self.c_share = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.c_gamma = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.c_beta = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:

        norm = self.norm(x)

        c = F.interpolate(c, size=x.size()[2:], mode="bilinear")
        c = self.c_share(c)

        gamma = self.c_gamma(c)
        beta = self.c_beta(c)

        return norm * (1 + gamma) + beta


class SPADEResBlk(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, shortcut=False):
        super(SPADEResBlk, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        self.norm0 = SPADE(in_ch, in_ch)
        self.norm1 = SPADE(in_ch, out_ch)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        if shortcut:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
            self.norm_sc = SPADE(in_ch, in_ch)

        self.shortcut = shortcut

    def _shortcut(self,
                  x: torch.Tensor,
                  c: torch.Tensor) -> torch.Tensor:

        if self.shortcut:
            out = self.c_sc(self.relu(self.norm_sc(x, c)))
        else:
            out = x

        return out

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:

        x_sc = self._shortcut(x, c)

        h = self.c0(self.relu(self.norm0(x, c)))
        h = self.c1(self.relu(self.norm1(h, c)))

        h = h + x_sc
        h = self.up(h)

        return h


class LineEncoder(nn.Module):
    def __init__(self,
                 in_ch: int,
                 base=64):
        super(LineEncoder, self).__init__()

        self.enc = self._make_encoder(in_ch, base)

    @staticmethod
    def _make_encoder(in_ch: int, base: int):
        modules = []
        modules.append(CBR(in_ch, base, 7, 1, 3))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*8, 4, 2, 1))
        modules.append(CBR(base*8, base*8, 4, 2, 1))
        modules.append(CBR(base*8, base*16, 4, 2, 1))

        return nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        enc_list = []
        for layer in self.enc:
            x = layer(x)
            enc_list.append(x)

        return enc_list


class Decoder(nn.Module):
    def __init__(self, base=64, latent_dim=128):
        super(Decoder, self).__init__()

        self.l0 = nn.Linear(latent_dim, 4 * 4 * base * 16)
        self.dec = self._make_decoder(base)
        self.cout = nn.Sequential(
            nn.Conv2d(base, 3, 7, 1, 3),
            nn.Tanh()
        )

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(SPADEResBlk(base*16, base*8, shortcut=True))
        modules.append(SPADEResBlk(base*8, base*8))
        modules.append(SPADEResBlk(base*8, base*4, shortcut=True))
        modules.append(SPADEResBlk(base*4, base*4))
        modules.append(SPADEResBlk(base*4, base*2, shortcut=True))
        modules.append(SPADEResBlk(base*2, base*2))
        modules.append(SPADEResBlk(base*2, base, shortcut=True))

        return nn.ModuleList(modules)

    def forward(self,
                z: torch.Tensor,
                enc_list: List[torch.Tensor]) -> torch.Tensor:

        batch = z.size(0)
        h = self.l0(z).view(batch, 1024, 4, 4)

        for index, layer in enumerate(self.dec):
            h = layer(h, enc_list[-index-1])

        h = self.cout(h)

        return h


class Generator(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int):
        super(Generator, self).__init__()

        self.enc = LineEncoder(in_ch)
        self.dec = Decoder(latent_dim=latent_dim)

        init_weights(self.enc)
        init_weights(self.dec)

    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:

        enc_list = self.enc(x)
        h = self.dec(z, enc_list)

        return h


class Discriminator(nn.Module):
    def __init__(self,
                 base=64,
                 multi_patterns=3):

        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(multi_patterns):
            self.cnns.append(self._make_nets(base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    @staticmethod
    def _make_nets(base: int):
        model = nn.Sequential(
            CBR(3, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*16, 4, 2, 1),
            CBR(base*16, base*16, 4, 2, 1),
            nn.Conv2d(base*16, 1, 1, 1, 0)
        )

        init_weights(model)

        return model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for model in self.cnns:
            h = model(x)
            outputs.append(h)
            x = self.down(x)

        return outputs
