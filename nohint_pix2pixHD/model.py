import torch
import numpy as np
import torch.nn as nn

from typing import List
from torchvision import models
from torch.nn import init
from torch.nn.utils import spectral_norm


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


def down_sample(x: torch.Tensor) -> torch.Tensor:
    down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    return down(x)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h]


# Basic components of Generator and Discriminator
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


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):

        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + x


class LocalEnhancer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 base=64,
                 num_layers=9):

        super(LocalEnhancer, self).__init__()

        self.enc = self._make_encoder(in_ch, base)
        self.res = self._make_res(base, num_layers)
        self.dec = self._make_decoder(base)
        self.out = nn.Sequential(
            nn.Conv2d(base, 3, 7, 1, 3),
            nn.Tanh()
        )
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        init_weights(self.enc)
        init_weights(self.dec)
        init_weights(self.res)
        init_weights(self.out)

    @staticmethod
    def _make_encoder(in_ch: int, base: int):
        modules = []
        modules.append(CBR(in_ch, base, 7, 1, 3, activ="relu"))
        modules.append(CBR(base, base*2, 4, 2, 1, activ="relu"))
        modules.append(CBR(base*2, base*4, 4, 2, 1, activ="relu"))
        modules.append(CBR(base*4, base*8, 4, 2, 1, activ="relu"))
        modules.append(CBR(base*8, base*16, 4, 2, 1, activ="relu"))

        modules = nn.ModuleList(modules)

        return modules

    @staticmethod
    def _make_res(base: int, num_layers: int):
        modules = []
        for _ in range(num_layers):
            modules.append(ResBlock(base*16, base*16))

        modules = nn.ModuleList(modules)

        return modules

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*16, base*8, 3, 1, 1, up=True, activ="relu"))
        modules.append(CBR(base*8, base*4, 3, 1, 1, up=True, activ="relu"))
        modules.append(CBR(base*4, base*2, 3, 1, 1, up=True, activ="relu"))
        modules.append(CBR(base*2, base, 3, 1, 1, up=True, activ="relu"))

        modules = nn.ModuleList(modules)

        return modules

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.enc:
            x = layer(x)

        return x

    def _res(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.res:
            x = layer(x)

        return x

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.dec:
            x = layer(x)

        return x

    def forward(self,
                x: torch.Tensor,
                pretrain: bool) -> torch.Tensor:

        x = self.down(x)
        x = self._encode(x)
        x = self._res(x)
        x = self._decode(x)

        if pretrain:
            return self.out(x)
        else:
            return x


class GlobalGenerator(nn.Module):
    def __init__(self,
                 in_ch: int,
                 base=32):
        super(GlobalGenerator, self).__init__()

        self.gen = self._make_net(in_ch, base)
        self.out = nn.Sequential(
            nn.Conv2d(base, 3, 7, 1, 3),
            nn.Tanh()
        )

        init_weights(self.gen)
        init_weights(self.out)

    @staticmethod
    def _make_net(in_ch: int, base: int):
        modules = []
        modules.append(CBR(in_ch, base, 7, 1, 3))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(ResBlock(base*2, base*2))
        modules.append(ResBlock(base*2, base*2))
        modules.append(ResBlock(base*2, base*2))
        modules.append(CBR(base*2, base, 3, 1, 1, up=True))

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor,
                le: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.gen):
            x = layer(x)
            if index == 1:
                x += le

        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self,
                 in_ch=6,
                 multi_pattern=3,
                 base=64):

        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(multi_pattern):
            self.cnns.append(self._make_nets(in_ch, base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    @staticmethod
    def _make_nets(in_ch: int, base: int):
        modules = []
        modules.append(CBR(in_ch, base, 4, 2, 1))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*8, 4, 2, 1))

        model = nn.ModuleList(modules)

        init_weights(model)

        return model

    def forward(self, x: torch.Tensor) -> (List[torch.Tensor], List[torch.Tensor]):
        feats = []
        outputs = []
        for model in self.cnns:
            h = x
            for layer in model:
                x = layer(x)
                feats.append(x)
            outputs.append(x)
            x = self.down(h)

        return feats, outputs
