import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import List
from torch.nn import init
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


def init_weights(net: nn.Module, init_type='normal'):
    net.apply(weights_init_normal)


class Vgg19(nn.Module):
    def __init__(self,
                 layer=None,
                 requires_grad=False):

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
            h = h_relu5

        return h


class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 up=False,
                 norm="in",
                 activ="lrelu"):

        super(CBR, self).__init__()

        modules = []
        modules = self._preprocess(modules, up)
        modules.append(nn.Conv2d(in_ch, out_ch, kernel, stride, pad))
        modules = self._norm(modules, norm, out_ch)
        modules = self._activ(modules, activ)

        self.cbr = nn.ModuleList(modules)

    @staticmethod
    def _preprocess(modules: List, up: bool) -> List:
        if up:
            modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))

        return modules

    @staticmethod
    def _norm(modules: List,
              norm: str,
              out_ch: int) -> List:

        if norm == "bn":
            modules.append(nn.BatchNorm2d(out_ch))
        elif norm == "in":
            modules.append(nn.BatchNorm2d(out_ch))

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


class SACat(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 latent_dim: int):

        super(SACat, self).__init__()
        self.c0 = nn.Conv2d(in_ch*2 + latent_dim, out_ch, 1, 1, 0)
        self.c1 = nn.Conv2d(out_ch, out_ch, 1, 1, 0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:

        h, w = x.size(2), x.size(3)
        z = z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        h = self.relu(self.c0(torch.cat([x, extractor, z], dim=1)))
        h = self.sigmoid(self.c1(h))

        return h


class SACatResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 latent_dim: int):

        super(SACatResBlock, self).__init__()
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn0 = nn.InstanceNorm2d(out_ch)
        self.sa = SACat(out_ch, out_ch, latent_dim)

        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:

        h = self.relu(self.bn0(self.c0(x)))
        h = h * self.sa(h, extractor, z)

        return h + x


class LatentEncoder(nn.Module):
    def __init__(self,
                 in_ch: int,
                 latent_dim: int,
                 base=64):
        super(LatentEncoder, self).__init__()

        self.down = nn.Sequential(
            CBR(in_ch, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(base*8, base*2),
            nn.ReLU(),
            nn.Linear(base*2, latent_dim)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.gap(h).squeeze(3).squeeze(2)

        return self.mlp(h)


class MaskEncoder(nn.Module):
    def __init__(self):
        super(MaskEncoder, self).__init__()

        self.vgg = Vgg19(requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)


class Generator(nn.Module):
    def __init__(self,
                 in_ch: int,
                 latent_dim: int,
                 num_layers=4,
                 base=64):
        super(Generator, self).__init__()

        self.l_dim = latent_dim

        self.mask_encoder = MaskEncoder()
        self.encoder = self._make_encoder(in_ch, base, latent_dim)
        self.resblock = self._make_resblock(num_layers, base, latent_dim)
        self.decoder = self._make_decoder(base)

        init_weights(self.mask_encoder)
        init_weights(self.encoder)
        init_weights(self.resblock)
        init_weights(self.decoder)

    @staticmethod
    def _make_encoder(in_ch: int, base: int, latent_dim: int):
        modules = []
        modules.append(CBR(in_ch + latent_dim, base, 7, 1, 3))
        modules.append(CBR(base + latent_dim, base*2, 4, 2, 1))
        modules.append(CBR(base*2 + latent_dim, base*4, 4, 2, 1))
        modules.append(CBR(base*4 + latent_dim, base*8, 4, 2, 1))
        modules.append(CBR(base*8 + latent_dim, base*8, 4, 2, 1))

        modules = nn.ModuleList(modules)

        return modules

    @staticmethod
    def _make_resblock(num_layers: int, base: int, latent_dim: int):
        modules = [SACatResBlock(base*8, base*8, latent_dim) for _ in range(num_layers)]
        modules = nn.ModuleList(modules)

        return modules

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*16, base*8, 3, 1, 1, up=True))
        modules.append(CBR(base*16, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*8, base*2, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base, 3, 1, 1, up=True))
        modules.append(nn.Conv2d(base, 3, 3, 1, 1))
        modules.append(nn.Tanh())

        return nn.ModuleList(modules)

    def _generate_noise(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        z = np.random.normal(size=(batchsize, self.l_dim))
        z = torch.cuda.FloatTensor(z)

        return z

    @staticmethod
    def _modify(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        z = z.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        x = torch.cat([x, z], axis=1)

        return x

    def _encode(self, x: torch.Tensor, z: torch.Tensor) -> List[torch.Tensor]:
        encode_list = []
        for layer in self.encoder:
            x = self._modify(x, z)
            x = layer(x)
            encode_list.append(x)

        return encode_list

    def _resblock(self, x: torch.Tensor,
                  extractor: torch.Tensor,
                  z: torch) -> torch.Tensor:

        for layer in self.resblock:
            x = layer(x, extractor, z)

        return x

    def _decode(self,
                x: torch.Tensor,
                encode_list: List[torch.Tensor]) -> torch.Tensor:

        for index, layer in enumerate(self.decoder):
            if index < 4:
                x = layer(torch.cat([x, encode_list[-index-1]], dim=1))
            else:
                x = layer(x)

        return x

    def forward(self,
                x: torch.Tensor,
                m: torch.Tensor,
                z=None) -> (torch.Tensor, torch.Tensor):

        if z is None:
            z = self._generate_noise(x)
        encode_list = self._encode(x, z)
        extractor = self.mask_encoder(m)
        enc_list_copy = copy.copy(encode_list)

        h = self._resblock(encode_list[-1], extractor, z)
        h = self._decode(h, enc_list_copy)

        return z, h


class Discriminator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 base=64,
                 multi_pattern=3):

        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(multi_pattern):
            self.cnns.append(self._make_nets(in_ch, base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    @staticmethod
    def _make_nets(in_ch: int, base: int):
        model = nn.Sequential(
            CBR(in_ch, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
            CBR(base*8, base*16, 4, 2, 1),
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
