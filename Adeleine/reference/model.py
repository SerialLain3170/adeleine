import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.nn import init
from torchvision import models
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


def calc_mean_std(feat: torch.Tensor, eps=1e-5) -> (torch.Tensor, torch.Tensor):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_linear(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = style_feat[:, :512], style_feat[:, 512:]
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
            h = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return h


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


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 norm="in"):

        super(ResBlock, self).__init__()

        if norm == "bn":
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch)
        )

        elif norm == "in":
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + x


class AdaINResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(AdaINResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.c0(x)
        h = self.relu(adain(h, z))
        h = self.c1(h)
        h = self.relu(adain(h, z))

        return h + x


class AdaINMLPResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(AdaINMLPResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.c0(x)
        h = self.relu(adain_linear(h, z))
        h = self.c1(h)
        h = self.relu(adain_linear(h, z))

        return h + x


class SACat(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(SACat, self).__init__()
        self.c0 = nn.Conv2d(in_ch*2, out_ch, 1, 1, 0)
        self.c1 = nn.Conv2d(out_ch, out_ch, 1, 1, 0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.c0(torch.cat([x, extractor], dim=1)))
        h = self.sigmoid(self.c1(h))

        return h


class SECat(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(SECat, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_ch*2, out_ch, bias=False),
            nn.ReLU(),
            nn.Linear(out_ch, in_ch),
            nn.Sigmoid()
        )

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        batch, ch = x.size(0), x.size(1)
        x_pool = self.avg_pool(x).view(batch, ch)
        extractor = self.avg_pool(extractor).view(batch, ch)
        h = self.se(torch.cat([x_pool, extractor], dim=1)).view(batch, ch, 1, 1)

        return h.expand_as(x)


class SACatResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(SACatResBlock, self).__init__()
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(out_ch)
        self.sa = SACat(out_ch, out_ch)

        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn0(self.c0(x)))
        h = h * self.sa(h, extractor)

        return h + x


class SECatResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(SECatResBlock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.se = SECat(out_ch, int(out_ch/16))

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.cbr(x)
        h = h * self.se(h, extractor)

        return h + x


class SCFT(nn.Module):
    def __init__(self, base=512):
        super(SCFT, self).__init__()

        self.cq = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.ck = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.cv = nn.Conv2d(base, base, 1, 1, 0)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = base ** 0.5

    def forward(self,
                vs: torch.Tensor,
                vr: torch.Tensor) -> torch.Tensor:
        batch, ch, height, width = vs.size()
        h_q = self.cq(vs).view(batch, int(ch/8), height * width).permute(0, 2, 1)
        h_k = self.ck(vr).view(batch, int(ch/8), height * width)
        energy = torch.bmm(h_q, h_k) / self.scale
        attention = self.softmax(energy).permute(0, 2, 1)
        h_v = self.cv(vr).view(batch, ch, height * width)

        h = torch.bmm(h_v, attention)
        h = h.view(batch, ch, height, width)

        return h + vs


class Generator(nn.Module):
    def __init__(self,
                 scft_base=64):

        super(Generator, self).__init__()

        self.pool2 = nn.AvgPool2d(2, 2, 0)
        self.pool4 = nn.AvgPool2d(8, 4, 2)

        self.se = self._make_encoder(base=scft_base)
        self.ce = self._make_encoder(base=scft_base)
        mid_base = scft_base * (16 + 8 + 4)

        self.scft = SCFT(mid_base)
        self.res = self._make_reslayer(mid_base)

        self.dec = self._make_decoder(base=scft_base)

        self.out = nn.Sequential(
            nn.Conv2d(scft_base*2, 3, 7, 1, 3),
            nn.Tanh()
        )

        init_weights(self.ce)
        init_weights(self.se)
        init_weights(self.res)
        init_weights(self.scft)
        init_weights(self.dec)
        init_weights(self.out)

    @staticmethod
    def _make_encoder(base: int):
        modules = []
        modules.append(CBR(3, base, 3, 1, 1))
        modules.append(CBR(base, base, 3, 1, 1))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*2, 3, 1, 1))
        modules.append(CBR(base*2, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*4, 3, 1, 1))
        modules.append(CBR(base*4, base*8, 4, 2, 1))
        modules.append(CBR(base*8, base*8, 3, 1, 1))
        modules.append(CBR(base*8, base*16, 4, 2, 1))
        modules.append(CBR(base*16, base*16, 3, 1, 1))

        return nn.ModuleList(modules)

    @staticmethod
    def _make_reslayer(base: int):
        modules = []
        modules.append(ResBlock(base, base))
        modules.append(ResBlock(base, base))
        modules.append(ResBlock(base, base))
        modules.append(ResBlock(base, base))

        return nn.ModuleList(modules)

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*(16+8+4), base*8, 3, 1, 1, up=True))
        modules.append(CBR(base*16, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*8, base*2, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base*2, 3, 1, 1, up=True))

        return nn.ModuleList(modules)

    def _content_encode(self,
                        x: torch.Tensor) -> (torch.Tensor, List[torch.Tensor]):
        encode_list = []
        final_list = []
        for i, layer in enumerate(self.ce):
            x = layer(x)
            if i == 3:
                encode_list.append(x)
            elif i == 5:
                encode_list.append(x)
                final_list.append(self.pool4(x))
            elif i == 7:
                encode_list.append(x)
                final_list.append(self.pool2(x))
            elif i == 9:
                final_list.append(x)

        x = torch.cat(final_list, dim=1)

        return x, encode_list

    def _style_encode(self, x: torch.Tensor) -> torch.Tensor:
        final_list = []
        for i, layer in enumerate(self.ce):
            x = layer(x)
            if i == 5:
                final_list.append(self.pool4(x))
            elif i == 7:
                final_list.append(self.pool2(x))
            elif i == 9:
                final_list.append(x)

        x = torch.cat(final_list, dim=1)

        return x

    def _res(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.res:
            x = layer(x)

        return x

    def _decode(self,
                x: torch.Tensor,
                encode_list: List[torch.Tensor]) -> torch.Tensor:
        for index, layer in enumerate(self.dec):
            if index in [1, 2, 3]:
                x = layer(torch.cat([x, encode_list[-index]], dim=1))
            else:
                x = layer(x)

        return self.out(x)

    def forward(self,
                x: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        ce, mid_layer_list = self._content_encode(x)
        se = self._style_encode(s)

        h = self.scft(ce, se)
        h = self._res(h)
        h = self._decode(h, mid_layer_list)

        return h


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(3):
            self.cnns.append(self._make_nets(base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_nets(self, base: int):
        model = nn.Sequential(
            CBR(3, base, 4, 2, 1),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
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
