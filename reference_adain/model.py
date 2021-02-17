import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
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
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_linear(content_feat: torch.Tensor,
                 style_feat: torch.Tensor) -> torch.Tensor:
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


class AdaINResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int):
        super(AdaINResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
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

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
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
                extracotr: torch.Tensor) -> torch.Tensor:
        h = self.cbr(x)
        h = h * self.se(h, extracotr)

        return h + x


class StyleEncoderVgg(nn.Module):
    def __init__(self):
        super(StyleEncoderVgg, self).__init__()

        self.vgg = Vgg19(requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x)


class StyleEncoder(nn.Module):
    def __init__(self, base=64):
        super(StyleEncoder, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, 7, 1, 3),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
            ResBlock(base*8, base*8),
            ResBlock(base*8, base*8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class StyleEncoderMLP(nn.Module):
    def __init__(self, base=64):
        super(StyleEncoderMLP, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, 7, 1, 3),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1),
            ResBlock(base*8, base*8),
            ResBlock(base*8, base*8)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(base*8, base*8),
            nn.ReLU(),
            nn.Linear(base*8, base*8),
            nn.ReLU(),
            nn.Linear(base*8, base*16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = self.pool(h).squeeze(3).squeeze(2)
        h = self.mlp(h)

        return h


class Generator(nn.Module):
    def __init__(self,
                 base=64,
                 layers=4,
                 attn_type="adain",
                 ):
        super(Generator, self).__init__()

        self.attn_type = attn_type
        self.ce = self._make_content_encoder(base)
        self.se = self._make_style_encoder(attn_type)
        self.res = self._make_reslayer(base, layers, attn_type)
        self.dec = self._make_decoder(base)
        self.out = nn.Sequential(
            nn.Conv2d(base*2, 3, 7, 1, 3),
            nn.Tanh()
        )

        init_weights(self.ce)
        init_weights(self.se)
        init_weights(self.res)
        init_weights(self.dec)
        init_weights(self.out)

    @staticmethod
    def _make_content_encoder(base: int):
        modules = []
        modules.append(CBR(3, base, 7, 1, 3))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*8, 4, 2, 1))
        modules.append(CBR(base*8, base*8, 4, 2, 1))
        modules.append(ResBlock(base*8, base*8))
        modules.append(ResBlock(base*8, base*8))

        return nn.ModuleList(modules)

    @staticmethod
    def _make_style_encoder(attn_type):
        if attn_type == "linear":
            return StyleEncoderMLP()
        else:
            return StyleEncoder()

    @staticmethod
    def _make_reslayer(base: int, layers: int, attn_type: str):
        if attn_type == "adain":
            modules = [AdaINResBlock(base*8, base*8) for _ in range(layers)]
        elif attn_type == "linear":
            modules = [AdaINMLPResBlock(base*8, base*8) for _ in range(layers)]
        if attn_type == "sa":
            modules = [SACatResBlock(base*8, base*8) for _ in range(layers)]
        if attn_type == "se":
            modules = [SECatResBlock(base*8, base*8) for _ in range(layers)]

        return nn.ModuleList(modules)

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*16, base*8, 3, 1, 1, up=True))
        modules.append(CBR(base*16, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*8, base*2, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base*2, 3, 1, 1, up=True))

        return nn.ModuleList(modules)

    def _content_encode(self, x: torch.Tensor) -> (torch.Tensor, [torch.Tensor]):
        encode_list = []
        for i, layer in enumerate(self.ce):
            x = layer(x)
            if i in [1, 2, 3, 4]:
                encode_list.append(x)

        return x, encode_list

    def _res(self,
             x: torch.Tensor,
             s: torch.Tensor) -> torch.Tensor:
        for layer in self.res:
            x = layer(x, s)

        return x

    def _decode(self,
                x: torch.Tensor,
                encode_list: List[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.dec):
            x = layer(torch.cat([x, encode_list[-i-1]], dim=1))

        return self.out(x)

    def forward(self,
                x: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        x, mid_layer_list = self._content_encode(x)
        se = self.se(s)

        x = self._res(x, se)
        x = self._decode(x, mid_layer_list)

        return x


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
