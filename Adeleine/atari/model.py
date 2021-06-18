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


# AdaIN modules
def calc_mean_std(feat: torch.Tensor,
                  eps=1e-5) -> (torch.Tensor, torch.Tensor):
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
                 style_feat: torch.Tensor,
                 sep_dim: int) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = style_feat[:, :sep_dim], style_feat[:, sep_dim:]
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
            for x in range(27):
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
            h = self.slice5(h_relu4)

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
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, in_ch: int, out_ch: int):
        super(SACatResBlock, self).__init__()
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn0 = nn.InstanceNorm2d(out_ch)
        self.sa = SACat(out_ch, out_ch)

        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn0(self.c0(x)))
        h = h * self.sa(h, extractor)

        return h + x


class SECatResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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


# Main components
class ContentEncoder(nn.Module):
    def __init__(self,
                 in_ch=3,
                 base=64):
        super(ContentEncoder, self).__init__()

        self.encoder = self._make_encoder(in_ch, base)
        self.res = nn.Sequential(
            ResBlock(base*8, base*8),
            ResBlock(base*8, base*8)
        )

    @staticmethod
    def _make_encoder(in_ch: int, base: int):
        modules = []
        modules.append(CBR(in_ch, base, 7, 1, 3))
        modules.append(CBR(base, base*2, 4, 2, 1))
        modules.append(CBR(base*2, base*4, 4, 2, 1))
        modules.append(CBR(base*4, base*8, 4, 2, 1))
        modules.append(CBR(base*8, base*8, 4, 2, 1))

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor) -> (torch.Tensor, List[torch.Tensor]):

        mid_layer_list = []
        for layer in self.encoder:
            x = layer(x)
            mid_layer_list.append(x)

        h = self.res(x)

        return h, mid_layer_list


class StyleEncoderVgg(nn.Module):
    def __init__(self):
        super(StyleEncoderVgg, self).__init__()

        self.vgg = Vgg19(requires_grad=False)

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


class GuideDecoder(nn.Module):
    def __init__(self, base=64):
        super(GuideDecoder, self).__init__()

        self.decoder = self._make_decoder(base)
        self.out_layer = nn.Sequential(
            nn.Conv2d(base, 3, 3, 1, 1),
            nn.Tanh()
        )

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*8, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base*2, 3, 1, 1, up=True))
        modules.append(CBR(base*2, base, 3, 1, 1, up=True))

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)

        return self.out_layer(x)


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.decoder = self._make_decoder(base)
        self.out_layer = nn.Sequential(
            nn.Conv2d(base*2, 3, 7, 1, 3),
            nn.Tanh()
        )

    @staticmethod
    def _make_decoder(base: int):
        modules = []
        modules.append(CBR(base*16, base*8, 3, 1, 1, up=True))
        modules.append(CBR(base*16, base*4, 3, 1, 1, up=True))
        modules.append(CBR(base*8, base*2, 3, 1, 1, up=True))
        modules.append(CBR(base*4, base*2, 3, 1, 1, up=True))

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor,
                mid_layer_list: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.decoder):
            x = layer(torch.cat([x, mid_layer_list[-index-1]], dim=1))

        return self.out_layer(x)


class Generator(nn.Module):
    def __init__(self,
                 in_ch=6,
                 base=64,
                 num_layers=4,
                 attn_type="sa",
                 guide=False):

        super(Generator, self).__init__()

        self.ce = ContentEncoder(in_ch=in_ch)
        self.se = self._make_style_encoder(attn_type)
        self.res = self._make_reslayer(attn_type, base, num_layers)
        self.dec = Decoder()
        self.guide = guide

        init_weights(self.ce)
        init_weights(self.se)
        init_weights(self.res)
        init_weights(self.dec)

        if guide:
            self.g_dec1 = GuideDecoder()
            self.g_dec2 = GuideDecoder()
            init_weights(self.g_dec1)
            init_weights(self.g_dec2)

    @staticmethod
    def _make_style_encoder(attn_type: str) -> nn.Module:
        if attn_type == "linear":
            model = StyleEncoderMLP()
        else:
            model = StyleEncoderVgg()

        return model

    @staticmethod
    def _make_reslayer(attn_type: str, base: int, num_layers: int):
        if attn_type == "adain":
            modules = [AdaINResBlock(base*8, base*8) for _ in range(num_layers)]
        elif attn_type == "linear":
            modules = [AdaINMLPResBlock(base*8, base*8) for _ in range(num_layers)]
        elif attn_type == "sa":
            modules = [SACatResBlock(base*8, base*8) for _ in range(num_layers)]
        elif attn_type == "se":
            modules = [SECatResBlock(base*8, base*8) for _ in range(num_layers)]

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor,
                style: torch.Tensor) -> torch.Tensor:
        ce, mid_layer_list = self.ce(x)
        se = self.se(style)

        if self.guide:
            g1 = self.g_dec1(ce)

        for layer in self.res:
            ce = layer(ce, se)

        if self.guide:
            g2 = self.g_dec1(ce)

        h = self.dec(ce, mid_layer_list)

        if self.guide:
            return h, g1, g2
        else:
            return h


class Discriminator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 multi_pattern=3,
                 base=64):

        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(multi_pattern):
            self.cnns.append(self._make_nets(in_ch, base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    @staticmethod
    def _make_nets(in_ch: int, base: int):
        model = nn.Sequential(
            CBR(in_ch, base, 4, 2, 1, sn=True),
            CBR(base, base*2, 4, 2, 1, sn=True),
            CBR(base*2, base*4, 4, 2, 1, sn=True),
            CBR(base*4, base*8, 4, 2, 1, sn=True),
            spectral_norm(nn.Conv2d(base*8, 1, 1, 1, 0))
        )

        return model

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for model in self.cnns:
            h = model(x)
            outputs.append(h)
            x = self.down(x)

        return outputs
