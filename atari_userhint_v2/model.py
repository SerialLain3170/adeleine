import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.nn import init
from torchvision import models
from torch.nn.utils import spectral_norm


class EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride=1,
        pad=0,
        bias=True,
        groups=1
    ):
        super(EqualizedConv2d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel, kernel)
        )
        self.scale = 1 / math.sqrt(in_ch * kernel ** 2)

        self.stride = stride
        self.pad = pad
        self.groups = groups

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.pad,
            groups=self.groups
        )

        return out


class GroupConv(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 groups: int):
        super(GroupConv, self).__init__()

        self.c0 = nn.Conv2d(in_ch,
                            out_ch,
                            kernel,
                            stride,
                            pad,
                            groups=groups)

        scale = 1 / math.sqrt(in_ch * kernel ** 2)
        torch.nn.init.normal_(self.c0.weight.data, std=scale)

    def forward(self, x):
        return self.c0(x)


class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias=True,
        bias_init=0,
        lr_mul=1,
    ):
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_ch, in_ch).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_ch)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out


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


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = torch.hub.load('RF5/danbooru-pretrained', 'resnet34')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = self.up(x)

        return x


# Basic components of encoder or decoder
class CBR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int,
                 stride: int,
                 pad: int,
                 up=False,
                 norm="no",
                 activ="lrelu",
                 sn=False,
                 bias=True):

        super(CBR, self).__init__()

        modules = []
        modules = self._preprocess(modules, up)
        modules = self._conv(modules, in_ch, out_ch, kernel, stride, pad, sn, bias)
        modules = self._norm(modules, norm, out_ch)
        modules = self._activ(modules, activ)

        self.cbr = nn.ModuleList(modules)

    @staticmethod
    def _preprocess(modules: List, up: bool) -> List:
        if up:
            modules.append(nn.UpsamplingBilinear2d(scale_factor=2))

        return modules

    @staticmethod
    def _conv(modules: List,
              in_ch: int,
              out_ch: int,
              kernel: int,
              stride: int,
              pad: int,
              sn: bool,
              bias=True) -> List:
        if sn:
            modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, kernel, stride, pad, bias=bias)))
        else:
            modules.append(EqualizedConv2d(in_ch, out_ch, kernel, stride, pad, bias=bias))

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
            CBR(in_ch, out_ch, 3, 1, 1),
            CBR(out_ch, out_ch, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + x


class DownResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 down=True,
                 sn=False):
        super(DownResBlock, self).__init__()

        self.res = nn.Sequential(
            CBR(in_ch, out_ch, 4, 2, 1, sn=sn),
            CBR(out_ch, out_ch, 3, 1, 1, sn=sn)
        )
        self.relu = nn.LeakyReLU()
        self.down = down
        self.downsample = nn.AvgPool2d(3, 2, 1)

        if sn:
            self.c_sc = spectral_norm(EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False))
        else:
            self.c_sc = EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def _skip(self, x: torch.Tensor) -> torch.Tensor:
        if self.down:
            x = self.downsample(x)

        return self.relu(self.c_sc(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self._skip(x)
        x = self.res(x)

        return x + skip


class SEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 sn=False):
        super(SEBlock, self).__init__()

        if sn:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                spectral_norm(EqualizedConv2d(in_ch, int(in_ch/16), 1, 1, 0, bias=False)),
                nn.ReLU(),
                spectral_norm(EqualizedConv2d(int(in_ch/16), in_ch, 1, 1, 0, bias=False)),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                EqualizedConv2d(in_ch, int(in_ch/16), 1, 1, 0, bias=False),
                nn.ReLU(),
                EqualizedConv2d(int(in_ch/16), in_ch, 1, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class SEResNeXtBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 cardinality: int,
                 sn=False,
                 norm="in",
                 activ="lrelu"):

        super(SEResNeXtBlock, self).__init__()

        modules = []
        modules.append(CBR(in_ch, out_ch, 1, 1, 0, sn=sn, bias=False))
        modules = self._groupconv(modules, out_ch, out_ch, cardinality, sn)
        modules = self._norm(modules, out_ch, norm)
        modules = self._activ(modules, activ)
        modules = self._conv(modules, out_ch, out_ch, sn)
        modules.append(SEBlock(out_ch, sn=sn))
        self.xblock = nn.ModuleList(modules)

        if in_ch != out_ch:
            if sn:
                self.shortcut = spectral_norm(EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False))
            else:
                self.shortcut = EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False)

        self.relu = nn.LeakyReLU()
        self.match = in_ch == out_ch

    @staticmethod
    def _groupconv(modules: List, in_ch: int, out_ch: int, cardinality: int, sn: bool):
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, groups=cardinality)
        scale = 1 / math.sqrt(in_ch * 3 ** 2)
        torch.nn.init.normal_(conv.weight.data, std=scale)
        if sn:
            modules.append(spectral_norm(conv))
        else:
            modules.append(conv)

        return modules

    @staticmethod
    def _conv(modules: List, in_ch: int, out_ch: int, sn: bool):
        if sn:
            modules.append(spectral_norm(EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False)))
        else:
            modules.append(EqualizedConv2d(in_ch, out_ch, 1, 1, 0, bias=False))

        return modules

    @staticmethod
    def _norm(modules: List, out_ch: int, norm: str):
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

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.match:
            return x
        else:
            return self.shortcut(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self._shortcut(x)
        for layer in self.xblock:
            x = layer(x)
        x = x + skip
        x = self.relu(x)

        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 cardinality: int,
                 layers: int,
                 ):
        super(UpBlock, self).__init__()

        self.flag = layers != 1

        if layers != 1:
            modules = [SEResNeXtBlock(in_ch, out_ch, cardinality=cardinality)]
            for _ in range(layers - 1):
                modules.append(SEResNeXtBlock(out_ch, out_ch, cardinality=cardinality))
            modules.append(nn.Upsample(scale_factor=2, mode="bilinear"))

            self.up = nn.ModuleList(modules)

        else:
            self.up = nn.Sequential(
                SEResNeXtBlock(in_ch, out_ch, cardinality=cardinality),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )

    def forward(self, x):
        if self.flag:
            for layer in self.up:
                x = layer(x)

        else:
            x = self.up(x)

        return x


class SACat(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SACat, self).__init__()
        self.c0 = EqualizedConv2d(in_ch*2, out_ch, 1, 1, 0, bias=False)
        self.c1 = EqualizedConv2d(out_ch, out_ch, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.c0(torch.cat([x, extractor], dim=1)))
        h = self.sigmoid(self.c1(h))

        return h


class SACatResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SACatResBlock, self).__init__()
        self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
        self.sa = SACat(out_ch, out_ch)

        self.relu = nn.LeakyReLU()

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.c0(x))
        h = h * self.sa(h, extractor)

        return h + x


class SACatResNeXtBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(SACatResNeXtBlock, self).__init__()
        self.seresnext = SEResNeXtBlock(in_ch, out_ch, 32)
        self.sa = SACat(out_ch, out_ch)

    def forward(self,
                x: torch.Tensor,
                extractor: torch.Tensor) -> torch.Tensor:
        h = self.seresnext(x)
        h = h * self.sa(h, extractor)

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
        modules.append(CBR(in_ch, base, 3, 1, 1))
        modules.append(DownResBlock(base, base*2))
        modules.append(DownResBlock(base*2, base*4))
        modules.append(DownResBlock(base*4, base*8))
        modules.append(DownResBlock(base*8, base*8))

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


class AtariEncoder(nn.Module):
    def __init__(self,
                 encoder_type="i2v",
                 mid_ch=512,
                 out_ch=512):
        super(AtariEncoder, self).__init__()

        self.encoder_type = encoder_type

        if encoder_type == "i2v":
            model = caffemodel2pytorch.Net(
                prototxt='./illustration2vec_PyTorch/illustration2vec/illust2vec_tag.prototxt',
                weights='./illustration2vec_PyTorch/illustration2vec/illust2vec_tag_ver200.caffemodel',
                caffe_proto='https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
            )

            for param in model.parameters():
                param.requires_grad = False

            del model.relu5_2
            del model.pool5
            del model.conv6_1
            del model.relu6_1
            del model.conv6_2
            del model.relu6_2
            del model.conv6_3
            del model.relu6_3
            del model.conv6_4
            del model.pool6

            self.i2v = model

            self.enc = nn.Sequential(
                CBR(mid_ch, out_ch, 1, 1, 0, bias=False),
                CBR(out_ch, out_ch, 3, 1, 1, activ="lrelu")
            )
        elif encoder_type == "vgg":
            model = Vgg19(requires_grad=False)
            self.enc = nn.Sequential(
                model,
                CBR(mid_ch, out_ch, 1, 1, 0, bias=False),
                ResBlock(out_ch, out_ch),
                ResBlock(out_ch, out_ch)
            )
        elif encoder_type == "res":
            self.enc = nn.Sequential(
                ResNet(),
                CBR(mid_ch, out_ch, 1, 1, 0, bias=False),
                ResBlock(out_ch, out_ch),
                ResBlock(out_ch, out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.repeat(1, 3, 1, 1)
        if self.encoder_type == "i2v":
            x = self.i2v(x)["conv5_2"]

        return self.enc(x)


class GuideDecoder(nn.Module):
    def __init__(self, base=64):
        super(GuideDecoder, self).__init__()

        self.decoder = self._make_decoder(base)
        self.out_layer = nn.Sequential(
            EqualizedConv2d(base, 3, 3, 1, 1),
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
    def __init__(self,
                 base=64,
                 resnext=False,
                 up_layers=[10, 5, 5, 3]):
        super(Decoder, self).__init__()

        self.decoder = self._make_decoder(base, resnext, up_layers)
        self.out_layer = nn.Sequential(
            EqualizedConv2d(base, base, 3, 1, 1),
            nn.InstanceNorm2d(base),
            nn.LeakyReLU(),
            EqualizedConv2d(base, 3, 3, 1, 1),
            nn.Tanh()
        )

    @staticmethod
    def _make_decoder(base: int, resnext: bool, up_layers: list):
        modules = []

        if resnext:
            modules.append(UpBlock(base*16, base*8, cardinality=16, layers=up_layers[0]))
            modules.append(UpBlock(base*16, base*4, cardinality=16, layers=up_layers[1]))
            modules.append(UpBlock(base*8, base*2, cardinality=16, layers=up_layers[2]))
            modules.append(UpBlock(base*4, base, cardinality=16, layers=up_layers[3]))
        else:
            modules.append(CBR(base*16, base*8, 3, 1, 1, up=True))
            modules.append(CBR(base*16, base*4, 3, 1, 1, up=True))
            modules.append(CBR(base*8, base*2, 3, 1, 1, up=True))
            modules.append(CBR(base*4, base, 3, 1, 1, up=True))

        modules = nn.ModuleList(modules)

        return modules

    def forward(self,
                x: torch.Tensor,
                mid_layer_list: List[torch.Tensor]) -> torch.Tensor:
        for index, layer in enumerate(self.decoder):
            x = layer(torch.cat([x, mid_layer_list[-index-1]], dim=1))

        return self.out_layer(x)


class Generator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 base=64,
                 num_layers=10,
                 up_layers=[10, 5, 5, 3],
                 guide=False,
                 resnext=False,
                 encoder_type="i2v"):
        super(Generator, self).__init__()

        self.ce = ContentEncoder(in_ch=in_ch, base=base)
        self.se = AtariEncoder(out_ch=base*8,
                               encoder_type=encoder_type)
        self.res = self._make_reslayer(base, num_layers, resnext)
        self.dec = Decoder(base=base, resnext=resnext, up_layers=up_layers)
        self.guide = guide

        if guide:
            self.g_dec1 = GuideDecoder(base=base)
            self.g_dec2 = GuideDecoder(base=base)

    @staticmethod
    def _make_reslayer(base: int, num_layers: int, resnext: bool):
        if resnext:
            modules = [SACatResNeXtBlock(base*8, base*8) for _ in range(num_layers)]
        else:
            modules = [SACatResBlock(base*8, base*8) for _ in range(num_layers)]

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


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 sn=False):
        super(DiscriminatorBlock, self).__init__()

        self.block = nn.Sequential(
            SEResNeXtBlock(in_ch, in_ch, cardinality=16, sn=sn, norm="no"),
            SEResNeXtBlock(in_ch, in_ch, cardinality=16, sn=sn, norm="no"),
            CBR(in_ch, out_ch, 4, 2, 1, sn=sn, norm="no")
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self,
                 in_ch=3,
                 multi_pattern=3,
                 base=64,
                 sn=False,
                 resnext=False,
                 patch=True,
                 ):

        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(multi_pattern):
            self.cnns.append(self._make_nets(in_ch, base, sn, resnext, patch))
        self.fcs = self._make_fcs(in_ch, base, sn)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.patch = patch

    @staticmethod
    def _make_fcs(in_ch: int, base: int, sn: bool):
        modules = []
        if sn:
            modules.append(spectral_norm(EqualizedLinear(base*16*8*8, 1)))
            modules.append(spectral_norm(EqualizedLinear(base*16*4*4, 1)))
            modules.append(spectral_norm(EqualizedLinear(base*16*2*2, 1)))
        else:
            modules.append(EqualizedLinear(base*16*8*8, 1))
            modules.append(EqualizedLinear(base*16*4*4, 1))
            modules.append(EqualizedLinear(base*16*2*2, 1))

        return nn.ModuleList(modules)

    @staticmethod
    def _make_nets(in_ch: int, base: int, sn: bool, resnext: bool, patch: bool):
        modules = []

        if resnext:
            modules.append(CBR(in_ch, base, 4, 2, 1, sn=sn, norm="no"))
            modules.append(CBR(base, base*2, 4, 2, 1, sn=sn, norm="no"))
            modules.append(DiscriminatorBlock(base*2, base*4, sn=sn))
            modules.append(DiscriminatorBlock(base*4, base*8, sn=sn))
            modules.append(DiscriminatorBlock(base*8, base*16, sn=sn))
            modules.append(SEResNeXtBlock(base*16, base*16, cardinality=16, sn=sn, norm="no"))
        else:
            modules.append(CBR(in_ch, base, 4, 2, 1, sn=sn, norm="no"))
            modules.append(CBR(base, base*2, 4, 2, 1, sn=sn, norm="no"))
            modules.append(CBR(base*2, base*4, 4, 2, 1, sn=sn, norm="no"))
            modules.append(CBR(base*4, base*8, 4, 2, 1, sn=sn, norm="no"))

        if patch:
            modules.append(spectral_norm(EqualizedConv2d(base*16, 1, 1, 1, 0, bias=False)))

        return nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> (List[torch.Tensor], List[torch.Tensor]):
        outputs = []
        mid_points = []
        for p, model in enumerate(self.cnns):
            h = x
            for i, layer in enumerate(model):
                x = layer(x)
                if i < 6:
                    mid_points.append(x)

            if not self.patch:
                x = x.view(x.size(0), -1)
                x = self.fcs[p](x)

            outputs.append(x)
            x = self.down(h)

        return mid_points, outputs
