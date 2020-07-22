import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torchvision import models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adain_linear(content_feat, style_feat):
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

    def forward(self, x):
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


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, up=False):
        super(CBR, self).__init__()

        if up:
            self.cbr = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        else:
            self.cbr = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        h = self.cbr(x)

        return h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, x):
        return self.res(x) + x


class AdaINResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AdaINResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, z):
        h = self.c0(x)
        h = self.relu(adain(h, z))
        h = self.c1(h)
        h = self.relu(adain(h, z))

        return h + x


class AdaINMLPResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AdaINMLPResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, z):
        h = self.c0(x)
        h = self.relu(adain_linear(h, z))
        h = self.c1(h)
        h = self.relu(adain_linear(h, z))

        return h + x


class SACat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SACat, self).__init__()
        self.c0 = nn.Conv2d(in_ch*2, out_ch, 1, 1, 0)
        self.c1 = nn.Conv2d(out_ch, out_ch, 1, 1, 0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, extractor):
        h = self.relu(self.c0(torch.cat([x, extractor], dim=1)))
        h = self.sigmoid(self.c1(h))

        return h


class SECat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SECat, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_ch*2, out_ch, bias=False),
            nn.ReLU(),
            nn.Linear(out_ch, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x, extractor):
        batch, ch = x.size(0), x.size(1)
        x_pool = self.avg_pool(x).view(batch, ch)
        extractor = self.avg_pool(extractor).view(batch, ch)
        h = self.se(torch.cat([x_pool, extractor], dim=1)).view(batch, ch, 1, 1)

        return h.expand_as(x)


class SACatResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SACatResBlock, self).__init__()
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(out_ch)
        self.sa = SACat(out_ch, out_ch)

        self.relu = nn.ReLU()

    def forward(self, x, extractor):
        h = self.relu(self.bn0(self.c0(x)))
        h = h * self.sa(h, extractor)

        return h + x


class SECatResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SECatResBlock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.se = SECat(out_ch, int(out_ch/16))

    def forward(self, x, extracotr):
        h = self.cbr(x)
        h = h * self.se(h, extracotr)

        return h + x


class ContentEncoder(nn.Module):
    def __init__(self, base=64):
        super(ContentEncoder, self).__init__()

        self.c0 = CBR(3, base, 7, 1, 3)
        self.c1 = CBR(base, base*2, 4, 2, 1)
        self.c2 = CBR(base*2, base*4, 4, 2, 1)
        self.c3 = CBR(base*4, base*8, 4, 2, 1)
        self.c4 = CBR(base*8, base*8, 4, 2, 1)

        self.res = nn.Sequential(
            ResBlock(base*8, base*8),
            ResBlock(base*8, base*8)
        )

    def forward(self, x):
        mid_layer_list = []
        h0 = self.c0(x)
        h1 = self.c1(h0)
        mid_layer_list.append(h1)
        h2 = self.c2(h1)
        mid_layer_list.append(h2)
        h3 = self.c3(h2)
        mid_layer_list.append(h3)
        h4 = self.c4(h3)
        mid_layer_list.append(h4)
        h = self.res(h4)

        return h, mid_layer_list


class StyleEncoderVgg(nn.Module):
    def __init__(self):
        super(StyleEncoderVgg, self).__init__()

        self.vgg = Vgg19(requires_grad=True)

    def forward(self, x):
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

    def forward(self, x):
        return self.enc(x)


class SCFTEncoder(nn.Module):
    def __init__(self, base=16):
        super(SCFTEncoder, self).__init__()

        self.c0 = CBR(3, base, 3, 1, 1)
        self.c1 = CBR(base, base, 3, 1, 1)
        self.c2 = CBR(base, base*2, 4, 2, 1)
        self.c3 = CBR(base*2, base*2, 3, 1, 1)
        self.c4 = CBR(base*2, base*4, 4, 2, 1)
        self.c5 = CBR(base*4, base*4, 3, 1, 1)
        self.c6 = CBR(base*4, base*8, 4, 2, 1)
        self.c7 = CBR(base*8, base*8, 3, 1, 1)
        self.c8 = CBR(base*8, base*16, 4, 2, 1)
        self.c9 = CBR(base*16, base*16, 3, 1, 1)

        self.pool2 = nn.AvgPool2d(2, 2, 0)
        self.pool4 = nn.AvgPool2d(8, 4, 2)

    def forward(self, x):
        mid_layer_list = []

        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h1 = self.c3(h)
        h = self.c4(h1)
        h2 = self.c5(h)
        h = self.c6(h2)
        h3 = self.c7(h)
        h = self.c8(h3)
        h4 = self.c9(h)

        mid_layer_list.append(h1)
        mid_layer_list.append(h2)
        mid_layer_list.append(h3)

        h2 = self.pool4(h2)
        h3 = self.pool2(h3)
        h = torch.cat([h2, h3, h4], dim=1)

        return h, mid_layer_list


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

    def forward(self, x):
        h = self.enc(x)
        h = self.pool(h).squeeze(3).squeeze(2)
        h = self.mlp(h)

        return h


class SCFT(nn.Module):
    def __init__(self, base=512):
        super(SCFT, self).__init__()

        self.cq = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.ck = nn.Conv2d(base, int(base/8), 1, 1, 0)
        self.cv = nn.Conv2d(base, base, 1, 1, 0)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = base ** 0.5

    def forward(self, vs, vr):
        batch, ch, height, width = vs.size()
        h_q = self.cq(vs).view(batch, int(ch/8), height * width).permute(0, 2, 1)
        h_k = self.ck(vr).view(batch, int(ch/8), height * width)
        energy = torch.bmm(h_q, h_k) / self.scale
        attention = self.softmax(energy).permute(0, 2, 1)
        h_v = self.cv(vr).view(batch, ch, height * width)

        h = torch.bmm(h_v, attention)
        h = h.view(batch, ch, height, width)

        return h + vs


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.c0 = CBR(base*16, base*8, 3, 1, 1, up=True)
        self.c1 = CBR(base*16, base*4, 3, 1, 1, up=True)
        self.c2 = CBR(base*8, base*2, 3, 1, 1, up=True)
        self.c3 = CBR(base*4, base*2, 3, 1, 1, up=True)
        self.out = nn.Sequential(
            nn.Conv2d(base*2, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x, mid_layer_list):
        h = self.c0(torch.cat([x, mid_layer_list[-1]], dim=1))
        h = self.c1(torch.cat([h, mid_layer_list[-2]], dim=1))
        h = self.c2(torch.cat([h, mid_layer_list[-3]], dim=1))
        h = self.c3(torch.cat([h, mid_layer_list[-4]], dim=1))
        return self.out(h)

        return h


class SCFTDecoder(nn.Module):
    def __init__(self, base=16):
        super(SCFTDecoder, self).__init__()

        self.c0 = CBR(base*(16+8+4), base*8, 3, 1, 1, up=True)
        self.c1 = CBR(base*16, base*4, 3, 1, 1, up=True)
        self.c2 = CBR(base*8, base*2, 3, 1, 1, up=True)
        self.c3 = CBR(base*4, base*2, 3, 1, 1, up=True)
        self.out = nn.Sequential(
            nn.Conv2d(base*2, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x, mid_layer_list):
        h = self.c0(x)
        h = self.c1(torch.cat([h, mid_layer_list[-1]], dim=1))
        h = self.c2(torch.cat([h, mid_layer_list[-2]], dim=1))
        h = self.c3(torch.cat([h, mid_layer_list[-3]], dim=1))
        return self.out(h)

        return h


class Style2Paint(nn.Module):
    def __init__(self,
                 scft_base=64):

        super(Style2Paint, self).__init__()

        self.se = SCFTEncoder(base=scft_base)
        self.ce = SCFTEncoder(base=scft_base)
        mid_base = scft_base * (16 + 8 + 4)

        self.scft = SCFT(mid_base)
        self.adain0 = ResBlock(mid_base, mid_base)
        self.adain1 = ResBlock(mid_base, mid_base)
        self.adain2 = ResBlock(mid_base, mid_base)
        self.adain3 = ResBlock(mid_base, mid_base)

        init_weights(self.scft)

        self.dec = SCFTDecoder(base=scft_base)

        init_weights(self.ce)
        init_weights(self.se)
        init_weights(self.adain0)
        init_weights(self.adain1)
        init_weights(self.adain2)
        init_weights(self.adain3)
        init_weights(self.dec)

    def forward(self, x, style):
        ce, mid_layer_list = self.ce(x)

        se, _ = self.se(style)

        h = self.scft(ce, se)
        h = self.adain0(h)
        h = self.adain1(h)
        h = self.adain2(h)
        h = self.adain3(h)

        h = self.dec(h, mid_layer_list)

        return h


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(3):
            self.cnns.append(self._make_nets(base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_nets(self, base):
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

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            h = model(x)
            outputs.append(h)
            x = self.down(x)

        return outputs
