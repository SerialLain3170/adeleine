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
            for x in range(21, 30):
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

        return h_relu5


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, up=False):
        super(CBR, self).__init__()

        if up:
            self.cbr = nn.Sequential(
                nn.Upsample(scale_factor=2),
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

        self.vgg = Vgg19()

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


class Style2Paint(nn.Module):
    def __init__(self, base=64):
        super(Style2Paint, self).__init__()

        self.ce = ContentEncoder()
        self.se = StyleEncoder()

        self.adain0 = AdaINResBlock(base*8, base*8)
        self.adain1 = AdaINResBlock(base*8, base*8)
        self.adain2 = AdaINResBlock(base*8, base*8)
        self.adain3 = AdaINResBlock(base*8, base*8)

        self.dec = Decoder()

        init_weights(self.ce)
        init_weights(self.se)
        init_weights(self.adain0)
        init_weights(self.adain1)
        init_weights(self.adain2)
        init_weights(self.adain3)
        init_weights(self.dec)

    def forward(self, x, style):
        ce, mid_layer_list = self.ce(x)
        se = self.se(style)

        h = self.adain0(ce, se)
        h = self.adain1(h, se)
        h = self.adain2(h, se)
        h = self.adain3(h, se)

        h = self.dec(h, mid_layer_list)

        return h


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(3, base, 7, 1, 3),
            CBR(base, base*2, 4, 2, 1),
            CBR(base*2, base*4, 4, 2, 1),
            CBR(base*4, base*8, 4, 2, 1),
            CBR(base*8, base*8, 4, 2, 1)
        )

        init_weights(self.dis)

    def forward(self, x):
        return self.dis(x)
