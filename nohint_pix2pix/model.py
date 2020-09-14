import torch
import numpy as np
import torch.nn as nn

from typing import List, Dict
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


class Generator(nn.Module):
	def __init__(self,
				 in_ch: int,
				 base=64):
		super(Generator, self).__init__()

		self.enc = self._make_encoder(in_ch, base)
		self.dec = self._make_decoder(base)
		self.out = nn.Sequential(
			nn.Conv2d(base, 3, 3, 1, 1),
			nn.Tanh()
		)

		init_weights(self.enc)
		init_weights(self.dec)
		init_weights(self.out)

	@staticmethod
	def _make_encoder(in_ch: int, base: int):
		modules = []
		modules.append(CBR(in_ch, base, 3, 1, 1, norm="bn"))
		modules.append(CBR(base, base*2, 4, 2, 1, norm="bn"))
		modules.append(CBR(base*2, base*4, 4, 2, 1, norm="bn"))
		modules.append(CBR(base*4, base*8, 4, 2, 1, norm="bn"))
		modules.append(CBR(base*8, base*8, 4, 2, 1, norm="bn"))
		modules.append(CBR(base*8, base*16, 4, 2, 1, norm="bn"))

		modules = nn.ModuleList(modules)

		return modules

	@staticmethod
	def _make_decoder(base: int):
		modules = []
		modules.append(CBR(base*16, base*8, 3, 1, 1, up=True, norm="bn"))
		modules.append(CBR(base*16, base*8, 3, 1, 1, up=True, norm="bn"))
		modules.append(CBR(base*16, base*4, 3, 1, 1, up=True, norm="bn"))
		modules.append(CBR(base*8, base*2, 3, 1, 1, up=True, norm="bn"))
		modules.append(CBR(base*4, base, 3, 1, 1, up=True, norm="bn"))

		modules = nn.ModuleList(modules)

		return modules

	def _encode(self, x: torch.Tensor) -> (torch.Tensor, List[torch.Tensor]):
		encode_list = []
		for layer in self.enc:
			x = layer(x)
			encode_list.append(x)

		return x, encode_list

	def _decode(self,
				x: torch.Tensor,
				encode_list: List[torch.Tensor]) -> torch.Tensor:
		for index, layer in enumerate(self.dec):
			if index in [1, 2, 3, 4]:
				x = layer(torch.cat([x, encode_list[-index-1]], dim=1))
			else:
				x = layer(x)

		return self.out(x)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x, encode_list = self._encode(x)
		x = self._decode(x, encode_list)

		return x


class Discriminator(nn.Module):
	def __init__(self, base: int=64):
		super(Discriminator, self).__init__()

		self.dis = nn.Sequential(
			CBR(3, base, 4, 2, 1, norm="bn"),
			CBR(base, base*2, 4, 2, 1, norm="bn"),
			CBR(base*2, base*4, 4, 2, 1, norm="bn"),
			CBR(base*4, base*8, 4, 2, 1, norm="bn"),
			CBR(base*8, base*16, 4, 2, 1, norm="bn"),
			nn.Conv2d(base*16, 1, 1, 1, 0)
		)

		init_weights(self.dis)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.dis(x)
