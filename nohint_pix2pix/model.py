import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda, initializers, Chain

xp = cuda.cupy


class CBR(Chain):
	def __init__(self,
				 in_ch,
				 out_ch,
				 kernel,
				 stride,
				 padding,
				 up=False,
				 activ=F.relu):

		self.up = up
		self.activ = activ
		w = initializers.GlorotUniform()
		super(CBR, self).__init__()

		with self.init_scope():
			self.c0 = L.Convolution2D(in_ch, out_ch, kernel, stride, padding, initialW=w)
			self.bn0 = L.BatchNormalization(out_ch)

	def __call__(self, x):
		if self.up:
			x = F.unpooling_2d(x, 2, 2, 0, cover_all=False)

		h = self.activ(self.bn0(self.c0(x)))

		return h


class UNet(Chain):
	def __init__(self, base=64):
		super(UNet, self).__init__()
		w = initializers.GlorotUniform()

		with self.init_scope():
			self.e0 = CBR(3, base, 3, 1, 1)
			self.e1 = CBR(base, base*2, 4, 2, 1, activ=F.leaky_relu)
			self.e2 = CBR(base*2, base*4, 4, 2, 1, activ=F.leaky_relu)
			self.e3 = CBR(base*4, base*8, 4, 2, 1, activ=F.leaky_relu)
			self.e4 = CBR(base*8, base*8, 4, 2, 1, activ=F.leaky_relu)
			self.e5 = CBR(base*8, base*16, 4, 2, 1, activ=F.leaky_relu)

			self.d0 = CBR(base*16, base*8, 3, 1, 1, up=True, activ=F.leaky_relu)
			self.d1 = CBR(base*16, base*8, 3, 1, 1, up=True, activ=F.leaky_relu)
			self.d2 = CBR(base*16, base*4, 3, 1, 1, up=True, activ=F.leaky_relu)
			self.d3 = CBR(base*8, base*2, 3, 1, 1, up=True, activ=F.leaky_relu)
			self.d4 = CBR(base*4, base, 3, 1, 1, up=True, activ=F.leaky_relu)
			self.out = L.Convolution2D(base, 3, 1, 1, 0, initialW=w)

	def __call__(self, x):
		h = self.e0(x)
		h1 = self.e1(h)
		h2 = self.e2(h1)
		h3 = self.e3(h2)
		h4 = self.e4(h3)
		h5 = self.e5(h4)

		h = self.d0(h5)
		h = self.d1(F.concat([h, h4]))
		h = self.d2(F.concat([h, h3]))
		h = self.d3(F.concat([h, h2]))
		h = self.d4(F.concat([h, h1]))
		h = self.out(h)

		return F.tanh(h)


class Discriminator(Chain):
	def __init__(self, base=64):
		super(Discriminator, self).__init__()

		w = initializers.GlorotUniform()
		with self.init_scope():
			self.cbr0 = CBR(3, base, 4, 2, 1)
			self.cbr1 = CBR(base, base*2, 4, 2, 1)
			self.cbr2 = CBR(base*2, base*4, 4, 2, 1)
			self.cbr3 = CBR(base*4, base*8, 4, 2, 1)
			self.cbr4 = CBR(base*8, base*16, 4, 2, 1)
			self.cout = L.Convolution2D(base*16, 1, 1, 1, 0, initialW=w)

	def __call__(self, x):
		h = self.cbr0(x)
		h = self.cbr1(h)
		h = self.cbr2(h)
		h = self.cbr3(h)
		h = self.cbr4(h)
		h = self.cout(h)

		return h
