import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, initializers, Variable, Chain

class Unet(Chain):
	def __init__(self, base = 32):
		super(Unet, self).__init__(
			conv0 = L.Convolution2D(3,base,3,1,1),
			conv1 = L.Convolution2D(base,base*2,4,2,1),
			conv2 = L.Convolution2D(base*2,base*4,4,2,1),
			conv3 = L.Convolution2D(base*4,base*8,4,2,1),
			conv4 = L.Convolution2D(base*8,base*8,4,2,1),
			conv5 = L.Convolution2D(base*8,base*8,4,2,1),
			conv6 = L.Convolution2D(base*8,base*8,4,2,1),
			conv7 = L.Convolution2D(base*8,base*8,4,2,1),

			dconv7 = L.Deconvolution2D(base*8,base*8,4,2,1),
			dconv6 = L.Deconvolution2D(base*16,base*8,4,2,1),
			dconv5 = L.Deconvolution2D(base*16,base*8,4,2,1),
			dconv4 = L.Deconvolution2D(base*16,base*8,4,2,1),
			dconv3 = L.Deconvolution2D(base*16,base*4,4,2,1),
			dconv2 = L.Deconvolution2D(base*8,base*2,4,2,1),
			dconv1 = L.Deconvolution2D(base*4,base,4,2,1),
			dconv0 = L.Convolution2D(base*2,3,3,1,1),

			bnc0 = L.BatchNormalization(base),
			bnc1 = L.BatchNormalization(base*2),
			bnc2 = L.BatchNormalization(base*4),
			bnc3 = L.BatchNormalization(base*8),
			bnc4 = L.BatchNormalization(base*8),
			bnc5 = L.BatchNormalization(base*8),
			bnc6 = L.BatchNormalization(base*8),
			bnc7 = L.BatchNormalization(base*8),

			bndc7 = L.BatchNormalization(base*8),
			bndc6 = L.BatchNormalization(base*8),
			bndc5 = L.BatchNormalization(base*8),
			bndc4 = L.BatchNormalization(base*8),
			bndc3 = L.BatchNormalization(base*4),
			bndc2 = L.BatchNormalization(base*2),
			bndc1 = L.BatchNormalization(base)
		)

	def __call__(self,x):
		enc0 = F.leaky_relu(self.bnc0(self.conv0(x)))
		enc1 = F.leaky_relu(self.bnc1(self.conv1(enc0)))
		enc2 = F.leaky_relu(self.bnc2(self.conv2(enc1)))
		enc3 = F.leaky_relu(self.bnc3(self.conv3(enc2)))
		enc4 = F.leaky_relu(self.bnc4(self.conv4(enc3)))
		enc5 = F.leaky_relu(self.bnc5(self.conv5(enc4)))
		enc6 = F.leaky_relu(self.bnc6(self.conv6(enc5)))
		enc7 = F.leaky_relu(self.bnc7(self.conv7(enc6)))

		dec7 = F.relu(self.bndc7(self.dconv7(enc7)))
		del enc7
		dec6 = F.relu(self.bndc6(self.dconv6(F.concat([enc6, dec7]))))
		del dec7, enc6
		dec5 = F.relu(self.bndc5(self.dconv5(F.concat([enc5, dec6]))))
		del dec6, enc5
		dec4 = F.relu(self.bndc4(self.dconv4(F.concat([enc4, dec5]))))
		del dec5, enc4
		dec3 = F.relu(self.bndc3(self.dconv3(F.concat([enc3, dec4]))))
		del dec4, enc3
		dec2 = F.relu(self.bndc2(self.dconv2(F.concat([enc2, dec3]))))
		del dec3, enc2
		dec1 = F.relu(self.bndc1(self.dconv1(F.concat([enc1, dec2]))))
		del dec2, enc1
		dec0 = self.dconv0(F.concat([enc0,dec1]))
		del dec1, enc0

		return dec0

class Discriminator(Chain):
	def __init__(self, base = 32):
		super(Discriminator,self).__init__(
			conv1 = L.Convolution2D(3,base,4,2,1),
			conv2 = L.Convolution2D(base,base,3,1,1),
			conv3 = L.Convolution2D(base,base*2,4,2,1),
			conv4 = L.Convolution2D(base*2,base*2,3,1,1),
			conv5 = L.Convolution2D(base*2,base*4,4,2,1),
			conv6 = L.Convolution2D(base*4,base*4,3,1,1),
			conv7 = L.Convolution2D(base*4,base*8,4,2,1),
			l8 = L.Linear(None,2,initialW = initializers.HeNormal(math.sqrt(0.02*math.sqrt(8*8*base*8)/2))),

			bnc1 = L.BatchNormalization(base),
			bnc2 = L.BatchNormalization(base),
			bnc3 = L.BatchNormalization(base*2),
			bnc4 = L.BatchNormalization(base*2),
			bnc5 = L.BatchNormalization(base*4),
			bnc6 = L.BatchNormalization(base*4),
			bnc7 = L.BatchNormalization(base*8),
		)

	def __call__(self,x):
		h = F.relu(self.bnc1(self.conv1(x)))
		h = F.relu(self.bnc2(self.conv2(h)))
		h = F.relu(self.bnc3(self.conv3(h)))
		h = F.relu(self.bnc4(self.conv4(h)))
		h = F.relu(self.bnc5(self.conv5(h)))
		h = F.relu(self.bnc6(self.conv6(h)))
		h = F.relu(self.bnc7(self.conv7(h)))
		return self.l8(h)