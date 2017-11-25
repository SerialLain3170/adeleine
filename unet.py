import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, initializers, Variable, Chain

class Unet(Chain):
	def __init__(self):
		super(Unet, self).__init__(
			conv0 = L.Convolution2D(3,32,3,1,1),
			conv1 = L.Convolution2D(32,64,4,2,1),
			conv2 = L.Convolution2D(64,128,4,2,1),
			conv3 = L.Convolution2D(128,256,4,2,1),
			conv4 = L.Convolution2D(256,256,4,2,1),
			conv5 = L.Convolution2D(256,256,4,2,1),
			conv6 = L.Convolution2D(256,256,4,2,1),
			conv7 = L.Convolution2D(256,256,4,2,1),

			dconv7 = L.Deconvolution2D(256,256,4,2,1),
			dconv6 = L.Deconvolution2D(512,256,4,2,1),
			dconv5 = L.Deconvolution2D(512,256,4,2,1),
			dconv4 = L.Deconvolution2D(512,256,4,2,1),
			dconv3 = L.Deconvolution2D(512,128,4,2,1),
			dconv2 = L.Deconvolution2D(256,64,4,2,1),
			dconv1 = L.Deconvolution2D(128,32,4,2,1),
			dconv0 = L.Convolution2D(64,3,3,1,1),

			bnc0 = L.BatchNormalization(32),
			bnc1 = L.BatchNormalization(64),
			bnc2 = L.BatchNormalization(128),
			bnc3 = L.BatchNormalization(256),
			bnc4 = L.BatchNormalization(256),
			bnc5 = L.BatchNormalization(256),
			bnc6 = L.BatchNormalization(256),
			bnc7 = L.BatchNormalization(256),

			bndc7 = L.BatchNormalization(256),
			bndc6 = L.BatchNormalization(256),
			bndc5 = L.BatchNormalization(256),
			bndc4 = L.BatchNormalization(256),
			bndc3 = L.BatchNormalization(128),
			bndc2 = L.BatchNormalization(64),
			bndc1 = L.BatchNormalization(32)
		)

	def forward(self,x):
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
	def __init__(self):
		super(Discriminator,self).__init__(
			conv1 = L.Convolution2D(3,32,4,2,1),
			conv2 = L.Convolution2D(32,32,3,1,1),
			conv3 = L.Convolution2D(32,64,4,2,1),
			conv4 = L.Convolution2D(64,64,3,1,1),
			conv5 = L.Convolution2D(64,128,4,2,1),
			conv6 = L.Convolution2D(128,128,3,1,1),
			conv7 = L.Convolution2D(128,256,4,2,1),
			l8 = L.Linear(None,2,initialW = initializers.HeNormal(math.sqrt(0.02*math.sqrt(8*8*256)/2))),

			bnc1 = L.BatchNormalization(32),
			bnc2 = L.BatchNormalization(32),
			bnc3 = L.BatchNormalization(64),
			bnc4 = L.BatchNormalization(64),
			bnc5 = L.BatchNormalization(128),
			bnc6 = L.BatchNormalization(128),
			bnc7 = L.BatchNormalization(256),
		)

	def forward(self,x):
		h = F.relu(self.bnc1(self.conv1(x)))
		h = F.relu(self.bnc2(self.conv2(h)))
		h = F.relu(self.bnc3(self.conv3(h)))
		h = F.relu(self.bnc4(self.conv4(h)))
		h = F.relu(self.bnc5(self.conv5(h)))
		h = F.relu(self.bnc6(self.conv6(h)))
		h = F.relu(self.bnc7(self.conv7(h)))
		return self.l8(h)