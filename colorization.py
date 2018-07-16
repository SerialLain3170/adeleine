import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, serializers, initializers, optimizers, cuda, Chain
from unet import Unet, Discriminator
import pylab
import numpy as np
import os
import argparse

xp = cuda.cupy

def set_optimizer(model, alpha, beta):
	optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))

	return optimizer

parser = argparse.ArgumentParser(description="pix2pix")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 10, type = int, help = "batch size")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")
parser.add_argument("--lam", default = 10.0, type = float, help = "the weight of content loss")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lam = args.lam

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

x_train = np.load('line.npy').astype(np.float32)
t_train = np.load('trim.npy').astype(np.float32)
x_test = np.load('test.npy').astype(np.float32)
Ntrain, channels, width, height = x_train.shape
Ntest = x_test.shape[0]

unet_model = Unet()
cuda.get_device(0).use()
unet_model.to_gpu()

dis_model = Discriminator()
dis_model.to_gpu()

unet_opt = set_optimizer(unet_model, 0.0002, 0.5)
dis_opt = set_optimizer(dis_model, 0.0002, 0.5)

for epoch in range(epochs):
	sum_unet_loss = 0
	sum_dis_loss = 0
	perm = np.random.permutation(Ntrain)
	for batch in range(0, Ntrain, batchsize):
		x = np.zeros((batchsize,channels,width,height), dtype=np.float32)
		t = np.zeros((batchsize,channels,width,height), dtype=np.float32)
		for j in range(batchsize):
			rnd = np.random.randint(Ntrain)
			x[j,:,:,:] = x_train[rnd]
			t[j,:,:,:] = t_train[rnd]

		x = Variable(cuda.to_gpu(x))
		t = Variable(cuda.to_gpu(t))
		y = unet_model(x)
		y_dis = dis_model(y)
		t_dis = dis_model(t)

		dis_loss = F.softmax_cross_entropy(y_dis, Variable(xp.zeros(batchsize,dtype=xp.int32)))
		fake_loss = F.softmax_cross_entropy(y_dis, Variable(xp.ones(batchsize,dtype=xp.int32)))
		real_loss = F.softmax_cross_entropy(t_dis, Variable(xp.zeros(batchsize,dtype=xp.int32)))

		loss_unet = lam * F.mean_absolute_error(y,t) + dis_loss
		loss_dis = fake_loss + real_loss

		unet_model.cleargrads()
		loss_unet.backward()
		loss_unet.unchain_backward()
		unet_opt.update()

		dis_model.cleargrads()
		loss_dis.backward()
		loss_dis.unchain_backward()
		dis_opt.update()

		sum_unet_loss += loss_unet.data.get()
		sum_dis_loss += loss_dis.data.get()

		if epoch % interval == 0 and batch == 0:
			serializers.save_npz('colorization.model', unet_model)
			for i in range(Ntest):
				line = (x_test[i]*255.0).transpose(1,2,0).astype(np.uint8)
				pylab.subplot(2,Ntest,2*i+1)
				pylab.imshow(line)
				pylab.axis('off')
				pylab.savefig(image_out_dir + '/color_%d.png' %epoch)

				x = Variable(cuda.to_gpu(x_test[i]))
				x = x.reshape(1,channels,width,height)
				with chainer.using_config('train', False):
					y = unet_model(x)
				y = y.data.get()
				tmp = (np.clip(y[0,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
				pylab.subplot(2,Ntest,2*i+2)
				pylab.imshow(tmp)
				pylab.axis('off')
				pylab.savefig( image_out_dir + '/color_%d.png' %epoch)
	print("epoch : {} Unet loss : {} Dis loss : {}".format(epoch, sum_unet_loss/Ntrain, sum_dis_loss/Ntrain))