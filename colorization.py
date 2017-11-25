import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, serializers, initializers, optimizers, cuda, Chain
from unet import Unet, Discriminator
import pylab
import matplotlib.pyplot as plt
import numpy as np
import os

xp = cuda.cupy

def set_optimizer(model, alpha, beta):
	optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
	return optimizer

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

x_train = np.load('line.npy').astype(np.float32)[0:4900]
t_train = np.load('trim.npy').astype(np.float32)[0:4900]
x_test = np.load('line.npy').astype(np.float32)[4901:4910]

epochs = 100
interval = 10
batchsize = 10
lam = 100
Ntrain = x_train.shape[0]
Ntest = x_test.shape[0]

unet_model = Unet()
cuda.get_device(0).use()
unet_model.to_gpu()

dis_model = Discriminator()
cuda.get_device(0).use()
dis_model.to_gpu()

unet_opt = set_optimizer(unet_model, 0.0002, 0.5)
dis_opt = set_optimizer(dis_model, 0.0002, 0.5)

for epoch in range(epochs):
	sum_unet_loss = 0
	sum_dis_loss = 0
	perm = np.random.permutation(Ntrain)
	for batch in range(0, Ntrain, batchsize):
		x = Variable(cuda.to_gpu(x_train[perm[batch : batch + batchsize]]))
		t = Variable(cuda.to_gpu(t_train[perm[batch : batch + batchsize]]))
		y = unet_model.forward(x)
		y_dis = dis_model.forward(y)
		t_dis = dis_model.forward(t)

		dis_loss = F.softmax_cross_entropy(y_dis, Variable(xp.zeros(batchsize,dtype=xp.int32)))
		fake_loss = F.softmax_cross_entropy(y_dis, Variable(xp.ones(batchsize,dtype=xp.int32)))
		real_loss = F.softmax_cross_entropy(t_dis, Variable(xp.zeros(batchsize,dtype=xp.int32)))

		loss_unet = lam*F.mean_absolute_error(y,t) + dis_loss
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
	print("epoch : {} Unet loss : {} Dis loss : {}".format(epoch, sum_unet_loss/Ntrain, sum_dis_loss/Ntrain))

serializers.save_npz('colorization.model', unet_model)

for i in range(Ntest):
	line = (x_test[i]*127.5 + 127.5).transpose(1,2,0)
	line = line.astype(np.uint8)
	pylab.imshow(line)
	pylab.axis('off')
	pylab.savefig('./output/line_%d.png' %i)

	x = Variable(cuda.to_gpu(x_test[i]))
	x = x.reshape(1,3,128,128)
	with chainer.using_config('train', False):
		y = unet_model.forward(x)
	y = y.data.get()
	tmp = (np.clip(y[0,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
	pylab.imshow(tmp)
	pylab.axis('off')
	pylab.savefig( './output/colorization_%d.png' %i)
