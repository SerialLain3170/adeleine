import chainer

from chainer import optimizers


def set_optimizer(model, alpha, beta, weight_decay):
	optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

	return optimizer