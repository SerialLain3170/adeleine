import chainer

from chainer import optimizers, cuda

xp = cuda.cupy


def set_optimizer(model, alpha, beta, weight_decay):
	optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

	return optimizer


def call_zeros(tensor):
    zeros = xp.zeros_like(tensor).astype(xp.float32)
    zeros = chainer.as_variable(zeros)

    return zeros


def call_ones(tensor):
    ones = xp.ones_like(tensor).astype(xp.float32)
    ones = chainer.as_variable(ones)

    return ones