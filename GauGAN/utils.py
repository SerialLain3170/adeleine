from chainer import optimizers


def set_optimizer(model, alpha=0.0002, beta=0.0, beta2=0.999):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta, beta2=beta2)
    optimizer.setup(model)

    return optimizer