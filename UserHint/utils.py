from chainer import optimizers


def set_optimizer(model, alpha=0.0001, beta=0.9):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer
