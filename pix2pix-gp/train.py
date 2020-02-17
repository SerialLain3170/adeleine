import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import argparse

from pathlib import Path
from chainer import cuda, serializers
from model import Generator, Discriminator
from dataset import DatasetLoader
from utils import set_optimizer, call_zeros
from visualize import Visualizer

xp = cuda.cupy


class Pix2pixGPLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def zero_centered_gradient_penalty_fake(fake, y):
        grad, = chainer.grad([fake], [y], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        zeros = call_zeros(grad)

        loss = 10 * F.mean_squared_error(grad, zeros)

        return loss

    @staticmethod
    def zero_centered_gradient_penalty_real(discriminator, t):
        t = chainer.Variable(t.data)
        real = discriminator(t)

        grad, = chainer.grad([real], [t], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        zeros = call_zeros(grad)

        loss = 10 * F.mean_squared_error(grad, zeros)

        return loss

    def dis_loss(self, discriminator, y, t):
        y_dis = discriminator(y)
        t_dis = discriminator(t)

        loss = self.zero_centered_gradient_penalty_fake(y_dis, y)
        loss += self.zero_centered_gradient_penalty_real(discriminator, t)

        return F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis)) + loss

    @staticmethod
    def gen_loss(discriminator, y):
        y_dis = discriminator(y)

        return F.mean(F.softplus(-y_dis))

    @staticmethod
    def content_loss(y, t):
        return 10.0 * F.mean_absolute_error(y, t)


def train(epochs,
        iterations,
        batchsize,
        validsize,
        outdir,
        modeldir,
        extension,
        train_size,
        valid_size,
        data_path,
        sketch_path,
        digi_path,
        learning_rate,
        beta1,
        weight_decay):

    # Dataset definition
    dataset = DatasetLoader(data_path, sketch_path, digi_path,
                            extension, train_size, valid_size)
    print(dataset)
    x_val, t_val = dataset.valid(validsize)

    # Model & Optimizer definition
    generator = Generator()
    generator.to_gpu()
    gen_opt = set_optimizer(generator, learning_rate, beta1, weight_decay)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, learning_rate, beta1, weight_decay)

    # Loss function definition
    lossfunc = Pix2pixGPLossCalculator()

    # Visualization definition
    visualizer = Visualizer()

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0
        for batch in range(0, iterations, batchsize):
            x, t = dataset.train(batchsize)

            # Discriminator update
            y = generator(x)
            y.unchain_backward()

            dis_loss = lossfunc.dis_loss(discriminator, y, t)

            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()

            sum_dis_loss += dis_loss.data

            # Generator update
            y = generator(x)

            gen_loss = lossfunc.gen_loss(discriminator, y)
            gen_loss += lossfunc.content_loss(y, t)

            generator.cleargrads()
            gen_loss.backward()
            gen_opt.update()

            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

                with chainer.using_config("train", False):
                    y = generator(x_val)

                x = x_val.data.get()
                t = t_val.data.get()
                y = y.data.get()

                visualizer(x, t, y, outdir, epoch, validsize)

        print(f"epoch: {epoch}")
        print(f"dis loss: {sum_dis_loss/iterations} gen loss: {sum_gen_loss/iterations}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="pix2pix")
	parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
	parser.add_argument("--i", type=int, default=20000, help="the number of iterations")
	parser.add_argument("--b", type=int, default=16, help="batch size")
	parser.add_argument("--v", type=int, default=12, help="valid size")
	parser.add_argument("--outdir", type=Path, default='outdir', help="output directory")
	parser.add_argument("--modeldir", type=Path, default='modeldir', help="model output directory")
	parser.add_argument("--ext", type=str, default=".jpg", help="extension of training images")
	parser.add_argument("--ts", type=int, default=128, help="size of training images")
	parser.add_argument("--vs", type=int, default=512, help="size of validation images")
	parser.add_argument("--lr", type=float, default=0.0002, help="alpha of Adam")
	parser.add_argument("--b1", type=float, default=0.5, help="beta1 of Adam")
	parser.add_argument("--wd", type=float, default=0.00001, help="weight decay of optimizer")
	parser.add_argument("--data_path", type=Path, help="path which contains color images")
	parser.add_argument("--sketch_path", type=Path, help="path which contains sketch images")
	parser.add_argument("--digi_path", type=Path, help="path which contains digital images")
	args = parser.parse_args()

	outdir = args.outdir
	outdir.mkdir(exist_ok=True)

	modeldir = args.modeldir
	modeldir.mkdir(exist_ok=True)

	train(args.e, args.i, args.b, args.v, outdir, modeldir, args.ext, args.ts, args.vs,
		  args.data_path, args.sketch_path, args.digi_path, args.lr, args.b1, args.wd)
