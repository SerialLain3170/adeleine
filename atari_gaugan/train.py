import chainer
import chainer.functions as F
from chainer import cuda, serializers
import numpy as np
import argparse

from pathlib import Path
from model import Generator, Discriminator, Prior, Encoder
from dataset import DataLoader
from utils import set_optimizer
from evaluation import Evaluaton

xp = cuda.cupy
cuda.get_device(0).use()


class GauGANLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return 10.0 * F.mean_absolute_error(y, t)

    @staticmethod
    def dis_loss(discriminator, y, t):
        y_adv_list, _ = discriminator(y)
        t_adv_list, _ = discriminator(t)

        sum_loss = 0

        for y_adv, t_adv in zip(y_adv_list, t_adv_list):
            loss = F.mean(F.relu(1. - t_adv)) + F.mean(F.relu(1. + y_adv))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def gen_loss(discriminator, y, t):
        y_dis_list, y_feats = discriminator(y)
        _, t_feats = discriminator(t)

        sum_loss = 0

        # adversarial loss
        for y_dis in y_dis_list:
            loss = -F.mean(y_dis)
            sum_loss += loss

        # feature matching loss
        for yf_list, tf_list in zip(y_feats, t_feats):
            for yf, tf in zip(yf_list, tf_list):
                _, ch, height, width = yf.shape
                sum_loss += 10.0 * F.mean_absolute_error(yf, tf) / (ch * height * width)

        return sum_loss


def train(epochs,
          iterations,
          batchsize,
          validsize,
          outdir,
          modeldir,
          data_path,
          extension,
          img_size,
          latent_dim,
          learning_rate,
          beta1,
          beta2,
          enable):

    # Dataset Definition
    dataloader = DataLoader(data_path, extension, img_size, latent_dim)
    print(dataloader)
    color_valid, line_valid = dataloader(validsize, mode="valid")
    noise_valid = dataloader.noise_generator(validsize)

    # Model Definition
    if enable:
        encoder = Encoder()
        encoder.to_gpu()
        enc_opt = set_optimizer(encoder)

    generator = Generator()
    generator.to_gpu()
    gen_opt = set_optimizer(generator, learning_rate, beta1, beta2)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, learning_rate, beta1, beta2)

    # Loss Funtion Definition
    lossfunc = GauGANLossFunction()

    # Evaluation Definition
    evaluator = Evaluaton()

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0
        for batch in range(0, iterations, batchsize):
            color, line = dataloader(batchsize)
            z = dataloader.noise_generator(batchsize)

            # Discriminator update
            if enable:
                mu, sigma = encoder(color)
                z = F.gaussian(mu, sigma)
            y = generator(z, line)

            y.unchain_backward()

            dis_loss = lossfunc.dis_loss(
                discriminator,
                F.concat([y, line]),
                F.concat([color, line])
            )

            discriminator.cleargrads()
            dis_loss.backward()
            dis_opt.update()
            dis_loss.unchain_backward()

            sum_dis_loss += dis_loss.data

            # Generator update
            z = dataloader.noise_generator(batchsize)

            if enable:
                mu, sigma = encoder(color)
                z = F.gaussian(mu, sigma)
            y = generator(z, line)

            gen_loss = lossfunc.gen_loss(
                discriminator,
                F.concat([y, line]),
                F.concat([color, line])
            )
            gen_loss += lossfunc.content_loss(y, color)

            if enable:
                gen_loss += 0.05 * F.gaussian_kl_divergence(mu, sigma) / batchsize

            generator.cleargrads()
            if enable:
                encoder.cleargrads()
            gen_loss.backward()
            gen_opt.update()
            if enable:
                enc_opt.update()
            gen_loss.unchain_backward()

            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

                with chainer.using_config("train", False):
                    y = generator(noise_valid, line_valid)
                y = y.data.get()
                sr = line_valid.data.get()
                cr = color_valid.data.get()

                evaluator(y, cr, sr, outdir, epoch, validsize=validsize)

        print(f"epoch: {epoch}")
        print(f"dis loss: {sum_dis_loss / iterations} gen loss: {sum_gen_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GauGAN")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=2000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--v', type=int, default=12, help="valid size")
    parser.add_argument('--outdir', type=Path, default='outdir', help="output directory")
    parser.add_argument('--modeldir', type=Path, default='modeldir', help="model output directory")
    parser.add_argument('--ext', type=str, default=".jpg", help="extension of training images")
    parser.add_argument('--size', type=int, default=224, help="the size of training images")
    parser.add_argument('--dim', type=int, default=256, help="dimensions of latent space")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate of Adam")
    parser.add_argument('--b1', type=float, default=0.0, help="beta1 of Adam")
    parser.add_argument('--b2', type=float, default=0.999, help="beta2 of Adam")
    parser.add_argument('--data_path', type=Path, help="path which contains training data")
    parser.add_argument('--encoder', action="store_true", help="enable image encoder")

    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, outdir, modeldir, args.data_path,
          args.ext, args.size, args.dim, args.lr, args.b1, args.b2, args.encoder)
