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


def downsampling(array):
    d2 = F.average_pooling_2d(array, 3, 2, 1)
    d4 = F.average_pooling_2d(d2, 3, 2, 1)

    return d2, d4


class GauGANLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    @staticmethod
    def dis_hinge_loss(discriminator, y, t):
        y_dis, _ = discriminator(y)
        t_dis, _ = discriminator(t)

        return F.mean(F.relu(1. - t_dis)) + F.mean(F.relu(1. + y_dis))

    @staticmethod
    def gen_hinge_loss(discriminator, y, t):
        y_dis, y_feats = discriminator(y)
        _, t_feats = discriminator(t)

        sum_loss = 0
        for yf, tf in zip(y_feats, t_feats):
            _, ch, height, width = yf.shape
            sum_loss += 10.0 * F.mean_absolute_error(yf, tf) / (ch * height * width)

        return -F.mean(y_dis) + sum_loss


def train(epochs, iterations, batchsize, validsize, path, outdir,
          con_weight, kl_weight, enable):
    # Dataset Definition
    dataloader = DataLoader(path)
    print(dataloader)
    color_valid, line_valid, _, _ = dataloader(validsize, mode="valid")
    noise_valid = dataloader.noise_generator(validsize)

    # Model Definition
    if enable:
        encoder = Encoder()
        encoder.to_gpu()
        enc_opt = set_optimizer(encoder)

    generator = Generator()
    generator.to_gpu()
    gen_opt = set_optimizer(generator)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator)

    discriminator_d2 = Discriminator()
    discriminator_d2.to_gpu()
    dis2_opt = set_optimizer(discriminator_d2)

    discriminator_d4 = Discriminator()
    discriminator_d4.to_gpu()
    dis4_opt = set_optimizer(discriminator_d4)

    # Loss Funtion Definition
    lossfunc = GauGANLossFunction()

    # Evaluation Definition
    evaluator = Evaluaton()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            color, line, _, _ = dataloader(batchsize)

            color_d2, color_d4 = downsampling(color)
            line_d2, line_d4 = downsampling(line)
            z = dataloader.noise_generator(batchsize)

            if enable:
                mu, sigma = encoder(color)
                z = F.gaussian(mu, sigma)
            y = generator(z, line)
            y_d2, y_d4 = downsampling(y)

            y.unchain_backward()
            y_d2.unchain_backward()
            y_d4.unchain_backward()

            loss = lossfunc.dis_hinge_loss(
                discriminator,
                F.concat([y, line]),
                F.concat([color, line])
            )
            loss += lossfunc.dis_hinge_loss(
                discriminator_d2,
                F.concat([y_d2, line_d2]),
                F.concat([color_d2, line_d2])
            )
            loss += lossfunc.dis_hinge_loss(
                discriminator_d4,
                F.concat([y_d4, line_d4]),
                F.concat([color_d4, line_d4])
            )

            discriminator.cleargrads()
            discriminator_d2.cleargrads()
            discriminator_d4.cleargrads()
            loss.backward()
            dis_opt.update()
            dis2_opt.update()
            dis4_opt.update()
            loss.unchain_backward()

            z = dataloader.noise_generator(batchsize)

            if enable:
                mu, sigma = encoder(color)
                z = F.gaussian(mu, sigma)
            y = generator(z, line)
            y_d2, y_d4 = downsampling(y)

            loss = lossfunc.gen_hinge_loss(
                discriminator,
                F.concat([y, line]),
                F.concat([color, line])
            )
            loss += lossfunc.gen_hinge_loss(
                discriminator_d2,
                F.concat([y_d2, line_d2]),
                F.concat([color_d2, line_d2])
            )
            loss += lossfunc.gen_hinge_loss(
                discriminator_d4,
                F.concat([y_d4, line_d4]),
                F.concat([color_d4, line_d4])
            )
            loss += con_weight * lossfunc.content_loss(y, color)
            loss += con_weight * lossfunc.content_loss(y_d2, color_d2)
            loss += con_weight * lossfunc.content_loss(y_d4, color_d4)

            if enable:
                loss += kl_weight * F.gaussian_kl_divergence(mu, sigma) / batchsize

            generator.cleargrads()
            if enable:
                encoder.cleargrads()
            loss.backward()
            gen_opt.update()
            if enable:
                enc_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{outdir}/generator.model", generator)
                serializers.save_npz(f"{outdir}/discriminator_0.model", discriminator)
                serializers.save_npz(f"{outdir}/discriminator_2.model", discriminator_d2)
                serializers.save_npz(f"{outdir}/discriminator_4.model", discriminator_d4)

                with chainer.using_config("train", False):
                    y = generator(noise_valid, line_valid)
                y = y.data.get()
                sr = line_valid.data.get()
                cr = color_valid.data.get()

                evaluator(y, cr, sr, outdir, epoch, validsize=validsize)
    
    print(f"epoch: {epoch}")
    print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GauGAN")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=10000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--v', type=int, default=3, help="valid size")
    parser.add_argument('--w', type=float, default=10.0, help="the weight of content loss")
    parser.add_argument('--kl', type=float, default=0.05, help="the weight of kl divergence loss")
    parser.add_argument('--encoder', action="store_true", help="enable image encoder")

    args = parser.parse_args()

    dataset_path = Path('./Dataset/danbooru-images/')
    outdir = Path('./outdir')
    outdir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, dataset_path, outdir, args.w, args.kl, args.encoder)