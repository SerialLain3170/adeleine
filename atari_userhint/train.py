import chainer
import chainer.functions as F
import numpy as np
import argparse

from chainer import cuda, serializers
from pathlib import Path
from model import Generator, Discriminator, VGG, SAGenerator, SAGeneratorWithGuide
from utils import set_optimizer
from dataset import DataLoader, RefineDataset
from evaluation import Evaluation

xp = cuda.cupy
cuda.get_device(0).use()


class LossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    @staticmethod
    def perceptual_loss(vgg, y, t):
        y_feat = vgg(y)
        t_feat = vgg(t)
        sum_loss = 0
        sum_loss += F.mean_squared_error(y_feat, t_feat)

        return sum_loss

    @staticmethod
    def dis_hinge_loss(y_list, t_list):
        sum_loss = 0
        for y, t in zip(y_list, t_list):
            loss = F.mean(F.relu(1. - t)) + F.mean(F.relu(1. + y))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def gen_hinge_loss(y_list):
        sum_loss = 0
        for y in y_list:
            loss = -F.mean(y)
            sum_loss += loss

        return sum_loss

    @staticmethod
    def positive_enforcing_loss(y):
        sum_loss = 0
        b, c, h, w = y.shape
        for color in range(3):
            ch = y[:, color, :, :]
            mean = F.mean(ch)
            mean = mean * chainer.as_variable(xp.ones(shape=(b, h, w)).astype(xp.float32))
            loss = F.mean_squared_error(ch, mean)
            sum_loss += loss

        return -sum_loss


def train(epochs,
          iterations,
          batchsize,
          validsize,
          data_path,
          sketch_path,
          digi_path,
          extension,
          img_size,
          outdir,
          modeldir,
          pretrained_epoch,
          adv_weight,
          enf_weight,
          sn,
          bn,
          activ):

    # Dataset Definition
    dataloader = DataLoader(data_path, sketch_path, digi_path,
                            extension=extension, img_size=img_size)
    print(dataloader)
    color_valid, line_valid, mask_valid, ds_valid = dataloader(validsize, mode="valid")

    # Model & Optimizer Definition
    generator = SAGeneratorWithGuide(attn_type="sa", bn=bn, activ=activ)
    #generator = SAGenerator(attn_type="sa", base=64)
    generator.to_gpu()
    gen_opt = set_optimizer(generator)

    discriminator = Discriminator(sn=sn)
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator)

    vgg = VGG()
    vgg.to_gpu()
    vgg_opt = set_optimizer(vgg)
    vgg.base.disable_update()

    # Loss Function Definition
    lossfunc = LossCalculator()

    # Evaluation Definition
    evaluator = Evaluation()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            color, line, mask, mask_ds = dataloader(batchsize)
            line_input = F.concat([line, mask])

            extractor = vgg(mask, extract=True)
            extractor = F.average_pooling_2d(extractor, 3, 2, 1)
            extractor.unchain_backward()

            if epoch > pretrained_epoch:
                adv_weight = 0.1
                enf_weight = 0.0

            # Discriminator update
            fake, _ = generator(line_input, mask_ds, extractor)
            y_dis = discriminator(fake, extractor)
            t_dis = discriminator(color, extractor)
            loss = adv_weight * lossfunc.dis_hinge_loss(y_dis, t_dis)

            fake.unchain_backward()

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            # Generator update
            fake, guide = generator(line_input, mask_ds, extractor)
            y_dis = discriminator(fake, extractor)

            loss = adv_weight * lossfunc.gen_hinge_loss(y_dis)
            loss += enf_weight * lossfunc.positive_enforcing_loss(fake)
            loss += lossfunc.content_loss(fake, color)
            loss += 0.9 * lossfunc.content_loss(guide, color)
            loss += lossfunc.perceptual_loss(vgg, fake, color)

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

                extractor = vgg(line_valid, extract=True)
                extractor = F.average_pooling_2d(extractor, 3, 2, 1)
                extractor.unchain_backward()
                line_valid_input = F.concat([line_valid, mask_valid])

                with chainer.using_config('train', False):
                    y_valid, guide_valid = generator(line_valid_input, ds_valid, extractor)

                y_valid = y_valid.data.get()
                c_valid = color_valid.data.get()
                input_valid = line_valid_input.data.get()
                guide_valid = guide_valid.data.get()

                evaluator(y_valid, c_valid, input_valid, guide_valid, outdir, epoch, validsize)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


def train_refine(epochs,
                 iterations,
                 batchsize,
                 validsize,
                 data_path,
                 sketch_path,
                 digi_path,
                 st_path,
                 extension,
                 img_size,
                 crop_size,
                 outdir,
                 modeldir,
                 adv_weight,
                 enf_weight):

    # Dataset Definition
    dataloader = RefineDataset(data_path, sketch_path, digi_path, st_path,
                               extension=extension, img_size=img_size, crop_size=crop_size)
    print(dataloader)
    color_valid, line_valid, mask_valid, ds_valid, cm_valid = dataloader(validsize, mode="valid")

    # Model & Optimizer Definition
    generator = SAGeneratorWithGuide(attn_type="sa", base=64, bn=True, activ=F.relu)
    generator.to_gpu()
    gen_opt = set_optimizer(generator)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator)

    vgg = VGG()
    vgg.to_gpu()
    vgg_opt = set_optimizer(vgg)
    vgg.base.disable_update()

    # Loss Function Definition
    lossfunc = LossCalculator()

    # Evaluation Definition
    evaluator = Evaluation()

    iteration = 0

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0
        for batch in range(0, iterations, batchsize):
            iteration += 1
            color, line, mask, mask_ds, color_mask = dataloader(batchsize)
            line_input = F.concat([line, mask])

            extractor = vgg(color_mask, extract=True)
            extractor = F.average_pooling_2d(extractor, 3, 2, 1)
            extractor.unchain_backward()

            # Discriminator update
            fake, _ = generator(line_input, mask_ds, extractor)
            y_dis = discriminator(fake, extractor)
            t_dis = discriminator(color, extractor)
            loss = adv_weight * lossfunc.dis_hinge_loss(y_dis, t_dis)

            fake.unchain_backward()

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            sum_dis_loss += loss.data

            # Generator update
            fake, guide = generator(line_input, mask_ds, extractor)
            y_dis = discriminator(fake, extractor)

            loss = adv_weight * lossfunc.gen_hinge_loss(y_dis)
            loss += lossfunc.content_loss(fake, color)
            loss += 0.9 * lossfunc.content_loss(guide, color)

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_gen_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

                extractor = vgg(cm_valid, extract=True)
                extractor = F.average_pooling_2d(extractor, 3, 2, 1)
                extractor.unchain_backward()
                line_valid_input = F.concat([line_valid, mask_valid])

                with chainer.using_config('train', False):
                    y_valid, guide_valid = generator(line_valid_input, ds_valid, extractor)

                y_valid = y_valid.data.get()
                c_valid = color_valid.data.get()
                input_valid = line_valid_input.data.get()
                cm_val = cm_valid.data.get()
                guide_valid = guide_valid.data.get()
                input_valid = np.concatenate([input_valid[:, 3:6], cm_val], axis=1)

                evaluator(y_valid, c_valid, input_valid, guide_valid, outdir, epoch, validsize)

            print(f"iter: {iteration} dis loss: {sum_dis_loss} gen loss: {gen_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAM")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=2000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--v', type=int, default=5, help="valid size")
    parser.add_argument('--outdir', type=Path, default='outdir', help="output directory")
    parser.add_argument('--modeldir', type=Path, default='modeldir', help="model output directory")
    parser.add_argument('--ext', type=str, default='.jpg', help="extension of training images")
    parser.add_argument('--size', type=int, default=256, help="size of training images")
    parser.add_argument('--isize', type=int, default=512, help="the overall size")
    parser.add_argument('--pre', type=int, default=100, help="epochs of pretraining")
    parser.add_argument('--enf', type=float, default=0.001, help="the weight of content loss")
    parser.add_argument('--a', type=float, default=0.01, help="the weight of adversarial loss")
    parser.add_argument('--sn', action="store_true", help="enable spectral normalization")
    parser.add_argument('--bn', action="store_true", help="enable batch normalization in G")
    parser.add_argument('--act', default=F.relu, help="activation function in G")
    parser.add_argument('--data_path', type=Path, help="path which contains color images")
    parser.add_argument('--sketch_path', type=Path, help="path which contains sketches")
    parser.add_argument('--digi_path', type=Path, help="path which contains digitals")
    parser.add_argument('--st_path', type=Path, help="path which contains spatial transformers")
    parser.add_argument('--type', type=str, default='normal', help="type of training")

    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    if args.type == "normal":
        train(args.e, args.i, args.b, args.v, args.data_path, args.sketch_path, args.digi_path,
              args.ext, args.size, outdir, modeldir, args.pre, args.a, args.enf,
              args.sn, args.bn, args.act)

    elif args.type == "refine":
        train_refine(args.e, args.i, args.b, args.v, args.data_path, args.sketch_path,
                     args.digi_path, args.st_path, args.ext, args.isize, args.size,
                     outdir, modeldir, args.a, args.enf)

    raise AttributeError
