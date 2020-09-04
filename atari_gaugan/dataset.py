import numpy as np
import random
import cv2 as cv
import copy
import chainer

from xdog import xdog_process
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()


class DataLoader:
    def __init__(self,
                 path,
                 extension='.jpg',
                 img_size=224,
                 latent_dim=256):

        self.path = path
        self.pathlist = list(self.path.glob(f"**/*{extension}"))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

        self.size = img_size
        self.latent_dim = latent_dim

        self.interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

    def __str__(self):
        return f"dataset path: {self.path} train data: {self.train_len}"

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    @staticmethod
    def _random_crop(line, color, size):
        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _variable(image_list):
        return chainer.as_variable(xp.array(image_list).astype(xp.float32))

    def noise_generator(self, batchsize):
        noise = xp.random.normal(size=(batchsize, self.latent_dim)).astype(xp.float32)

        return chainer.as_variable(noise)

    def _prepare_pair(self, image_path, size, repeat=16):
        interpolation = random.choice(self.interpolations)

        color = cv.imread(str(image_path))
        line = xdog_process(str(image_path))

        line, color = self._random_crop(line, color, size=size)

        color = self._coordinate(color)
        line = self._coordinate(line)

        return (color, line)

    def __call__(self, batchsize, mode='train'):
        color_box = []
        line_box = []

        for _ in range(batchsize):
            if mode == 'train':
                rnd = np.random.randint(self.train_len)
                image_path = self.train[rnd]
            elif mode == 'valid':
                rnd = np.random.randint(self.valid_len)
                image_path = self.valid[rnd]
            else:
                raise AttributeError

            color, line = self._prepare_pair(image_path, size=self.size)

            color_box.append(color)
            line_box.append(line)

        color = self._variable(color_box)
        line = self._variable(line_box)

        return (color, line)
