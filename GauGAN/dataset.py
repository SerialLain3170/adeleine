import numpy as np
import random
import cv2 as cv
import copy
import chainer

from xdog import line_process
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

    def __str__(self):
        return f"dataset path: {self.path} train data: {self.train_len}"

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.9)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    @staticmethod
    def _random_crop(line, color, size=224):
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

    @staticmethod
    def noise_generator(batchsize):
        noise = xp.random.normal(size=(batchsize, 256)).astype(xp.float32)

        return chainer.as_variable(noise)

    @staticmethod
    def _making_mask(mask, color, size=224):
        choice = np.random.choice(['width', 'height', 'diag'])

        if choice == 'width':
            rnd_height = np.random.randint(4, 8)
            rnd_width = np.random.randint(4, 64)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 64)
            rnd_width = np.random.randint(4, 8)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'diag':
            rnd_height = np.random.randint(4, 8)
            rnd_width = np.random.randint(4, 64)

            rnd1 = np.random.randint(size - rnd_height - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            for index in range(rnd_width):
                mask[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index]

        return mask

    def _prepare_pair(self, image_path, size=224, repeat=16):
        interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )
        interpolation = random.choice(interpolations)

        color = cv.imread(str(image_path))
        line = line_process(str(image_path))

        line, color = self._random_crop(line, color, size=size)
        mask = copy.copy(line)

        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)
        mask_ds = cv.resize(mask, (int(size/2), int(size/2)), interpolation=interpolation)

        color = self._coordinate(color)
        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (color, line, mask, mask_ds)

    def __call__(self, batchsize, mode='train', size=224):
        color_box = []
        line_box = []
        mask_box = []
        mask_ds_box = []

        for _ in range(batchsize):
            if mode == 'train':
                rnd = np.random.randint(self.train_len)
                image_path = self.train[rnd]
            elif mode == 'valid':
                rnd = np.random.randint(self.valid_len)
                image_path = self.valid[rnd]
            else:
                raise AttributeError

            color, line, mask, mask_ds = self._prepare_pair(image_path, size=size)

            color_box.append(color)
            line_box.append(line)
            mask_box.append(mask)
            mask_ds_box.append(mask_ds)

        color = self._variable(color_box)
        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)

        return (color, line, mask, mask_ds)