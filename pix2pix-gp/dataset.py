import numpy as np
import random
import cv2 as cv
import copy
import chainer
import chainer.functions as F

from xdog import xdog_process
from chainer import cuda
from pathlib import Path
from PIL import Image

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 digi_path: Path,
                 extension='.jpg',
                 train_size=128,
                 valid_size=512):

        self.data_path = data_path
        self.skecth_path = sketch_path
        self.digi_path = digi_path
        self.extension = extension
        self.train_size = train_size
        self.valid_size = valid_size

        self.interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )

        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

    def __str__(self):
        return f"dataset path: {self.data_path} train data: {self.train_len}"

    # Initialization method
    def _train_val_split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    # Line art preparation method
    @staticmethod
    def _add_intensity(img, intensity=1.7):
        const = 255.0 ** (1.0 - intensity)
        img = (const * (img ** intensity))

        return img

    @staticmethod
    def _morphology(img):
        method = np.random.choice(["dilate", "erode"])
        if method == "dilate":
            img = cv.dilate(img, (5, 5), iterations=1)
        elif method == "erode":
            img = cv.erode(img, (5, 5), iterations=1)

        return img

    @staticmethod
    def _color_variant(img, max_value=30):
        color = np.random.randint(max_value + 1)
        img[img < 200] = color

        return img

    def _detail_preprocess(self, img):
        intensity = np.random.randint(2)
        morphology = np.random.randint(2)
        color_variance = np.random.randint(2)

        if intensity:
            img = self._add_intensity(img)
        if morphology:
            img = self._morphology(img)
        if color_variance:
            img = self._color_variant(img)

        return img

    def _xdog_preprocess(self, path):
        img = xdog_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.skecth_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _digital_preprocess(self, path):
        filename = path.name
        line_path = self.digi_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _preprocess(self, path):
        method = np.random.choice(["xdog", "pencil", "digital"])

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)
        elif method == "digital":
            img = self._digital_preprocess(path)

        img = self._detail_preprocess(img)

        return img

    # Preprocess method
    @staticmethod
    def _random_crop(line, color, size):
        scale = np.random.randint(288, 768)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))

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

    def _prepare_pair(self, image_path, size, mode="train"):
        color = cv.imread(str(image_path))
        line = self._preprocess(image_path)

        if mode == "train":
            line, color = self._random_crop(line, color, size=size)

        color = self._coordinate(color)
        line = self._coordinate(line)

        return (line, color)

    def train(self, batchsize):
        color_box = []
        line_box = []

        for _ in range(batchsize):
            rnd = np.random.randint(self.train_len)
            image_path = self.train_list[rnd]

            line, color = self._prepare_pair(image_path, size=self.train_size, mode="train")

            color_box.append(color)
            line_box.append(line)

        color = self._variable(color_box)
        line = self._variable(line_box)

        return (line, color)

    def valid(self, validsize):
        color_box = []
        line_box = []

        for v in range(validsize):
            image_path = self.val_list[v]

            line, color = self._prepare_pair(image_path, size=self.valid_size, mode="valid")

            color_box.append(color)
            line_box.append(line)

        color = self._variable(color_box)
        line = self._variable(line_box)

        return (line, color)
