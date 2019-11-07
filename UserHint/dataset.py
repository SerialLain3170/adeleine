import numpy as np
import random
import cv2 as cv
import copy
import chainer
import chainer.functions as F
import torch

from xdog import line_process
from chainer import cuda
from pathlib import Path
from scipy.stats import truncnorm
from PIL import Image
from torchvision import transforms
from torch.utils.serialization import load_lua

xp = cuda.cupy
cuda.get_device(0).use()


class DataLoader:
    def __init__(self, path, paint_type="cell", interpolate=False):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)
        self.paint_type = paint_type
        self.interpolate = interpolate
        self.digi_model, self.d_mean, self.d_std = self._digital_model_load()
        self.line_path = Path("./danbooru-copy/")

    def __str__(self):
        return f"dataset path: {self.path} train data: {self.train_len}"

    def _digital_model_load(self):
        modelpath = "model_gan.t7"
        cache = load_lua(modelpath)
        model = cache.model
        immean = cache.mean
        imstd = cache.std
        model.evaluate()
        model.cuda()

        return model, immean, imstd

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    def _xdog_preprocess(self, path):
        img = line_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.line_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _digital_preprocess(self, img):
        img = img[:, :, 0]
        img = Image.fromarray(img)
        img = ((transforms.ToTensor()(img) - self.d_mean) / self.d_std).unsqueeze(0)
        img = self.digi_model.forward(img.cuda()).float()

        img = img[0].permute(1, 2, 0).repeat(1, 1, 3)
        img = (img * 255).detach().cpu().numpy()

        return img

    def _preprocess(self, path):
        method = np.random.choice(["xdog", "pencil", "digital"])

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)
        elif method == "digital":
            img = self._pencil_preprocess(path)
            img = self._digital_preprocess(img)

        return img

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
                mask[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index]

        return mask

    @staticmethod
    def _interpolate(img):
        height, width = img.shape[0], img.shape[1]
        canvas = np.zeros(shape=(1024, 784, 3), dtype=np.uint8)
        canvas[:height, :width, :] = img

        return canvas

    def _prepare_pair(self, image_path, size=224):
        interpolations = (
            cv.INTER_LINEAR,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_CUBIC,
            cv.INTER_LANCZOS4
        )
        interpolation = random.choice(interpolations)

        color = cv.imread(str(image_path))
        line = self._preprocess(image_path)

        line, color = self._random_crop(line, color, size=size)
        mask = copy.copy(line)
        repeat = np.random.randint(8, 20)
        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)
        mask_ds = cv.resize(mask, (int(size/2), int(size/2)), interpolation=interpolation)

        color = self._coordinate(color)
        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (color, line, mask, mask_ds)

    def _prepare_test(self, line_path, mask_path):
        print(f"line path: {line_path}")
        print(f"mask path: {mask_path}")
        line = cv.imread(str(line_path))
        mask = cv.imread(str(mask_path))
        if self.interpolate:
            line = self._interpolate(line)
            mask = self._interpolate(mask)
        if self.paint_type == "imp":
            line = cv.morphologyEx(line, cv.MORPH_CLOSE, (5, 5), iterations=2)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (5, 5), iterations=2)
        height, width = line.shape[0], line.shape[1]
        mask_ds = cv.resize(mask, (int(width/2), int(height/2)))

        print(line.shape, mask.shape, mask_ds.shape)

        line = self._coordinate(line)
        mask = self._coordinate(mask)
        mask_ds = self._coordinate(mask_ds)

        return (line, mask, mask_ds)

    def test(self, line_path, mask_path):
        line_box = []
        mask_box = []
        mask_ds_box = []

        line, mask, mask_ds = self._prepare_test(line_path, mask_path)
        line_box.append(line)
        mask_box.append(mask)
        mask_ds_box.append(mask_ds)

        line = self._variable(line_box)
        mask = self._variable(mask_box)
        mask_ds = self._variable(mask_ds_box)

        return (line, mask, mask_ds)

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


class RefineDataset:
    def __init__(self, path):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.train, self.valid = self._split(self.pathlist)
        self.train_len = len(self.train)
        self.valid_len = len(self.valid)

    def __str__(self):
        return f"dataset path: {self.path} train data: {self.train_len}"

    def _split(self, pathlist: list):
        split_point = int(len(self.pathlist) * 0.95)
        x_train = self.pathlist[:split_point]
        x_test = self.pathlist[split_point:]

        return x_train, x_test

    def _spatial_transformer(self, img):
        height, width = img.shape[1], img.shape[2]
        img = img[None, :, :]
        theta = truncnorm.rvs(0.0, 1, loc=0.0, scale=1.0, size=(1, 2, 3)).astype(np.float32)
        grid = F.spatial_transformer_grid(theta, (height, width))
        img = F.spatial_transformer_sampler(img, grid)

        img = img[0].data.transpose(1, 2, 0).astype(np.uint8)
        cv.imwrite('./test.png', img)

    def test(self):
        img = cv.imread(str(self.train[0]))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        self._spatial_transformer(img)


if __name__ == "__main__":
    path = Path('./danbooru-images/')
    dataset = RefineDataset(path)
    dataset.test()
