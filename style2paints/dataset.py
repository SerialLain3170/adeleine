import torch
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path
from xdog import xdog_process


class IllustDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 extension=".jpg"):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.sketch_path = sketch_path

    def _train_val_split(self, pathlist):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    def _coordinate(self, img):
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _xdog_preprocess(self, path):
        img = xdog_process(str(path))
        img = (img * 255.0).reshape(img.shape[0], img.shape[1], 1)
        img = np.tile(img, (1, 1, 3))

        return img

    def _pencil_preprocess(self, path):
        filename = path.name
        line_path = self.sketch_path / Path(filename)
        img = cv.imread(str(line_path))

        return img

    def _preprocess(self, path):
        method = np.random.choice(["xdog", "pencil"])

        if method == "xdog":
            img = self._xdog_preprocess(path)
        elif method == "pencil":
            img = self._pencil_preprocess(path)

        return img

    def valid(self, validsize):
        c_valid_box = []
        l_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color = cv.imread(str(color_path))
            line = self._preprocess(color_path)

            color = self._coordinate(color)
            line = self._coordinate(line)

            c_valid_box.append(color)
            l_valid_box.append(line)

        color = self._totensor(c_valid_box)
        line = self._totensor(l_valid_box)

        return color, line

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        color_path = self.train_list[idx]
        color =cv.imread(str(color_path))
        line = self._preprocess(color_path)

        return color, line


class IllustTestDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.pathlen = len(self.pathlist)

    def __repr__(self):
        return f"dataset length: {self.pathlen}"

    def __len__(self):
        return 99

    def __getitem__(self, idx):
        line_path = self.pathlist[idx]

        rnd = np.random.randint(self.pathlen)
        style_path = self.pathlist[rnd]

        return line_path, style_path


class LineCollator:
    def __init__(self, img_size=224):
        self.size = img_size

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
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _prepair(self, color, line):
        line, color = self._random_crop(line, color, size=self.size)

        color = self._coordinate(color)
        line = self._coordinate(line)

        return color, line

    def __call__(self, batch):
        c_box = []
        l_box = []

        for b in batch:
            color, line = b
            color, line = self._prepair(color, line)

            c_box.append(color)
            l_box.append(line)

        c = self._totensor(c_box)
        l = self._totensor(l_box)

        return (c, l)


class LineTestCollator:
    def __init__(self):
        pass

    @staticmethod
    def _random_crop(line, color):
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
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _prepare(self, image_path, style_path, size=512):
        color = cv.imread(str(image_path))
        line = line_process(str(image_path))

        color = cv.imread(str(style_path))

        color = self._coordinate(color)
        line = self._coordinate(line)

        return color, line

    def __call__(self, batch):
        c_box = []
        l_box = []

        for bpath, style in batch:
            color, line = self._prepare(bpath, style)

            c_box.append(color)
            l_box.append(line)

        c = self._totensor(c_box)
        l = self._totensor(l_box)

        return (c, l)