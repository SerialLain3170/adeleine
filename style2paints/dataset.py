import torch
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path
from xdog import line_process


class IllustDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.pathlist = list(self.path.glob('**/*.jpg'))
        self.pathlen = len(self.pathlist)

    def __repr__(self):
        return f"dataset length: {self.pathlen}"

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        return self.pathlist[idx]


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
    def __init__(self):
        pass

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

    def _prepair(self, image_path, size=224):
        color = cv.imread(str(image_path))
        line = line_process(str(image_path))

        line, color = self._random_crop(line, color, size=size)

        color = self._coordinate(color)
        line = self._coordinate(line)

        return color, line

    def __call__(self, batch):
        c_box = []
        l_box = []

        for bpath in batch:
            color, line = self._prepair(bpath)

            c_box.append(color)
            l_box.append(line)

        c = self._totensor(c_box)
        l = self._totensor(l_box)

        return (c, l)


class LinezTestCollator:
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
