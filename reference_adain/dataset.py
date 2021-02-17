import torch
import numpy as np
import cv2 as cv

from typing import List
from typing_extensions import Literal
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from hint_processor import LineProcessor

LineArt = List[Literal["xdog", "pencil", "digital", "blend"]]


class IllustDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 line_method: LineArt,
                 extension=".jpg",
                 train_size=224,
                 valid_size=256,
                 color_space="rgb",
                 line_space="rgb"):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.sketch_path = sketch_path

        self.train_size = train_size
        self.valid_size = valid_size

        self.line_process = LineProcessor(sketch_path, line_method)
        self.color_space = color_space
        self.line_space = line_space

    @staticmethod
    def _train_val_split(pathlist: List[Path]) -> (List, List):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    @staticmethod
    def _coordinate(img: np.array,
                    color_space: str) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _random_crop(line, color, size):
        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    def _preprocess(self, color, line):
        line, color = self._random_crop(line,
                                        color,
                                        size=self.train_size)

        color = self._coordinate(color, self.color_space)
        line = self._coordinate(line, self.line_space)

        return color, line

    def valid(self, validsize):
        c_valid_box = []
        l_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color = cv.imread(str(color_path))
            line = self.line_process(color_path)

            color = self._coordinate(color, self.color_space)
            line = self._coordinate(line, self.line_space)

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
        color = cv.imread(str(color_path))
        line = self.line_process(color_path)

        color, line = self._preprocess(color, line)

        return color, line


class IllustTestDataset(Dataset):
    def __init__(self, data_path: Path, sketch_path: Path):
        self.path = data_path
        self.pathlist = list(self.path.glob('**/*.png'))
        self.pathlen = len(self.pathlist)

        self.sketch_path = sketch_path

    def __repr__(self):
        return f"dataset length: {self.pathlen}"

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        line_path = self.pathlist[idx]
        line_path = self.sketch_path / line_path.name

        rnd = np.random.randint(self.pathlen)
        style_path = self.pathlist[rnd]

        return line_path, style_path


class LineTestCollator:
    def __init__(self):
        pass

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
        line = cv.imread(str(image_path))
        line = cv.resize(line, (192, 256), interpolation=cv.INTER_CUBIC)
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
