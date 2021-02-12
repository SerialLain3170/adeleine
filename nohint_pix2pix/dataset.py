import torch
import numpy as np
import cv2 as cv
import copy

from typing import List
from typing_extensions import Literal
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
    def _train_val_split(pathlist: List) -> (List, List):
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
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    @staticmethod
    def _random_crop(line: np.array,
                     color: np.array,
                     size: int) -> (np.array, np.array):
        scale = np.random.randint(288, 768)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    # Hint preparation method
    @staticmethod
    def _making_mask(mask: np.array,
                     color: np.array,
                     size: int) -> np.array:
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

    def _preprocess(self,
                    color: np.array,
                    line: np.array,
                    size: int):
        line, color = self._random_crop(line, color, size=size)

        # Hint preparation
        mask = copy.copy(line)
        repeat = np.random.randint(8, 20)
        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)

        return (color, line, mask)

    def valid(self, validsize: int):
        c_valid_box = []
        l_valid_box = []
        m_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color = cv.imread(str(color_path))
            line = self.line_process(color_path)

            color, line, mask = self._preprocess(color,
                                                 line,
                                                 size=self.valid_size)

            color = self._coordinate(color, self.color_space)
            line = self._coordinate(line, self.line_space)
            mask = self._coordinate(mask, self.color_space)

            c_valid_box.append(color)
            l_valid_box.append(line)
            m_valid_box.append(mask)

        color = self._totensor(c_valid_box)
        line = self._totensor(l_valid_box)
        mask = self._totensor(m_valid_box)

        return color, line, mask

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        # Color prepare
        color_path = self.train_list[idx]
        color = cv.imread(str(color_path))
        # Line prepare
        line = self.line_process(color_path)

        color, line, mask = self._preprocess(color,
                                             line,
                                             self.train_size)

        color = self._coordinate(color, self.color_space)
        line = self._coordinate(line, self.line_space)
        mask = self._coordinate(mask, self.color_space)

        return color, line, mask


class IllustTestDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.pathlist = list(self.path.glob('test_*.png'))
        self.pathlen = len(self.pathlist)

    def __repr__(self):
        return f"dataset length: {self.pathlen}"

    def __len__(self):
        return self.pathlen

    def __getitem__(self, idx) -> (Path, Path):
        line_path = self.pathlist[idx]
        line_name = str(line_path.name)
        style_name = "hint_" + line_name[5:]
        style_path = self.path / Path(style_name)

        return line_path, style_path


class LineTestCollator:
    def __init__(self, color_space="rgb"):
        self.color_space = color_space

    @staticmethod
    def _coordinate(img: np.array,
                    color_space: str) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _totensor(array_list: List[np.array]) -> torch.Tensor:
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _prepare(self,
                 line_path: Path,
                 style_path: Path) -> (np.array, np.array):
        mask = cv.imread(str(style_path))
        line = cv.imread(str(line_path))

        mask = self._coordinate(mask, self.color_space)
        line = self._coordinate(line, self.color_space)

        return line, mask

    def __call__(self, batch):
        l_box = []
        m_box = []

        for l_path, s_path in batch:
            print(l_path, s_path)
            line, mask = self._prepare(l_path, s_path)

            l_box.append(line)
            m_box.append(mask)

        l = self._totensor(l_box)
        m = self._totensor(m_box)

        return (l, m)
