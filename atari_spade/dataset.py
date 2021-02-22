import torch
import numpy as np
import cv2 as cv
import copy

from typing import List
from typing_extensions import Literal
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from hint_processor import LineProcessor

LineArt = List[Literal["xdog", "pencil", "digital", "blend"]]


def noise_generate(batchsize: int,
                   latent_dim: int) -> torch.Tensor:
    z = np.random.uniform(-1, 1, (batchsize, latent_dim)).astype(np.float32)
    z = torch.cuda.FloatTensor(z)

    return z


class BuildDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 line_method: LineArt,
                 extension=".jpg",
                 train_size=512,
                 valid_size=512,
                 color_space="rgb",
                 line_space="rgb"):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

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

    # Preprocess method
    @staticmethod
    def _random_crop(line: np.array,
                     color: np.array,
                     size: int) -> (np.array, np.array):

        scale = np.random.randint(516, 768)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

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

    # Hint preparation method
    @staticmethod
    def _making_mask(mask: np.array,
                     color: np.array,
                     size: int) -> np.array:

        choice = np.random.choice(['width', 'height', 'diag'])

        if choice == 'width':
            rnd_height = np.random.randint(8, 32)
            rnd_width = np.random.randint(8, 128)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'height':
            rnd_height = np.random.randint(8, 128)
            rnd_width = np.random.randint(8, 32)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

        elif choice == 'diag':
            rnd_height = np.random.randint(8, 32)
            rnd_width = np.random.randint(8, 128)

            rnd1 = np.random.randint(size - rnd_height - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            for index in range(rnd_width):
                mask[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index: rnd1 + rnd_height + index, rnd2 + index]

        return mask

    def _prepare_pair(self,
                      image_path: Path,
                      size: int):

        color = cv.imread(str(image_path))
        line = self.line_process(image_path)
        line, color = self._random_crop(line, color, size=size)

        # Hint preparation
        mask = copy.copy(line)
        repeat = np.random.randint(12, 20)
        for _ in range(repeat):
            mask = self._making_mask(mask, color, size=size)

        color = self._coordinate(color, self.color_space)
        line = self._coordinate(line, self.line_space)
        mask = self._coordinate(mask, self.color_space)

        return (color, line, mask)

    @staticmethod
    def _totensor(array_list: List[np.array]) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    @staticmethod
    def _fix(tensor: torch.Tensor, validsize: int) -> torch.Tensor:
        first = tensor[1].unsqueeze(0)
        tensor = first.repeat(validsize, 1, 1, 1)

        return tensor

    def valid(self, validsize):
        c_v_box = []
        l_v_box = []
        m_v_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color, line, mask = self._prepare_pair(color_path, self.valid_size)

            c_v_box.append(color)
            l_v_box.append(line)
            m_v_box.append(mask)

        c = self._totensor(c_v_box)
        l = self._totensor(l_v_box)
        m = self._totensor(m_v_box)

        c_fix = self._fix(c, validsize)
        l_fix = self._fix(l, validsize)
        m_fix = self._fix(m, validsize)

        return c, l, m, c_fix, l_fix, m_fix

    def __getitem__(self, idx):
        color_path = self.train_list[idx]
        color, line, mask = self._prepare_pair(color_path, self.train_size)

        return color, line, mask
