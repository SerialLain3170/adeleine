import torch
import numpy as np
import cv2 as cv
import copy

from typing import List
from typing_extensions import Literal
from torch.utils.data import Dataset
from pathlib import Path
from hint_processor import LineProcessor
from torchvision.transforms import Normalize
from stimulate import Stimulator
from utils import change_saturate, double_change_saturate

LineArt = List[Literal["xdog", "pencil", "digital", "blend"]]


class IllustDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 ss_path: Path,
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
        self.ss_path = ss_path

        self.train_size = train_size
        self.valid_size = valid_size

        self.line_process = LineProcessor(sketch_path, line_method)
        self.color_space = color_space
        self.line_space = line_space

        self.mean = np.array([181.9935, 169.014, 166.2345]).astype(np.float32)
        self.std = np.array([75.735, 76.9335, 75.9645]).astype(np.float32)

    @staticmethod
    def _train_val_split(pathlist: List) -> (List, List):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    def _coordinate(self,
                    img: np.array,
                    color_space: str,
                    imagenet_mean=False) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = img = img.transpose(2, 0, 1)

        if imagenet_mean:
            height, width = img.shape[1], img.shape[2]
            mean = np.tile(self.mean.reshape(3, 1, 1), (1, height, width))
            std = np.tile(self.std.reshape(3, 1, 1), (1, height, width))
            img = (img - mean) / std
        else:
            img = (img - 127.5) / 127.5

        return img

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    @staticmethod
    def _random_crop(line: np.array,
                     color: np.array,
                     ss: np.array,
                     size: int) -> (np.array, np.array):
        scale = np.random.randint(516, 518)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))
        ss = cv.resize(color, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]
        ss = ss[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color, ss

    @staticmethod
    def _random_coord_crop(line: np.array,
                           color: np.array,
                           ss: np.array,
                           size: int) -> (np.array, np.array):
        height, width = color.shape[0], color.shape[1]
        if height > width:
            scale = 514 / width
        else:
            scale = 514 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        line = cv.resize(line, (new_width, new_height))
        color = cv.resize(color, (new_width, new_height))
        ss = cv.resize(color, (new_width, new_height))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]
        ss = ss[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color, ss

    # Hint preparation method
    @staticmethod
    def _making_mask(mask: np.array,
                     color: np.array,
                     alpha: np.array,
                     size: int) -> np.array:

        def adjust(length, max_length, patch_size):
            if length + patch_size > max_length - 1:
                length = max_length - 2 - patch_size

            if length < 0:
                length = 0

            return length

        height, width = mask.shape[0], mask.shape[1]

        rnd_h = int(np.random.normal(loc=int(height/2), scale=(int(height/4))))
        rnd_w = int(np.random.normal(loc=int(width/2), scale=(int(width/4))))

        patch_size = np.random.randint(1, 13)

        rnd_h = adjust(rnd_h, height, patch_size)
        rnd_w = adjust(rnd_w, width, patch_size)

        patch = color[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :]
        patch = np.mean(patch, axis=(0, 1), keepdims=True).astype(np.uint8)
        patch = np.tile(patch, (patch_size, patch_size, 1))
        exist = np.ones((patch_size, patch_size, 1))
        mask[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :] = patch
        alpha[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :] = exist

        return mask, alpha

    def _preprocess(self,
                    color: np.array,
                    line: np.array,
                    ss: np.array,
                    size: int):
        color = change_saturate(color)
        line, color, ss = self._random_coord_crop(line, color, ss, size=size)

        # Hint preparation
        if np.random.randint(100) == 0:
            alpha = np.ones_like(line)[:, :, 0:1]
            return (color, line, color, alpha, ss)
        mask = 255 * np.ones_like(line)
        alpha = np.zeros_like(line)[:, :, 0:1]
        repeat = np.random.randint(55, 60)
        for _ in range(repeat):
            mask, alpha = self._making_mask(mask, color, alpha, size=size)

        return (color, line, mask, alpha, ss)

    def valid(self, validsize: int):
        c_valid_box = []
        l_i_valid_box = []
        m_valid_box = []
        l_m_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            #color_path = np.random.choice(self.val_list)
            color = cv.imread(str(color_path))
            line = self.line_process(color_path)

            color, line, mask, alpha, _ = self._preprocess(color,
                                                           line,
                                                           line,
                                                           size=self.valid_size)

            color = self._coordinate(color, self.color_space, imagenet_mean=False)
            line_i = self._coordinate(line, self.line_space, imagenet_mean=False)
            mask = self._coordinate(mask, self.color_space, imagenet_mean=False)
            line_m = self._coordinate(line, self.color_space, imagenet_mean=True)
            alpha = alpha.transpose(2, 0, 1).astype(np.float32)
            mask = np.concatenate([mask, alpha], axis=0)

            c_valid_box.append(color)
            l_i_valid_box.append(line_i)
            m_valid_box.append(mask)
            l_m_valid_box.append(line_m)

        color = self._totensor(c_valid_box)
        line_i = self._totensor(l_i_valid_box)
        mask = self._totensor(m_valid_box)
        line_m = self._totensor(l_m_valid_box)

        return color, line_i, mask, line_m

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
        # Quantized prepare
        ss_name = self.ss_path / Path(color_path.name)
        ss = cv.imread(str(ss_name))

        color, line, mask, alpha, ss = self._preprocess(color,
                                                        line,
                                                        ss,
                                                        self.train_size)

        color = self._coordinate(color, self.color_space, imagenet_mean=False)
        line_i = self._coordinate(line, self.line_space, imagenet_mean=False)
        mask = self._coordinate(mask, self.color_space, imagenet_mean=False)
        line_m = self._coordinate(line, self.color_space, imagenet_mean=True)

        alpha = alpha.transpose(2, 0, 1).astype(np.float32)
        mask = np.concatenate([mask, alpha], axis=0)

        return color, line_i, mask, line_m


class IllustRefineDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 st_path: Path,
                 line_method: LineArt,
                 extension=".jpg",
                 train_size=224,
                 valid_size=256,
                 color_space="rgb",
                 line_space="rgb",
                 overall_size=512,
                 spray_size=256,
                 spray_split=7,
                 overlap_num=3):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.sketch_path = sketch_path
        self.st_path = st_path

        self.train_size = train_size
        self.valid_size = valid_size

        self.color_space = color_space
        self.line_space = line_space

        self.mean = np.array([181.9935, 169.014, 166.2345]).astype(np.float32)
        self.std = np.array([75.735, 76.9335, 75.9645]).astype(np.float32)

        self.line_process = LineProcessor(sketch_path, line_method)

        self.stimulator = Stimulator(self.train_list,
                                     st_path,
                                     overall_size,
                                     spray_size,
                                     spray_split,
                                     overlap_num)

    @staticmethod
    def _train_val_split(pathlist: List) -> (List, List):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    def _coordinate(self,
                    img: np.array,
                    color_space: str,
                    imagenet_mean=False) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = img = img.transpose(2, 0, 1)

        if imagenet_mean:
            height, width = img.shape[1], img.shape[2]
            mean = np.tile(self.mean.reshape(3, 1, 1), (1, height, width))
            std = np.tile(self.std.reshape(3, 1, 1), (1, height, width))
            img = (img - mean) / std
        else:
            img = (img - 127.5) / 127.5

        return img

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    @staticmethod
    def _random_crop(line: np.array,
                     color: np.array,
                     st: np.array,
                     size: int) -> (np.array, np.array):
        scale = np.random.randint(516, 518)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))
        st = cv.resize(st, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]
        st = st[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color, st

    @staticmethod
    def _random_coord_crop(line: np.array,
                           color: np.array,
                           ss: np.array,
                           size: int) -> (np.array, np.array):
        height, width = color.shape[0], color.shape[1]
        if height > width:
            scale = 514 / width
        else:
            scale = 514 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        line = cv.resize(line, (new_width, new_height))
        color = cv.resize(color, (new_width, new_height))
        ss = cv.resize(ss, (new_width, new_height))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]
        ss = ss[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color, ss

    # Hint preparation method
    @staticmethod
    def _making_mask(mask: np.array,
                     color: np.array,
                     alpha: np.array,
                     size: int) -> np.array:

        def adjust(length, max_length, patch_size):
            if length + patch_size > max_length - 1:
                length = max_length - 2 - patch_size

            if length < 0:
                length = 0

            return length

        height, width = mask.shape[0], mask.shape[1]

        rnd_h = int(np.random.normal(loc=int(height/2), scale=(int(height/4))))
        rnd_w = int(np.random.normal(loc=int(width/2), scale=(int(width/4))))

        patch_size = np.random.randint(1, 13)

        rnd_h = adjust(rnd_h, height, patch_size)
        rnd_w = adjust(rnd_w, width, patch_size)

        patch = color[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :]
        patch = np.mean(patch, axis=(0, 1), keepdims=True).astype(np.uint8)
        patch = np.tile(patch, (patch_size, patch_size, 1))
        exist = np.ones((patch_size, patch_size, 1))
        mask[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :] = patch
        alpha[rnd_h: rnd_h + patch_size, rnd_w: rnd_w + patch_size, :] = exist

        return mask, alpha

    def _st_load(self, st_path):
        filename = st_path.name
        filepath = self.st_path / filename
        st_img = cv.imread(str(filepath))

        while st_img is None:
            print(f"{filepath} is not found! ")
            st_img = cv.imread(str(st_path))

        return st_img

    def _preprocess(self,
                    color: np.array,
                    line: np.array,
                    st: np.array,
                    size: int):
        line, color, st = self._random_coord_crop(line, color, st, size=size)

        # Hint preparation
        if np.random.randint(100) == 0:
            alpha = np.ones_like(line)[:, :, 0:1]
            return (color, line, color, alpha, st)
        mask = 255 * np.ones_like(line)
        alpha = np.zeros_like(line)[:, :, 0:1]
        repeat = np.random.randint(55, 60)
        for _ in range(repeat):
            mask, alpha = self._making_mask(mask, color, alpha, size=size)

        return (color, line, mask, alpha, st)

    def valid(self, validsize: int):
        c_valid_box = []
        l_i_valid_box = []
        m_valid_box = []
        st_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            #color_path = np.random.choice(self.val_list)
            color = cv.imread(str(color_path))
            line = self.line_process(color_path)
            st = self._st_load(color_path)
            color, st = double_change_saturate(color, st)
            st = self.stimulator(st)

            color, line, mask, alpha, st = self._preprocess(color,
                                                            line,
                                                            st,
                                                            size=self.valid_size)

            color = self._coordinate(color, self.color_space, imagenet_mean=False)
            line_i = self._coordinate(line, self.line_space, imagenet_mean=False)
            mask = self._coordinate(mask, self.color_space, imagenet_mean=False)
            st = self._coordinate(st, self.color_space, imagenet_mean=True)
            alpha = alpha.transpose(2, 0, 1).astype(np.float32)
            mask = np.concatenate([mask, alpha], axis=0)

            c_valid_box.append(color)
            l_i_valid_box.append(line_i)
            m_valid_box.append(mask)
            st_valid_box.append(st)

        color = self._totensor(c_valid_box)
        line_i = self._totensor(l_i_valid_box)
        mask = self._totensor(m_valid_box)
        st = self._totensor(st_valid_box)

        return color, line_i, mask, st

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
        # Quantized prepare
        st = self._st_load(color_path)
        color, st = double_change_saturate(color, st)
        st = self.stimulator(st)

        color, line, mask, alpha, st = self._preprocess(color,
                                                        line,
                                                        st,
                                                        self.train_size)

        color = self._coordinate(color, self.color_space, imagenet_mean=False)
        line_i = self._coordinate(line, self.line_space, imagenet_mean=False)
        mask = self._coordinate(mask, self.color_space, imagenet_mean=False)
        st = self._coordinate(st, self.color_space, imagenet_mean=True)

        alpha = alpha.transpose(2, 0, 1).astype(np.float32)
        mask = np.concatenate([mask, alpha], axis=0)

        return color, line_i, mask, st


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
        self.mean = np.array([164.76139251, 167.47864617, 181.13838569]).astype(np.float32)
        self.std = np.array([127.5, 127.5, 127.5]).astype(np.float32)

    def _coordinate(self,
                    img: np.array,
                    color_space: str,
                    imagenet_mean: bool) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = img.transpose(2, 0, 1)

        if imagenet_mean:
            height, width = img.shape[1], img.shape[2]
            mean = np.tile(self.mean.reshape(3, 1, 1), (1, height, width))
            std = np.tile(self.std.reshape(3, 1, 1), (1, height, width))
            img = (img - mean) / std
        else:
            img = (img - 127.5) / 127.5

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

        mask = self._coordinate(mask, self.color_space, imagenet_mean=False)
        line = self._coordinate(line, self.color_space, imagenet_mean=False)

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
