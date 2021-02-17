import torch
import cv2 as cv
import numpy as np

from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset


class IllustDataset(Dataset):
    def __init__(self,
                 color_path: Path,
                 sketch_path: Path,
                 dist_path: Path,
                 anime_dir_list: List[str],
                 extension=".png",
                 train_size=256,
                 valid_size=256,
                 scale=512,
                 frame_range=8
                 ):

        super(IllustDataset, self).__init__()

        self.c_path = color_path
        self.s_path = sketch_path
        self.d_path = dist_path
        self.train_size = train_size
        self.valid_size = valid_size
        self.scale = scale
        self.frame_range = frame_range

        self.data_list = list(self.c_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.data_list)
        self.dir_list = anime_dir_list

        print(self.dir_list)

    def __repr__(self):
        return f"the number of dataset is {len(self.train_list)}"

    def __len__(self):
        return len(self.train_list)

    @staticmethod
    def _train_val_split(data_list: List) -> (List, List):
        data_len = len(data_list)
        split_point = int(data_len * 0.95)

        train_list = data_list[:split_point]
        val_list = data_list[split_point:]

        return train_list, val_list

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    @staticmethod
    def _coordinate(x: np.array) -> np.array:
        x = x.astype(np.float32)
        x = x[:, :, ::-1]
        x = x.transpose(2, 0, 1)
        x = (x - 127.5) / 127.5

        return x

    def _resize(self, x_list: List[np.array]) -> List[np.array]:
        cropped_list = []

        for img in x_list:
            img = cv.resize(img, (self.train_size, self.train_size))
            cropped_list.append(img)

        return cropped_list

    def _center_crop(self, img: np.array, movie: str) -> np.array:
        height, width = img.shape[0], img.shape[1]
        new_scale = float(self.scale / height)
        new_height = self.scale
        new_width = int(width * new_scale)
        new_center = int(new_width / 2)
        from_mat = cv.resize(img, (new_width, new_height))

        margin = int(self.scale / 2)
        from_mat = from_mat[:, new_center-margin: new_center+margin, :]

        return from_mat

    def _extract_range_from_txt(self,
                                inter_txt: str,
                                mode="train") -> (int, int, int):
        f = open(inter_txt)
        flist = list(f)
        flen = len(flist)

        if mode == "train":
            flist = flist[:int(flen * 0.95)]
        elif mode == "valid":
            flist = flist[int(flen * 0.95):]

        interval = np.random.choice(flist)
        interval = interval.rstrip("¥n").split(",")
        end, start = int(interval[0]), int(interval[1])

        start = np.random.randint(start, end - self.frame_range)
        end = start + self.frame_range
        focus = np.random.randint(start + 1, end)

        return start, end, focus

    def _img_load(self,
                  anime: str,
                  start: int,
                  end: int,
                  focus: int) -> List[np.array]:

        line_x_name = self.s_path / Path(anime) / Path(f"{str(focus)}.png")
        line_y0_name = self.s_path / Path(anime) / Path(f"{str(start)}.png")
        line_y1_name = self.s_path / Path(anime) / Path(f"{str(end)}.png")

        color_x_name = self.c_path / Path(anime) / Path(f"{str(focus)}.png")
        color_y0_name = self.c_path / Path(anime) / Path(f"{str(start)}.png")
        color_y1_name = self.c_path / Path(anime) / Path(f"{str(end)}.png")

        dist_x_name = self.d_path / Path(anime) / Path(f"{str(focus)}.png")
        dist_y0_name = self.d_path / Path(anime) / Path(f"{str(start)}.png")
        dist_y1_name = self.d_path / Path(anime) / Path(f"{str(end)}.png")

        line_x = cv.imread(str(line_x_name))
        line_y0 = cv.imread(str(line_y0_name))
        line_y1 = cv.imread(str(line_y1_name))

        color_x = cv.imread(str(color_x_name))
        color_y0 = cv.imread(str(color_y0_name))
        color_y1 = cv.imread(str(color_y1_name))

        color_x = self._center_crop(color_x, anime)
        color_y0 = self._center_crop(color_y0, anime)
        color_y1 = self._center_crop(color_y1, anime)

        dist_x = cv.imread(str(dist_x_name))
        dist_y0 = cv.imread(str(dist_y0_name))
        dist_y1 = cv.imread(str(dist_y1_name))

        return [line_x, line_y0, line_y1, color_x, color_y0, color_y1, dist_x, dist_y0, dist_y1]

    def valid(self, validsize: int) -> Tuple[int]:
        line_x = []
        line_y0 = []
        line_y1 = []
        color_x = []
        color_y0 = []
        color_y1 = []
        dist_x = []
        dist_y0 = []
        dist_y1 = []

        for _ in range(validsize):
            movie = np.random.choice(self.dir_list)
            inter_txt = self.c_path / Path(movie) / Path("separate.txt")
            start, end, focus = self._extract_range_from_txt(inter_txt)
            x_list = self._img_load(movie, start, end, focus)
            x_list = self._resize(x_list)
            x_list = [self._coordinate(x) for x in x_list]

            line_x.append(x_list[0])
            line_y0.append(x_list[1])
            line_y1.append(x_list[2])
            color_x.append(x_list[3])
            color_y0.append(x_list[4])
            color_y1.append(x_list[5])
            dist_x.append(x_list[6])
            dist_y0.append(x_list[7])
            dist_y1.append(x_list[8])

        line_x = self._totensor(line_x)
        line_y0 = self._totensor(line_y0)
        line_y1 = self._totensor(line_y1)
        color_x = self._totensor(color_x)
        color_y0 = self._totensor(color_y0)
        color_y1 = self._totensor(color_y1)
        dist_x = self._totensor(dist_x)
        dist_y0 = self._totensor(dist_y0)
        dist_y1 = self._totensor(dist_y1)

        return line_x, line_y0, line_y1, color_x, color_y0, color_y1, dist_x, dist_y0, dist_y1

    def __getitem__(self, idx):
        movie = np.random.choice(self.dir_list)
        inter_txt = self.c_path / Path(movie) / Path("separate.txt")
        start, end, focus = self._extract_range_from_txt(inter_txt)
        x_list = self._img_load(movie, start, end, focus)
        x_list = self._resize(x_list)
        x_list = [self._coordinate(x) for x in x_list]

        return x_list[0], x_list[1], x_list[2], x_list[3], x_list[4], x_list[5], x_list[6], x_list[7], x_list[8]
