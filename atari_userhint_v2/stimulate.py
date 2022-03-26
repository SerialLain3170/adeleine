import numpy as np
import cv2 as cv
import random
import colorsys

from pathlib import Path
from PIL import Image
from typing import List


class Stimulator:
    def __init__(self,
                 train_list: List[Path],
                 st_path: Path,
                 overall_size: int,
                 spray_size: int,
                 spray_split: int,
                 overlap_num: int):

        self.train_list = train_list
        self.train_len = len(self.train_list)
        self.st_path = st_path
        self.overall_size = overall_size
        self.spray_size = spray_size
        self.spray_split = spray_split
        self.overlap_num = overlap_num
        self.overlap_min = 20
        self.overlap_max = 100

    @staticmethod
    def _overlap_crop(color: np.array,
                      overall_size: int,
                      size: int) -> np.array:
        rnd1 = np.random.randint(overall_size - 1 - size)
        rnd2 = np.random.randint(overall_size - 1 - size)

        return color[rnd1: rnd1 + size, rnd2: rnd2 + size]

    @staticmethod
    def _overlap(color: np.array,
                 cropped: np.array,
                 overall_size: int,
                 size: int) -> np.array:
        rnd1 = np.random.randint(overall_size - 1 - size)
        rnd2 = np.random.randint(overall_size - 1 - size)
        color[rnd1: rnd1 + size, rnd2: rnd2 + size] = cropped

        return color

    def _making_overlap(self,
                        color: np.array,
                        overall_size: int,
                        iteration: int) -> np.array:
        for _ in range(iteration):
            image_path = self.train_list[np.random.randint(self.train_len)]
            overlap_color = cv.imread(str(image_path))
            size = np.random.randint(self.overlap_min, self.overlap_max)
            color_crop = self._overlap_crop(overlap_color, overall_size, size)
            color = self._overlap(color, color_crop, overall_size, size)

        return color

    @staticmethod
    def _min_dis(point, point_list):
        dis = []
        for p in point_list:
            dis.append(np.sqrt(np.sum(np.square(np.array(point)-np.array(p)))))

        return min(dis)

    @staticmethod
    def _get_dominant_color(image):
        image = image.convert('RGBA')
        image.thumbnail((200, 200))

        max_score = 0
        dominant_color = 0

        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
            if a == 0:
                continue

            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            if y > 0.9:
                continue
            if ((r > 230) & (g > 230) & (b > 230)) or ((r < 30) & (g < 30) & (b < 30)):
                continue
            # Calculate the score, preferring highly saturated colors.
            # Add 0.1 to the saturation so we don't completely ignore grayscale
            # colors by multiplying the count by zero, but still give them a low
            # weight.
            score = (saturation + 0.1) * count
            if score > max_score:
                max_score = score
                dominant_color = (r, g, b)

        return dominant_color

    def _spray(self, color_img, iteration=2, eps=1e-8):
        # To get dominant color, conversion array to pillow object
        img_pillow = Image.fromarray(color_img)
        color = self._get_dominant_color(img_pillow)

        # To process this method, conversion pillow object to array
        img = np.array(color_img)
        h = int(self.spray_size/self.spray_split)
        w = int(self.spray_size/self.spray_split)
        a_x = np.random.randint(0, h)
        a_y = np.random.randint(0, w)
        b_x = np.random.randint(0, h)
        b_y = np.random.randint(0, w)
        begin_point = np.array([min(a_x, b_x), a_y])
        end_point = np.array([max(a_x, b_x), b_y])
        tan = (begin_point[1] - end_point[1]) / (begin_point[0] - end_point[0] + 0.001)

        center_point_list = []
        for i in range(begin_point[0], end_point[0]+1):
            a = i
            b = (i-begin_point[0])*tan + begin_point[1]
            center_point_list.append(np.array([int(a), int(b)]))
        center_point_list = np.array(center_point_list)

        lamda = random.uniform(0.01, 10)
        paper = np.zeros((h, w, 3))
        mask = np.zeros((h, w))
        center = [int(h/2), int(w/2)]
        paper[center[0], center[1], :] = color
        for i in range(h):
            for j in range(w):
                dis = self._min_dis([i, j], center_point_list)
                paper[i, j, :] = np.array(color)/(np.exp(lamda*dis) + eps)
                mask[i, j] = np.array([255])/(np.exp(lamda*dis) + eps)

        paper = (paper).astype('uint8')
        mask = (mask).astype('uint8')

        mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
        im = cv.resize(paper, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)

        # To implement paste method, conversion array to pillow object
        imq = Image.fromarray(im)
        imp = img_pillow.copy()

        imp.paste(imq, (0, 0, imp.size[0], imp.size[1]), mask=Image.fromarray(mask))
        imp = np.asarray(imp)

        return imp

    def _spatial_transformer(self, st_img: np.array, overall_size: int) -> np.array:
        st_img = cv.resize(st_img, (overall_size, overall_size))

        return st_img

    def __call__(self, st_img) -> np.array:
        st_img = self._spatial_transformer(st_img, self.overall_size)
        st_img = self._making_overlap(st_img, self.overall_size, self.overlap_num)
        st_img = self._spray(st_img)

        return st_img
