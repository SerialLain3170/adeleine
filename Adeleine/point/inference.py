import torch
import torch.nn as nn
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path
from tqdm import tqdm
from .model import Generator
from .utils import E2EGuidedFilter


class PointInferer:
    def __init__(self, pretrain_path):
        self.model = self._model_load(pretrain_path)
        self.mean = np.array([181.9935, 169.014, 166.2345]).astype(np.float32)
        self.std = np.array([75.735, 76.9335, 75.9645]).astype(np.float32)

        self.out_filter = E2EGuidedFilter(radius=1, eps=1e-2)
        self.out_filter.cuda()

    @staticmethod
    def _model_load(pretrain_path):
        model = Generator(in_ch=5,
                          base=64,
                          num_layers=10,
                          up_layers=[10, 5, 5, 3],
                          guide=True,
                          resnext=True,
                          encoder_type="res")
        weight = torch.load(pretrain_path)
        model.load_state_dict(weight)
        model.cuda()
        model.eval()

        return model

    def _coordinate(self,
                    img: np.array,
                    color_space="rgb",
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

        img = np.expand_dims(img, axis=0)
        img = torch.cuda.FloatTensor(img)

        return img

    @staticmethod
    def _normalize(img: np.array):
        img = np.expand_dims(img[:, :, 0], axis=0).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        img = torch.cuda.FloatTensor(img)

        return img

    @staticmethod
    def _denorm(img: torch.Tensor):
        img = img[0].detach().cpu().numpy()
        img = np.clip(img*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return img

    def __call__(self, line, mask, exist):
        line_x = self._coordinate(line, color_space="gray")
        line_m = self._coordinate(line, imagenet_mean=True)
        mask = self._coordinate(mask)
        exist = self._normalize(exist)
        x = torch.cat([line_x, mask, exist], dim=1)

        with torch.no_grad():
            y, _, _ = self.model(x, line_m)
            y = self.out_filter(line_x, y)

        y = self._denorm(y)

        return y
