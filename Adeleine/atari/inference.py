import torch
import torch.nn as nn
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path
from tqdm import tqdm
from .model import Generator


class AtariInferer:
    def __init__(self, pretrain_path):
        self.model = self._model_load(pretrain_path)

    @staticmethod
    def _model_load(pretrain_path):
        model = Generator()
        weight = torch.load(pretrain_path)
        model.load_state_dict(weight)
        model.cuda()
        model.eval()

        return model

    @staticmethod
    def _preprocess(img):
        #img = cv.resize(img, (192, 256), interpolation=cv.INTER_CUBIC)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5
        img = np.expand_dims(img, axis=0)
        img = torch.cuda.FloatTensor(img.astype(np.float32))

        return img

    @staticmethod
    def _denorm(img: torch.Tensor):
        img = img[0].detach().cpu().numpy()
        img = np.clip(img*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return img

    def __call__(self, line, atari):
        line = self._preprocess(line)
        atari = self._preprocess(atari)

        x = torch.cat([line, atari], dim=1)

        with torch.no_grad():
            y = self.model(x, atari)

        y = self._denorm(y)
        line = self._denorm(line)
        atari = self._denorm(atari)

        return y, line, atari
