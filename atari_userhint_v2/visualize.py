import cv2 as cv
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import pylab

from typing import List
from pathlib import Path


class Visualizer:
    def __init__(self, color_space: str):
        self.color_space = color_space
        #self.mean = np.array([181.9935, 169.014, 166.2345]).astype(np.float32)
        #self.std = np.array([75.735, 76.9335, 75.9645]).astype(np.float32)
        self.mean = np.array([127.5, 127.5, 127.5]).astype(np.float32)
        self.std = np.array([127.5, 127.5, 127.5]).astype(np.float32)

    def _convert(self, img_array: torch.Tensor, alpha=False) -> np.array:
        tmp = img_array.detach().cpu().numpy()

        if tmp.shape[0] == 1:
            if alpha:
                tmp = tmp[0]*255.0
            else:
                tmp = tmp[0]*127.5 + 127.5
            tmp = tmp.astype(np.uint8)
            tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2RGB)

            return tmp

        if self.color_space == "yuv":
            tmp = tmp.transpose(1, 2, 0)
            tmp = tmp*127.5 + 127.5
            tmp = tmp.astype(np.uint8)
            tmp = cv.cvtColor(tmp, cv.COLOR_YCrCb2RGB)
        else:
            height, width = tmp.shape[1], tmp.shape[2]
            mean = np.tile(self.mean.reshape(3, 1, 1), (1, height, width))
            std = np.tile(self.std.reshape(3, 1, 1), (1, height, width))
            tmp = np.clip(tmp*std + mean, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def _save(self,
              img: torch.Tensor,
              width: int,
              index: int,
              outdir: Path,
              epoch: int,
              alpha=False):

        tmp = self._convert(img, alpha=alpha)
        pylab.subplot(width, width, index)
        pylab.imshow(tmp)
        pylab.axis("off")
        pylab.savefig(f"{outdir}/visualize_{epoch}.png")

    def __call__(self,
                 v_list: List[torch.Tensor],
                 y: torch.Tensor,
                 outdir: Path,
                 epoch: int,
                 testsize: int):

        width = testsize
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            self._save(v_list[0][index], width, 5*index+1, outdir, epoch)
            self._save(v_list[1][index][:3], width, 5*index+2, outdir, epoch)
            self._save(v_list[1][index][3:4], width, 5*index+3, outdir, epoch, alpha=True)
            self._save(v_list[2][index], width, 5*index+4, outdir, epoch)
            self._save(y[index], width, 5*index+5, outdir, epoch)
