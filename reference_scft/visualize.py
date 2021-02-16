import numpy as np
import cv2 as cv
import torch

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self, color_space: str):
        self.color_space = color_space

    def _convert(self, img_array: torch.Tensor) -> np.array:
        tmp = img_array.detach().cpu().numpy()

        if tmp.shape[0] == 1:
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
            tmp = np.clip(tmp*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def _save(self,
              img: torch.Tensor,
              width: int,
              index: int,
              outdir: Path,
              epoch: int):
        tmp = self._convert(img)
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

        width = int(testsize / 2)
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            self._save(v_list[0][index], width, 3*index+1, outdir, epoch)
            self._save(v_list[1][index], width, 3*index+2, outdir, epoch)
            self._save(y[index], width, 3*index+3, outdir, epoch)


class TestVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def _convert(img_array):
        tmp = np.clip(img_array*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def _save(self, img, width, index, outdir, epoch):
        tmp = self._convert(img)
        pylab.subplot(width, width, index)
        pylab.imshow(tmp)
        pylab.axis("off")
        pylab.savefig(f"{outdir}/visualize_{epoch}.png")

    def __call__(self, y, y_comp, color, line, outdir, epoch, testsize):
        width = testsize
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            self._save(line[index], width, 4*index+1, outdir, epoch)
            self._save(color[index], width, 4*index+2, outdir, epoch)
            self._save(y_comp[index], width, 4*index+3, outdir, epoch)
            self._save(y[index], width, 4*index+4, outdir, epoch)
