import torch
import numpy as np
import cv2 as cv

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pass

    def _convert(self, img: torch.Tensor) -> np.array:
        img = img.detach().cpu().numpy()
        img = np.clip(img*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return img

    def _draw(self,
              img: np.array,
              width: int,
              index: int,
              outdir: Path,
              iteration: int
              ):

        img = self._convert(img)
        pylab.subplot(width, width, index)
        pylab.imshow(img)
        pylab.axis("off")
        pylab.savefig(f"{outdir}/visualize_{iteration}.png")

    def __call__(self,
                 v_list: List[torch.Tensor],
                 ysim: torch.Tensor,
                 ymid: torch.Tensor,
                 y: torch.Tensor,
                 outdir: Path,
                 iteration: int,
                 validsize: int):

        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(validsize):
            self._draw(v_list[0][index], validsize, validsize * index + 1, outdir, iteration)
            self._draw(v_list[3][index], validsize, validsize * index + 2, outdir, iteration)
            self._draw(v_list[4][index], validsize, validsize * index + 3, outdir, iteration)
            self._draw(v_list[5][index], validsize, validsize * index + 4, outdir, iteration)
            self._draw(ysim[index], validsize, validsize * index + 5, outdir, iteration)
            self._draw(ymid[index], validsize, validsize * index + 6, outdir, iteration)
            self._draw(y[index], validsize, validsize * index + 7, outdir, iteration)


class InferenceVisualizer:
    def __init__(self,
                 outdir):

        self.outdir = outdir

    def _convert(self, img):
        img = np.clip(img*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return img

    def _draw(self, img, index, iteration):
        img = self._convert(img)
        pylab.subplot(1, 4, index)
        pylab.imshow(img)
        pylab.axis("off")
        pylab.savefig(f"{self.outdir}/visualize_{iteration}.png")

    def _save(self, img, index, iteration, suffix):
        img = self._convert(img)
        img = img[:, :, ::-1]
        cv.imwrite(f"{self.outdir}/{suffix}_{iteration}.png", img)

    def __call__(self, c_y0, c_y1, l, c, ytr, iteration):
        pylab.rcParams['figure.figsize'] = (24.0, 6.0)
        pylab.clf()

        self._draw(c_y0[0], 1, iteration)
        self._draw(c_y1[0], 2, iteration)
        self._draw(l[0], 3, iteration)
        self._draw(c[0], 4, iteration)

        self._save(c_y0[0], 1, iteration, "first")
        self._save(c_y1[0], 2, iteration, "last")
        self._save(c[0], 4, iteration, "constraint")
        self._save(ytr[0], 4, iteration, "transform")
