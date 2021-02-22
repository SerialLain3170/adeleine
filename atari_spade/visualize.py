import numpy as np
import torch

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def _denorm(img_array: torch.Tensor):
        tmp = img_array.detach().cpu().numpy()
        tmp = np.clip(tmp*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def _save(self,
              img: np.array,
              width: int,
              index: int,
              outdir: Path,
              epoch: int):

        tmp = self._denorm(img)
        pylab.subplot(width, width, index)
        pylab.imshow(tmp)
        pylab.axis("off")
        pylab.savefig(f"{outdir}/visualize_{epoch}.png")

    def __call__(self, v_list, y, outdir, epoch, testsize):
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            self._save(v_list[0][index], testsize, 4*index+1, outdir, epoch)
            self._save(v_list[1][index], testsize, 4*index+2, outdir, epoch)
            self._save(v_list[2][index], testsize, 4*index+3, outdir, epoch)
            self._save(y[index], testsize, 4*index+4, outdir, epoch)