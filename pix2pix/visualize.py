import numpy as np

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def _convert(img_array):
        tmp = np.clip(img_array*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def __call__(self, line, c, y, outdir, epoch, validsize):
        width = int(validsize / 2)
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(validsize):
            tmp = self._convert(line[index])
            pylab.subplot(width, width, 3 * index + 1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._convert(c[index])
            pylab.subplot(width, width, 3 * index + 2)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._convert(y[index])
            pylab.subplot(width, width, 3 * index + 3)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
