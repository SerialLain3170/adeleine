import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab


class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def _coordinate(array):
        tmp = np.clip(array * 127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def __call__(self, y, t, m, outdir, epoch, validsize):
        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(validsize):
            tmp = self._coordinate(m[index, 0:3])
            pylab.subplot(validsize, validsize, validsize * index + 1)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._coordinate(m[index, 3:6])
            pylab.subplot(validsize, validsize, validsize * index + 2)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._coordinate(t[index])
            pylab.subplot(validsize, validsize, validsize * index + 3)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = self._coordinate(y[index])
            pylab.subplot(validsize, validsize, validsize * index + 4)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")