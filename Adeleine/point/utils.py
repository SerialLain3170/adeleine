import torch
import torch.nn as nn
import shutil
import datetime

from pathlib import Path
from torch.autograd import Variable
from zmq import device


def session(session_name):
    session_path = Path("session") / Path(session_name)
    if session_path.exists():
        dt = datetime.datetime.now()
        dt = dt.strftime('%m%d-%H%M-%S%f')[:-4]
        session_name = f"{session_name}.{dt}"
        session_path = Path("session") / Path(session_name)

    modeldir_path = session_path / "ckpts"
    outdir_path = session_path / "vis"

    modeldir_path.mkdir(exist_ok=True, parents=True)
    outdir_path.mkdir(exist_ok=True, parents=True)

    shutil.copy("param.yaml", session_path)

    return outdir_path, modeldir_path


class MeanFilter(nn.Module):
    def __init__(self, radius):
        super(MeanFilter, self).__init__()

        self.radius = radius

    @staticmethod
    def xdiff(h, radius):
        l = h[:, :, radius: 2*radius+1]
        m = h[:, :, 2*radius+1:] - h[:, :, :-2*radius-1]
        r = h[:, :, -1:] - h[:, :, -2*radius-1: -radius-1]

        return torch.cat([l, m, r], dim=2)

    @staticmethod
    def ydiff(h, radius):
        l = h[:, :, :, radius: 2*radius+1]
        m = h[:, :, :, 2*radius+1:] - h[:, :, :, :-2*radius-1]
        r = h[:, :, :, -1:] - h[:, :, :, -2*radius-1: -radius-1]

        return torch.cat([l, m, r], dim=3)

    def forward(self, x):
        return self.ydiff(self.xdiff(x.cumsum(dim=2), self.radius).cumsum(dim=3), self.radius)


class E2EGuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super(E2EGuidedFilter, self).__init__()

        self.radius = radius
        self.eps = eps
        self.mean_filter = MeanFilter(radius)

    def forward(self, x, y):
        bx, cx, hx, wx = x.size()

        mask = torch.ones((1, 1, hx, wx), device=x.device, dtype=x.dtype)
        nump = self.mean_filter(mask)

        x_mean = self.mean_filter(x) / nump
        y_mean = self.mean_filter(y) / nump
        xy_cov = self.mean_filter(x * y) / nump - (x_mean * y_mean)
        x_var = self.mean_filter(x * x) / nump - (x_mean * x_mean)

        a = xy_cov / (x_var + self.eps)
        a_mean = self.mean_filter(a) / nump
        b_mean = self.mean_filter(y_mean - (a * x_mean)) / nump

        return a_mean * x + b_mean

