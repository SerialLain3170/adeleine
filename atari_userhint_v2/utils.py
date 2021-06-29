import torch
import torch.nn as nn
import shutil
import datetime
import cv2
import numpy as np

from pathlib import Path
from torch.autograd import Variable


def session(session_name: str) -> (Path, Path):
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


# The implementation of Guided Filtering is from
# https://github.com/wuhuikai/DeepGuidedFilter/tree/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch
def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


def change_saturate(bgr_img: np.array) -> np.array:
    alpha = np.random.uniform(1.0, 2.0)
    hsvimage = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
    hsvf = hsvimage.astype(np.float32)
    hsvf[:, :, 1] = np.clip(hsvf[:, :, 1] * alpha, 0, 255)
    hsv8 = hsvf.astype(np.uint8)

    img = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR_FULL)

    return img


def double_change_saturate(bgr_img: np.array, st_img: np.array) -> (np.array, np.array):
    alpha = np.random.uniform(1.0, 2.0)

    def _change_saturate(img: np.array, alpha: float) -> np.array:
        hsvimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) # BGR->HSV
        hsvf = hsvimage.astype(np.float32)
        hsvf[:, :, 1] = np.clip(hsvf[:, :, 1] * alpha, 0, 255)
        hsv8 = hsvf.astype(np.uint8)

        img = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR_FULL)

        return img

    bgr_img = _change_saturate(bgr_img, alpha)
    st_img = _change_saturate(st_img, alpha)

    return bgr_img, st_img
