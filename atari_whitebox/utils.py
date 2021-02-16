import shutil
import datetime
import torch
import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch.autograd import Variable
from skimage import segmentation, color


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


# random_color_shift, label2rgb and simple_superpixel are from
# https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/utils.py
def random_color_shift(img1: torch.Tensor,
                       img2: torch.Tensor,
                       alpha=0.2) -> (torch.Tensor, torch.Tensor):
    b1, g1, r1 = img1[:, 0, :, :], img1[:, 1, :, :], img1[:, 2, :, :]
    b2, g2, r2 = img2[:, 0, :, :], img2[:, 1, :, :], img2[:, 2, :, :]

    b_w = np.random.uniform(0.114 - alpha, 0.114 + alpha)
    g_w = np.random.uniform(0.587 - alpha, 0.587 + alpha)
    r_w = np.random.uniform(0.299 - alpha, 0.299 + alpha)

    out1 = (b_w * b1 + g_w * g1 + r_w * r1) / (b_w + g_w + r_w)
    out2 = (b_w * b2 + g_w * g2 + r_w * r2) / (b_w + g_w + r_w)

    out1 = out1.unsqueeze(1)
    out2 = out2.unsqueeze(1)

    return out1, out2


def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):

    #std_list = list()
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        #std = np.std(image[mask])
        #std_list.append(std)
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5*mean + 0.5*median
            elif 40 < std:
                color = np.median(image[mask], axis=0)
        out[mask] = color

    return out


def simple_superpixel(image, seg_num=200):
    def process_slic(image):
        seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                      compactness=10, convert2lab=True)
        image = label2rgb(seg_label, image, kind='mix')
        return image

    return process_slic(image)


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
