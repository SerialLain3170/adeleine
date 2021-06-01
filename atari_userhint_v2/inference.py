import torch
import yaml
import cv2 as cv
import numpy as np
import torch.nn as nn
import argparse

from pathlib import Path
from tqdm import tqdm
from model import Generator
from torch.utils.data import DataLoader
from dataset import IllustDataset, LineTestCollator
from visualize import Visualizer
from utils import GuidedFilter


def infer(config,
          data_path,
          sketch_path,
          ss_path,
          pretrained_path,
          outdir
          ):

    def convert(tmp):
        tmp = tmp[0].detach().cpu().numpy()

        if len(tmp.shape) == 2:
            tmp = np.tile(np.expand_dims(tmp, axis=0), (3, 1, 1))
        height, width = tmp.shape[1], tmp.shape[2]
        mean = np.tile(mean_org.reshape(3, 1, 1), (1, height, width))
        std = np.tile(std_org.reshape(3, 1, 1), (1, height, width))
        tmp = np.clip(tmp*std + mean, 0, 255).transpose(1, 2, 0).astype(np.uint8)
        tmp = tmp[:, :, ::-1]

        return tmp

    # config
    model_config = config["model"]
    data_config = config["dataset"]
    out_guide = config["train"]["out_guide"]
    guide_flag = model_config["generator"]["guide"]
    color_space = config["dataset"]["color_space"]

    # Dataset Definition
    dataset = IllustDataset(data_path,
                            sketch_path,
                            ss_path,
                            data_config["line_method"],
                            data_config["extension"],
                            data_config["train_size"],
                            data_config["valid_size"],
                            data_config["color_space"],
                            data_config["line_space"])

    # Model & Optimizer Definition
    model = Generator(in_ch=model_config["generator"]["in_ch"],
                      base=model_config["generator"]["base"],
                      num_layers=model_config["generator"]["num_layers"],
                      up_layers=model_config["generator"]["up_layers"],
                      guide=model_config["generator"]["guide"],
                      resnext=model_config["generator"]["resnext"],
                      encoder_type=model_config["generator"]["encoder_type"])
    weight = torch.load(pretrained_path)
    model.load_state_dict(weight)
    model.cuda()
    model.eval()

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=True)
    progress_bar = tqdm(dataloader)

    out_filter = GuidedFilter(r=1, eps=1e-2)
    out_filter.cuda()

    #mean_org = np.array([181.9935, 169.014, 166.2345]).astype(np.float32)
    #std_org = np.array([75.735, 76.9335, 75.9645]).astype(np.float32)
    mean_org = np.array([127.5, 127.5, 127.5]).astype(np.float32)
    std_org = np.array([127.5, 127.5, 127.5]).astype(np.float32)

    for index, data in enumerate(progress_bar):
        c_test, l_test, m_test, l_m_test = data
        c_test = c_test.cuda()
        l_test = l_test.cuda()
        m_test = m_test.cuda()
        l_m_test = l_m_test.cuda()
        x_test = torch.cat([l_test, m_test], dim=1)
        with torch.no_grad():
            if guide_flag:
                y, _, _ = model(x_test, l_m_test)
            else:
                y = model(x_test, l_m_test)

        y = out_filter(l_test, y)

        if color_space == "yuv":
            tmp = y.transpose(1, 2, 0)
            tmp = np.clip(tmp*127.5 + 127.5, 0, 255)
            tmp = tmp.astype(np.uint8)
            y = cv.cvtColor(tmp, cv.COLOR_YCrCb2BGR)
        else:
            l = convert(l_test)
            m = convert(m_test[:, :3, :, :])
            y = convert(y)
            y = np.concatenate([l, m, y], axis=1)
        cv.imwrite(f"{outdir}/{index}.png", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style2Paint")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing line arts")
    parser.add_argument('--ss_path', type=Path, help="path containing quantized arts")
    parser.add_argument('--outdir', type=Path, default="inferdir", help="output directory")
    parser.add_argument('--pre', type=Path, help="pretrain model path")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)

    infer(config, args.data_path, args.sketch_path, args.ss_path,args.pre, outdir)