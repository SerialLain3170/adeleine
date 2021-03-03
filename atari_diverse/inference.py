import torch
import yaml
import cv2 as cv
import numpy as np
import torch.nn as nn
import argparse

from xdog import XDoG
from pathlib import Path
from tqdm import tqdm
from model import Generator
from bicyclegan import Generator as BicycleGAN
from torch.utils.data import DataLoader
from dataset import IllustTestDataset, LineTestCollator
from visualize import Visualizer

import matplotlib
matplotlib.use("Agg")
import pylab


def prepare(path, batch, inferdir, index):
    color = cv.imread(str(path))
    line = XDoG()(str(path))

    height, width = line.shape[0], line.shape[1]
    line = line[:height-100, 50: width-50]

    cv.imwrite(f"{inferdir}/line_{index}.png", line*255)

    #print(line.shape)

    #line = cv.cvtColor(line, cv.COLOR_BGR2YCrCb)
    #line = np.tile(np.expand_dims(line, axis=0), (3, 1, 1))
    #line = line.transpose(2, 0, 1).astype(np.float32)
    #line = np.tile(np.expand_dims(line, axis=0), (batch, 1, 1, 1))
    #line = (line - 127.5) / 127.5

    #line = torch.cuda.FloatTensor(line)

    return line


def prepare_multi(path, index, validsize):
    line = cv.imread(f"{path}/line_{index+1}.png")
    hint = cv.imread(f"{path}/hint_{index+1}.png")

    line = cv.resize(line, (512, 512), interpolation=cv.INTER_CUBIC)
    line = line[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    line = np.tile(np.expand_dims(line, axis=0), (validsize, 1, 1, 1))
    line = (line - 127.5) / 127.5
    line = torch.cuda.FloatTensor(line)

    hint = cv.resize(hint, (512, 512), interpolation=cv.INTER_CUBIC)
    hint = hint[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    hint = np.tile(np.expand_dims(hint, axis=0), (validsize, 1, 1, 1))
    hint = (hint - 127.5) / 127.5
    hint = torch.cuda.FloatTensor(hint)

    return line, hint


def interpolate_noise_generate():
    noise_list = []
    for style in range(8):
        for point in range(8):
            basic_code = np.random.normal(size=(1, 8)).astype(np.float32)
            value = -1 + 0.25*point
            basic_code[:, style] = value

            noise_list.append(basic_code)

    return noise_list


def line_art_create(data_path):
    pathlist = list(data_path.glob("*.png"))
    print(len(pathlist))

    for index, path in enumerate(pathlist):
        if index >= 51:
            line = prepare(path, 1, "hint", index)

        if index == 200:
            break


def infer_multi(config,
                data_path,
                pre_flatter_path,
                outdir,
                pre_shader_path,
                ):

    pathlist = list(data_path.glob("*.png"))
    print(len(pathlist))

    weight = torch.load(pre_flatter_path)
    flatter = Generator(in_ch=6)
    flatter.load_state_dict(weight)
    flatter.cuda()
    flatter.eval()

    weight = torch.load(pre_shader_path)
    shader = BicycleGAN(in_ch=3, latent_dim=8)
    shader.load_state_dict(weight)
    shader.cuda()
    shader.eval()

    if config["inference"]["mode"] == "interpolate":
        for index, path in enumerate(tqdm(pathlist)):
            line, hint = prepare_multi("./input", index, 1)
            noise_list = interpolate_noise_generate()

            pylab.rcParams['figure.figsize'] = (16.0, 16.0)
            pylab.clf()

            for i, noise in enumerate(noise_list):
                noise = torch.cuda.FloatTensor(noise)
                with torch.no_grad():
                    x = torch.cat([line, hint], dim=1)
                    y = flatter(x, hint)
                    _, y = shader(y, hint, noise)

                y = y.detach().cpu().numpy()

                tmp = np.clip(y[0]*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
                #tmp = cv.cvtColor(tmp, cv.COLOR_YCrCb2RGB)
                pylab.subplot(8, 8, i + 1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig(f"{outdir}/visaulize_{index}.png")

                if index == 14:
                    break

    else:
        for index, path in enumerate(tqdm(pathlist)):
            line, hint = prepare_multi("./input", index, 16)

            with torch.no_grad():
                x = torch.cat([line, hint], dim=1)
                flat = flatter(x, hint)
                _, y = shader(flat, hint)

            y = y.detach().cpu().numpy()
            flat = flat[0].detach().cpu().numpy()

            pylab.rcParams['figure.figsize'] = (16.0, 16.0)
            pylab.clf()

            flat = np.clip(flat*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            flat = flat[:, :, ::-1]
            cv.imwrite(f"{outdir}/flat_{index}.png", flat)

            for i in range(16):
                tmp = np.clip(y[i]*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
                #tmp = cv.cvtColor(tmp, cv.COLOR_YCrCb2RGB)
                pylab.subplot(4, 4, i + 1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig(f"{outdir}/visaulize_{index}.png")

            if index == 14:
                break


def infer(config,
          data_path,
          pretrained_path,
          outdir,
          pre_shader_path=None
          ):

    # config
    model_config = config["model"]
    color_space = config["dataset"]["color_space"]
    train_type = config["train"]["train_type"]

    # Dataset Definition
    dataset = IllustTestDataset(data_path)
    collator = LineTestCollator(color_space)

    # Model & Optimizer Definition
    model = Generator(attn_type=model_config["flat_generator"]["attn_type"],
                      in_ch=model_config["flat_generator"]["in_ch"],
                      num_layers=model_config["flat_generator"]["num_layers"],
                      )
    weight = torch.load(pretrained_path)
    model.load_state_dict(weight)
    model.cuda()
    model.eval()

    if train_type == "shade":
        shader = Generator(model_config["shade_generator"]["in_ch"],
                           num_layers=model_config["shade_generator"]["num_layers"],
                           attn_type=model_config["shade_generator"]["attn_type"],
                           )
        weight = torch.load(pre_shader_path)
        shader.load_state_dict(weight)
        shader.cuda()
        shader.eval()

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collator,
                            drop_last=True)
    progress_bar = tqdm(dataloader)

    for index, data in enumerate(progress_bar):
        l_test, m_test = data
        x_test = torch.cat([l_test, m_test], dim=1)
        with torch.no_grad():
            y = model(x_test, m_test)

            if train_type == "shade":
                x = torch.cat([y, m_test], dim=1)
                y = shader(x, m_test)
                #y = shader(x_test, y)

        y = y[0].detach().cpu().numpy()

        if color_space == "yuv":
            tmp = y.transpose(1, 2, 0)
            tmp = np.clip(tmp*127.5 + 127.5, 0, 255)
            tmp = tmp.astype(np.uint8)
            y = cv.cvtColor(tmp, cv.COLOR_YCrCb2BGR)
        else:
            y = np.clip(y*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            y = y[:, :, ::-1]
        cv.imwrite(f"{outdir}/{index}.png", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style2Paint")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--outdir', type=Path, default="inferdir", help="output directory")
    parser.add_argument('--pre', type=Path, help="pretrain model path")
    parser.add_argument('--s_pre', type=Path, help="pretrain shader model path")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)

    infer_multi(config, args.data_path, args.pre, outdir, args.s_pre)
    #line_art_create(args.data_path)
