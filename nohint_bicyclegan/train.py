import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from model import Generator, Discriminator, LatentEncoder
from torch.utils.data import DataLoader
from dataset import DanbooruFacesDataset
from visualize import Visualizer
from utils import first_making, noise_generate

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class DiverseColorizeLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:

        return torch.mean(torch.abs(y - t))

    @staticmethod
    def latent_constrain_loss(y: torch.Tensor,
                              t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    @staticmethod
    def adversarial_disloss(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_genloss(discriminator: nn.Module,
                            y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            loss = torch.mean(softplus(-fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_hingedis(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            sum_loss += nn.ReLU()(1.0 + fake).mean()
            sum_loss += nn.ReLU()(1.0 - real).mean()

        return sum_loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            sum_loss += -fake.mean()

        return sum_loss

    @staticmethod
    def positive_enforcing_loss(y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        batch, ch, h, w = y.size()

        for color in range(3):
            perch = y[:, color, :, :]
            mean = torch.mean(perch)
            mean = mean * torch.ones_like(mean)
            loss = torch.mean((perch-mean)**2)
            sum_loss += loss

        return -sum_loss

    @staticmethod
    def kl_loss(y: torch.Tensor) -> torch.Tensor:
        x_2 = torch.pow(y, 2)
        loss = torch.mean(x_2)

        return loss

    @staticmethod
    def mode_seeking_regularize(y: torch.Tensor,
                                z: torch.Tensor) -> torch.Tensor:
        batchsize = y.size(0)
        index = torch.randperm(batchsize).cuda()

        lz = torch.mean(torch.abs(y-y[index, :])) / (torch.mean(torch.abs(z-z[index, :])) + 1e-9)
        loss = 1 / (lz + 1e-5)

        return loss


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 outdir_fix,
                 modeldir,
                 data_path,
                 sketch_path=None
                 ):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.outdir_fix = outdir_fix
        self.modeldir = modeldir

        self.dataset = DanbooruFacesDataset(data_path,
                                            sketch_path,
                                            self.data_config["extension"],
                                            self.data_config["train_size"],
                                            self.data_config["valid_size"],
                                            self.data_config["color_space"],
                                            self.data_config["line_space"])

        print(self.dataset)

        gen = Generator(model_config["generator"]["in_ch"],
                        latent_dim=model_config["generator"]["l_dim"],
                        num_layers=model_config["generator"]["num_layers"],
                        )
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        dis = Discriminator(model_config["discriminator"]["in_ch"],
                            multi_pattern=model_config["discriminator"]["multi"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])
        enc = LatentEncoder(latent_dim=model_config["encoder"]["l_dim"])
        self.enc, self.enc_opt = self._setting_model_optim(enc,
                                                           model_config["encoder"])

        self.lossfunc = DiverseColorizeLossCalculator()
        self.visualizer = Visualizer(self.data_config["color_space"])

    @staticmethod
    def _valid_prepare(dataset,
                       validsize: int,
                       latent_dim: int) -> (List[torch.Tensor], List[torch.Tensor]):
        c_val, l_val, m_val = dataset.valid(validsize)
        l_fix, m_fix, c_fix = first_making(l_val, m_val, c_val)
        #x_fix = torch.cat([l_fix, m_fix], dim=1)
        #x_val = torch.cat([l_val, m_val], dim=1)
        x_fix = l_fix
        x_val = l_val
        z_fix = noise_generate(x_fix, latent_dim)
        z_val = noise_generate(x_fix, latent_dim)

        return [x_val, l_val, m_val, c_val, z_val], [x_fix, l_fix, m_fix, c_fix, z_fix]

    def _eval(self,
              iteration: int,
              validsize: int,
              v_list: List[torch.Tensor],
              outdir: Path):

        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/model_{iteration}.pt")

        with torch.no_grad():
            _, y = self.gen(v_list[0], v_list[2], v_list[4])

        self.visualizer(v_list[1:4], y,
                        outdir, iteration, validsize)

    @staticmethod
    def _setting_model_optim(model: nn.Module,
                             config: Dict):
        model.cuda()
        if config["mode"] == "train":
            model.train()
        elif config["mode"] == "eval":
            model.eval()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["lr"],
                                     betas=(config["b1"], config["b2"]))

        return model, optimizer

    @staticmethod
    def _set_requires_grad(net: nn.Module,
                           requires: bool):
        for param in net.parameters():
            param.requires_grad = requires

    def _iter(self, data):
        color, line, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()

        #x = torch.cat([line, mask], dim=1)
        x = line
        z, y = self.gen(x, x)
        z_y = self.enc(y)
        z_t = self.enc(color)
        _, y_t = self.gen(x, x, z_t)

        # discriminate images themselves
        loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
                                                                       y.detach(),
                                                                       color)

        loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
                                                                        y_t.detach(),
                                                                        color)

        self.dis_opt.zero_grad()
        loss.backward()
        self.dis_opt.step()

        # generator & latent encoder update
        loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
                                                                            y)

        loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
                                                                             y_t)

        # latent constrain
        loss += self.loss_config["kl"] * self.lossfunc.kl_loss(z_t)
        loss += self.loss_config["content"] * self.lossfunc.content_loss(y_t, color)

        self.gen_opt.zero_grad()
        self.enc_opt.zero_grad()
        loss.backward(retain_graph=True)
        if self.loss_config["latent"] > 0:
            self._set_requires_grad(self.enc, False)
            loss = self.loss_config["latent"] * self.lossfunc.latent_constrain_loss(z, z_y)
            loss += self.loss_config["ms"] * self.lossfunc.mode_seeking_regularize(y, z)
            loss.backward()
            self._set_requires_grad(self.enc, True)
        self.gen_opt.step()
        self.enc_opt.step()

    def __call__(self):
        iteration = 0
        v_list, v_fix_list = self._valid_prepare(self.dataset,
                                                 self.train_config["validsize"],
                                                 self.train_config["l_dim"])

        for epoch in range(self.train_config["epoch"]):
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.train_config["batchsize"],
                                    shuffle=True,
                                    drop_last=True)
            progress_bar = tqdm(dataloader)

            for index, data in enumerate(progress_bar):
                iteration += 1
                self._iter(data)

                if iteration % self.train_config["snapshot_interval"] == 1:
                    self._eval(iteration,
                               self.train_config["validsize"],
                               v_list, self.outdir)
                    self._eval(iteration,
                               self.train_config["validsize"],
                               v_fix_list, self.outdir_fix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style2Paint")
    parser.add_argument('--outdir', type=Path, default='outdir', help="output directory")
    parser.add_argument('--outdir_fix', type=Path, default='outdir_fix', help="output directory")
    parser.add_argument('--modeldir', type=Path, default='modeldir', help="model output directory")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing color images")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    outdir_fix = args.outdir_fix
    outdir_fix.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config,
                      outdir,
                      outdir_fix,
                      modeldir,
                      args.data_path,
                      args.sketch_path)
    trainer()
