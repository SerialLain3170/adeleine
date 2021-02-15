import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
import pprint

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Generator, Discriminator, LatentEncoder
from dataset import DanbooruFacesDataset
from visualize import Visualizer
from utils import first_making, noise_generate, session
from loss import DiverseColorizeLossCalculator


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
                                            self.data_config["line_method"],
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

    @staticmethod
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.6f}"

        return report_dict

    def _eval(self,
              iteration: int,
              validsize: int,
              v_list: List[torch.Tensor],
              outdir: Path):

        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")
        torch.save(self.enc.state_dict(),
                   f"{self.modeldir}/latent_encoder_{iteration}.pt")

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

        loss = {}

        #x = torch.cat([line, mask], dim=1)
        x = line
        z, y = self.gen(x, x)
        z_y = self.enc(y)
        z_t = self.enc(color)
        _, y_t = self.gen(x, x, z_t)

        # discriminator process
        dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
                                                                                y.detach(),
                                                                                color)

        dis_loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
                                                                                 y_t.detach(),
                                                                                 color)

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        # generator process
        gen_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
                                                                                y)

        gen_adv_loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
                                                                                 y_t)

        kl_loss = self.loss_config["kl"] * self.lossfunc.kl_loss(z_t)
        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y_t, color)

        gen_loss = gen_adv_loss + kl_loss + con_loss

        self.gen_opt.zero_grad()
        self.enc_opt.zero_grad()
        gen_loss.backward(retain_graph=True)

        # latent encoder process
        if self.loss_config["latent"] > 0:
            self._set_requires_grad(self.enc, False)
            constraint_loss = self.loss_config["latent"] * self.lossfunc.latent_constrain_loss(z, z_y)
            ms_loss = self.loss_config["ms"] * self.lossfunc.mode_seeking_regularize(y, z)
            enc_loss = constraint_loss + ms_loss
            enc_loss.backward()
            self._set_requires_grad(self.enc, True)
        self.gen_opt.step()
        self.enc_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = gen_adv_loss.item()
        loss["loss_kl"] = kl_loss.item()
        loss["loss_content"] = con_loss.item()
        loss["loss_constraint"] = constraint_loss.item()
        loss["loss_ms"] = ms_loss.item()

        return loss

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

            with tqdm(total=len(self.dataset)) as pbar:
                for index, data in enumerate(dataloader):
                    iteration += 1
                    loss_dict = self._iter(data)
                    report_dict = self._build_dict(loss_dict,
                                                   epoch,
                                                   self.train_config["epoch"])

                    pbar.update(self.train_config["batchsize"])
                    pbar.set_postfix(**report_dict)

                    if iteration % self.train_config["snapshot_interval"] == 1:
                        self._eval(iteration,
                                   self.train_config["validsize"],
                                   v_list, self.outdir)
                        self._eval(iteration,
                                   self.train_config["validsize"],
                                   v_fix_list, self.outdir_fix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BicycleGAN")
    parser.add_argument('--session', type=str, default='bicyclegan', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing color images")
    args = parser.parse_args()

    outdir, outdir_fix, modeldir = session(args.session)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    trainer = Trainer(config,
                      outdir,
                      outdir_fix,
                      modeldir,
                      args.data_path,
                      args.sketch_path)
    trainer()
