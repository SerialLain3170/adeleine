import yaml
import torch
import torch.nn as nn
import argparse
import pprint

from collections import OrderedDict
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from loss import Pix2pixCalculator
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from dataset import IllustDataset
from visualize import Visualizer
from utils import session


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 sketch_path,
                 ):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir
        self.mask = self.train_config["mask"]

        self.dataset = IllustDataset(data_path,
                                     sketch_path,
                                     self.data_config["line_method"],
                                     self.data_config["extension"],
                                     self.data_config["train_size"],
                                     self.data_config["valid_size"],
                                     self.data_config["color_space"],
                                     self.data_config["line_space"])
        print(self.dataset)

        if self.mask:
            in_ch = 6
        else:
            in_ch = 3

        gen = Generator(in_ch=in_ch)
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        dis = Discriminator()
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.lossfunc = Pix2pixCalculator()
        self.visualizer = Visualizer(self.data_config["color_space"])

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
    def _valid_prepare(dataset,
                       validsize: int,
                       mask: bool) -> List[torch.Tensor]:

        c_val, l_val, m_val = dataset.valid(validsize)

        if mask:
            x_val = torch.cat([l_val, m_val], dim=1)
        else:
            x_val = l_val

        return [x_val, l_val, m_val, c_val]

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
              v_list: List[torch.Tensor]):
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

        with torch.no_grad():
            y = self.gen(v_list[0])

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _iter(self, data):
        color, line, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()

        loss = {}

        if self.mask:
            y = self.gen(torch.cat([line, mask], dim=1))
        else:
            y = self.gen(line)

        # discriminator process
        adv_dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
                                                                            y.detach(),
                                                                            color)
        gp_loss = self.loss_config["gp"] * self.lossfunc.gradient_penalty(self.dis,
                                                                        color,
                                                                        center=self.loss_config["center"])

        dis_loss = adv_dis_loss + gp_loss

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        # generator process
        adv_gen_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
                                                                            y)

        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y,
                                                                         color)

        gen_loss = adv_gen_loss + con_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = adv_dis_loss.item()
        loss["loss_gp"] = gp_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_content"] = con_loss.item()

        return loss

    def __call__(self):
        iteration = 0
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"],
                                     self.mask)

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
                                   v_list,
                                   )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2pix")
    parser.add_argument('--session', type=str, default='pix2pix', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
    args = parser.parse_args()

    outdir, modeldir = session(args.session)

    with open("param.yaml", "r") as f:
        config = yaml.safe_load(f)
        pprint.pprint(config)

    trainer = Trainer(config,
                      outdir,
                      modeldir,
                      args.data_path,
                      args.sketch_path,
                      )
    trainer()
