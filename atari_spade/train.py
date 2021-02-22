import yaml
import torch
import torch.nn as nn
import argparse
import pprint

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Generator, Discriminator, Vgg19
from dataset import BuildDataset, noise_generate
from visualize import Visualizer
from loss import SPADELossCalculator
from utils import session


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 outdir_fix,
                 modeldir,
                 data_path,
                 sketch_path,
                 ):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.outdir_fix = outdir_fix
        self.modeldir = modeldir

        self.dataset = BuildDataset(data_path,
                                    sketch_path,
                                    self.data_config["line_method"],
                                    self.data_config["extension"],
                                    self.data_config["train_size"],
                                    self.data_config["valid_size"],
                                    self.data_config["color_space"],
                                    self.data_config["line_space"]
                                    )
        print(self.dataset)

        gen = Generator(model_config["generator"]["in_ch"],
                        self.train_config["latent_dim"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])

        dis = Discriminator(multi_patterns=model_config["discriminator"]["multi"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = SPADELossCalculator()
        self.visualizer = Visualizer()
        self.l_dim = self.train_config["latent_dim"]

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
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.6f}"

        return report_dict

    @staticmethod
    def _valid_prepare(dataset,
                       validsize: int,
                       l_dim: int) -> List[torch.Tensor]:
        c_val, l_val, m_val, c_fix, l_fix, m_fix = dataset.valid(validsize)
        x_val = torch.cat([l_val, m_val], dim=1)
        x_fix = torch.cat([l_fix, m_fix], dim=1)
        z_fix = noise_generate(validsize, l_dim)

        return [x_val, l_val, m_val, c_val], [x_fix, l_fix, m_fix, c_fix], z_fix

    def _eval(self,
              l_dim: int,
              z_fix: torch.Tensor,
              iteration: int,
              validsize: int,
              v_list: List[torch.Tensor],
              fix_list: List[torch.Tensor]):

        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

        with torch.no_grad():
            y_fix = self.gen(z_fix, fix_list[0])
            z = noise_generate(validsize, l_dim)
            y = self.gen(z, v_list[0])

        self.visualizer(fix_list[1:], y_fix,
                        self.outdir_fix, iteration, validsize)

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _iter(self, data):
        color, line, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()

        loss = {}

        x = torch.cat([line, mask], dim=1)
        batchsize = x.size(0)
        z = noise_generate(batchsize, self.l_dim)

        # Discriminator update
        y = self.gen(z, x)
        dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_disloss(self.dis,
                                                                              y.detach(),
                                                                              color)

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        # Generator update
        y = self.gen(z, x)
        gen_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_genloss(self.dis, y)
        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, color)
        pef_loss = self.loss_config["pe"] * self.lossfunc.positive_enforcing_loss(y)

        gen_loss = gen_adv_loss + content_loss + pef_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = gen_adv_loss.item()
        loss["loss_content"] = content_loss.item()
        loss["loss_pef"] = pef_loss.item()

        return loss

    def __call__(self):
        iteration = 0
        v_list, fix_list, z_fix = self._valid_prepare(self.dataset,
                                                      self.train_config["validsize"],
                                                      self.l_dim)

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
                        self._eval(self.l_dim,
                                   z_fix,
                                   iteration,
                                   self.train_config["validsize"],
                                   v_list,
                                   fix_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPADE colorization")
    parser.add_argument('--session', type=str, default='spade', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
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
