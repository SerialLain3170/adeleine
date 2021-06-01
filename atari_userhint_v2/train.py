import copy
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
from dataset import IllustDataset
from visualize import Visualizer
from loss import LossCalculator
from utils import session, GuidedFilter


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 sketch_path,
                 ss_path):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir

        self.dataset = IllustDataset(data_path,
                                     sketch_path,
                                     ss_path,
                                     self.data_config["line_method"],
                                     self.data_config["extension"],
                                     self.data_config["train_size"],
                                     self.data_config["valid_size"],
                                     self.data_config["color_space"],
                                     self.data_config["line_space"])
        print(self.dataset)

        gen = Generator(model_config["generator"]["in_ch"],
                        base=model_config["generator"]["base"],
                        num_layers=model_config["generator"]["num_layers"],
                        up_layers=model_config["generator"]["up_layers"],
                        guide=model_config["generator"]["guide"],
                        resnext=model_config["generator"]["resnext"],
                        encoder_type=model_config["generator"]["encoder_type"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])
        self.guide = model_config["generator"]["guide"]

        dis = Discriminator(model_config["discriminator"]["in_ch"],
                            model_config["discriminator"]["multi"],
                            base=model_config["discriminator"]["base"],
                            sn=model_config["discriminator"]["sn"],
                            resnext=model_config["discriminator"]["resnext"],
                            patch=model_config["discriminator"]["patch"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.vgg = Vgg19(requires_grad=False, layer="four")
        self.vgg.cuda()
        self.vgg.eval()

        self.out_filter = GuidedFilter(r=1, eps=1e-2)
        self.out_filter.cuda()

        self.lossfunc = LossCalculator()
        self.visualizer = Visualizer(self.data_config["color_space"])

        self.scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(self.gen_opt, self.train_config["gamma"])
        self.scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(self.dis_opt, self.train_config["gamma"])

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
    def _valid_prepare(dataset, validsize: int) -> List[torch.Tensor]:
        c_val, l_i_val, m_val, l_m_val = dataset.valid(validsize)
        x_val = torch.cat([l_i_val, m_val], dim=1)

        return [x_val, l_i_val, m_val, c_val, l_m_val]

    @staticmethod
    def _build_dict(loss_dict: Dict[str, float],
                    epoch: int,
                    num_epochs: int) -> Dict[str, str]:

        report_dict = {}
        report_dict["epoch"] = f"{epoch}/{num_epochs}"
        for k, v in loss_dict.items():
            report_dict[k] = f"{v:.4f}"

        return report_dict

    def _loss_weight_scheduler(self, iteration: int):
        if iteration > 50000:
            self.loss_config["adv"] = 0.1
            self.loss_config["content"] = 1.0
            self.loss_config["pef"] = 0.0
        else:
            self.loss_config["adv"] = 0.1
            self.loss_config["content"] = 100.0
            self.loss_config["pef"] = 0.01

    def _eval(self,
              iteration: int,
              validsize: int,
              v_list: List[torch.Tensor]):

        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

        with torch.no_grad():
            mid = copy.copy(v_list[4])
            if self.guide:
                y, _, _ = self.gen(v_list[0], mid)
            else:
                y = self.gen(v_list[0], mid)

            y = self.out_filter(v_list[1], y)

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _iter(self, data):
        color, line, mask, line_m = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()
        line_m = line_m.cuda()

        loss = {}

        x = torch.cat([line, mask], dim=1)
        mid = line_m

        if self.guide:
            y, g1, g2 = self.gen(x, mid)
        else:
            y = self.gen(x, mid)

        y = self.out_filter(line, y)

        if self.loss_config["adv"] > 0:
            # discriminator update
            dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_disloss(self.dis,
                                                                                    y.detach(),
                                                                                    color)

            self.dis_opt.zero_grad()
            dis_loss.backward()
            self.dis_opt.step()
        else:
            dis_loss = torch.zeros(1).cuda()

        if self.loss_config["gp"] > 0.0:
            color.requires_grad = True
            gp_loss = self.loss_config["gp"] * self.lossfunc.gradient_penalty(self.dis,
                                                                                   color
                                                                                   )

            self.dis_opt.zero_grad()
            gp_loss.backward()
            self.dis_opt.step()

            color.requires_grad = False
        else:
            gp_loss = torch.zeros(1).cuda()

        if self.guide:
            y, g1, g2 = self.gen(x, mid)
        else:
            y = self.gen(x, mid)

        y = self.out_filter(line, y)

        # generator update
        if self.loss_config["adv"] > 0:
            adv_loss, fm_loss = self.lossfunc.adversarial_genloss(self.dis,
                                                                  y,
                                                                  color)
            adv_gen_loss = self.loss_config["adv"] * adv_loss
            fm_loss = self.loss_config["fm"] * fm_loss
        else:
            adv_gen_loss = torch.zeros(1).cuda()
            fm_loss = torch.zeros(1).cuda()

        tv_loss = self.loss_config["tv"] * self.lossfunc.total_variation_loss(y)
        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, color)
        pef_loss = self.loss_config["pef"] * self.lossfunc.positive_enforcing_loss(y)
        perceptual_loss = self.loss_config["perceptual"] * self.lossfunc.perceptual_loss(self.vgg, y, color)

        if self.guide:
            content_loss += self.loss_config["content"] * self.lossfunc.content_loss(g1, color)
            content_loss += self.loss_config["content"] * self.lossfunc.content_loss(g2, color)

        gen_loss = adv_gen_loss + fm_loss + tv_loss + content_loss + pef_loss + perceptual_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_fm"] = fm_loss.item()
        loss["loss_tv"] = tv_loss.item()
        loss["loss_content"] = content_loss.item()
        loss["loss_pef"] = pef_loss.item()
        loss["loss_perceptual"] = perceptual_loss.item()
        loss["loss_gp"] = gp_loss.item()

        return loss

    def __call__(self):
        iteration = 0
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"])

        for epoch in range(self.train_config["epoch"]):
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.train_config["batchsize"],
                                    shuffle=True,
                                    drop_last=True)

            with tqdm(total=len(self.dataset)) as pbar:
                for index, data in enumerate(dataloader):
                    self._loss_weight_scheduler(iteration)
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
                                   v_list)

            self.scheduler_dis.step()
            self.scheduler_gen.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="userguide_point")
    parser.add_argument('--session', type=str, default='userhint', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
    parser.add_argument('--ss_path', type=Path, help="path containing selective search images")
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
                      args.ss_path)
    trainer()
