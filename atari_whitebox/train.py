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
from utils import GuidedFilter, random_color_shift, session
from loss import WhiteBoxLossCalculator


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
                        num_layers=model_config["generator"]["num_layers"],
                        attn_type=model_config["generator"]["attn_type"],
                        guide=model_config["generator"]["guide"])
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["generator"])
        self.guide = model_config["generator"]["guide"]

        i_dis = Discriminator(model_config["image_dis"]["in_ch"],
                              model_config["image_dis"]["multi"])
        self.i_dis, self.i_dis_opt = self._setting_model_optim(i_dis,
                                                               model_config["image_dis"])

        s_dis = Discriminator(model_config["surface_dis"]["in_ch"],
                              model_config["surface_dis"]["multi"])
        self.s_dis, self.s_dis_opt = self._setting_model_optim(s_dis,
                                                               model_config["surface_dis"])

        t_dis = Discriminator(model_config["texture_dis"]["in_ch"],
                              model_config["texture_dis"]["multi"])
        self.t_dis, self.t_dis_opt = self._setting_model_optim(t_dis,
                                                               model_config["texture_dis"])

        self.guided_filter = GuidedFilter(r=5, eps=2e-1)
        self.guided_filter.cuda()

        self.out_guided_filter = GuidedFilter(r=1, eps=1e-2)
        self.out_guided_filter.cuda()

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = WhiteBoxLossCalculator()
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
    def _valid_prepare(dataset, validsize: int) -> List[torch.Tensor]:
        c_val, l_val, m_val = dataset.valid(validsize)
        x_val = torch.cat([l_val, m_val], dim=1)

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

    def _eval(self, iteration, validsize, v_list):
        torch.save(self.gen.state_dict(),
                   f"{self.modeldir}/generator_{iteration}.pt")
        torch.save(self.i_dis.state_dict(),
                   f"{self.modeldir}/image_discriminator_{iteration}.pt")
        torch.save(self.t_dis.state_dict(),
                   f"{self.modeldir}/texture_discriminator_{iteration}.pt")
        torch.save(self.s_dis.state_dict(),
                   f"{self.modeldir}/surface_discriminator_{iteration}.pt")

        with torch.no_grad():
            if self.guide:
                y, _, _ = self.gen(v_list[0], v_list[2])
            else:
                y = self.gen(v_list[0], v_list[2])

            if self.train_config["out_guide"]:
                y = self.out_guided_filter(v_list[1], y)

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _discriminator_zero_grad(self):
        if self.loss_config["image"] > 0:
            self.i_dis_opt.zero_grad()
        if self.loss_config["surface"] > 0:
            self.s_dis_opt.zero_grad()
        if self.loss_config["texture"] > 0:
            self.t_dis_opt.zero_grad()

    def _discriminator_step(self):
        if self.loss_config["image"] > 0:
            self.i_dis_opt.step()
        if self.loss_config["surface"] > 0:
            self.s_dis_opt.step()
        if self.loss_config["texture"] > 0:
            self.t_dis_opt.step()

    def _iter(self, data):
        color, line, mask, ss = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()
        ss = ss.cuda()

        loss = {}

        x = torch.cat([line, mask], dim=1)

        if self.guide:
            y, g1, g2 = self.gen(x, mask)
        else:
            y = self.gen(x, mask)

        if self.train_config["out_guide"]:
            y = self.out_guided_filter(line, y)

        # discriminate images themselves
        dis_adv_img_loss = self.loss_config["image"] * self.lossfunc.adversarial_hingedis(self.i_dis,
                                                                                          y.detach(),
                                                                                          color)

        # guided filter
        y_gf = self.guided_filter(y, y)
        t_gf = self.guided_filter(color, color)
        dis_adv_sur_loss = self.loss_config["surface"] * self.lossfunc.adversarial_hingedis(self.s_dis,
                                                                                            y_gf.detach(),
                                                                                            t_gf)

        # gray scale
        y_cs, t_cs = random_color_shift(y, color)
        dis_adv_tex_loss = self.loss_config["texture"] * self.lossfunc.adversarial_hingedis(self.t_dis,
                                                                                            y_cs.detach(),
                                                                                            t_cs)

        dis_loss = dis_adv_img_loss + dis_adv_sur_loss + dis_adv_tex_loss

        self._discriminator_zero_grad()
        dis_loss.backward()
        self._discriminator_step()

        gen_adv_img_loss = self.loss_config["image"] * self.lossfunc.adversarial_hingegen(self.i_dis,
                                                                                          y)

        # guided filter
        gen_adv_sur_loss = 0.1 * self.loss_config["surface"] * self.lossfunc.adversarial_hingegen(self.s_dis,
                                                                                                  y_gf)

        # gray scale
        gen_adv_tex_loss = self.loss_config["texture"] * self.lossfunc.adversarial_hingegen(self.t_dis,
                                                                                            y_cs)

        structure_loss = self.loss_config["structure"] * self.lossfunc.perceptual_loss(self.vgg, y, ss)
        tv_loss = self.loss_config["tv"] * self.lossfunc.total_variation_loss(y)
        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, color)

        if self.guide:
            con_loss += self.loss_config["content"] * self.lossfunc.content_loss(g1, color)
            con_loss += self.loss_config["content"] * self.lossfunc.content_loss(g2, color)

        gen_loss = gen_adv_img_loss + gen_adv_sur_loss + gen_adv_tex_loss + structure_loss + tv_loss + con_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis_img"] = dis_adv_img_loss.item()
        loss["loss_adv_dis_sur"] = dis_adv_sur_loss.item()
        loss["loss_adv_dis_tex"] = dis_adv_tex_loss.item()
        loss["loss_adv_gen_img"] = gen_adv_img_loss.item()
        loss["loss_adv_gen_sur"] = gen_adv_sur_loss.item()
        loss["loss_adv_gen_tex"] = gen_adv_tex_loss.item()
        loss["loss_structure"] = structure_loss.item()
        loss["loss_tv"] = tv_loss.item()
        loss["loss_content"] = con_loss.item()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whitebox Colorization")
    parser.add_argument('--session', type=str, default='whitebox', help="session name")
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
