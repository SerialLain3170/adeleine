import yaml
import torch
import argparse
import torch.nn as nn
import numpy as np
import pprint

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import ColorTransformNetwork, Discriminator, Vgg19
from model import TemporalConstraintNetwork, TemporalDiscriminator
from dataset import IllustDataset
from visualize import Visualizer
from loss import VideoColorizeLossCalculator
from utils import session, sum_totensor, movie_prepare


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 sketch_path,
                 dist_path,
                 pretrained_path):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir

        self.dataset = IllustDataset(data_path,
                                     sketch_path,
                                     dist_path,
                                     self.data_config["anime_dir"],
                                     self.data_config["extension"],
                                     self.data_config["train_size"],
                                     self.data_config["valid_size"],
                                     self.data_config["scale"],
                                     self.data_config["frame_range"])
        print(self.dataset)

        self.ctn = ColorTransformNetwork(layers=model_config["CTN"]["num_layers"])
        self.ctn.cuda()
        self.ctn.eval()

        weight = torch.load(pretrained_path)
        self.ctn.load_state_dict(weight)

        gen = TemporalConstraintNetwork()
        self.gen, self.gen_opt = self._setting_model_optim(gen,
                                                           model_config["TCN"])

        dis = Discriminator()
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        t_dis = TemporalDiscriminator()
        self.t_dis, self.t_dis_opt = self._setting_model_optim(t_dis,
                                                               model_config["temporal_discriminator"])

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = VideoColorizeLossCalculator()
        self.visualizer = Visualizer()

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
                       validsize: int) -> List[torch.Tensor]:

        l_x_v, l_y0_v, l_y1_v, c_x_v, c_y0_v, c_y1_v, d_x_v, d_y0_v, d_y1_v = dataset.valid(validsize)

        return [l_x_v, l_y0_v, l_y1_v, c_x_v, c_y0_v, c_y1_v, d_x_v, d_y0_v, d_y1_v]

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
                   f"{self.modeldir}/tcn_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")
        torch.save(self.t_dis.state_dict(),
                   f"{self.modeldir}/temporal_discriminator_{iteration}.pt")

        with torch.no_grad():
            ypre, ysim, ymid = self.ctn(v_list[0], v_list[1], v_list[2],
                                        v_list[6], v_list[7], v_list[8],
                                        v_list[4], v_list[5])
            y = movie_prepare(v_list[0], v_list[1], v_list[2], ypre, v_list[4], v_list[5])
            y = self.gen(y)

        y = y[:, 3:6, 1, :, :]

        self.visualizer(v_list, ymid, ypre, y,
                        self.outdir, iteration, validsize)

    def _iter(self, data):
        l_x, l_y0, l_y1, c_x, c_y0, c_y1, d_x, d_y0, d_y1 = data
        l_x, l_y0, l_y1, c_x, c_y0, c_y1, d_x, d_y0, d_y1 = sum_totensor(l_x, l_y0, l_y1, c_x, c_y0, c_y1, d_x, d_y0, d_y1)

        loss = {}

        with torch.no_grad():
            y, ysim, ymid = self.ctn(l_x, l_y0, l_y1,
                                     d_x, d_y0, d_y1,
                                     c_y0, c_y1)

        y = movie_prepare(l_x, l_y0, l_y1, y, c_y0, c_y1)
        t = movie_prepare(l_x, l_y0, l_y1, c_x, c_y0, c_y1)
        y = self.gen(y)

        dis_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_dis_loss(self.dis,
                                                                                    y[:, 3:6, 1, :, :],
                                                                                    c_x)

        t_dis_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_dis_loss(self.t_dis,
                                                                                      y,
                                                                                      t)

        dis_loss = dis_adv_loss + t_dis_adv_loss

        self.dis_opt.zero_grad()
        self.t_dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()
        self.t_dis_opt.step()

        t_gen_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_gen_loss(self.t_dis,
                                                                                      y)

        y = y[:, 3:6, 1, :, :]

        gen_adv_loss = self.loss_config["adv"] * self.lossfunc.adversarial_gen_loss(self.dis,
                                                                                    y)
        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, c_x)
        tv_loss = self.loss_config["tv"] * self.lossfunc.total_variation_loss(y)
        lc_loss = self.loss_config["constraint"] * self.lossfunc.latent_constraint_loss(ysim, ymid, c_x)
        perceptual_loss = self.loss_config["perceptual"] * self.lossfunc.perceptual_loss(self.vgg, y, c_x)

        gen_loss = gen_adv_loss + content_loss + tv_loss + lc_loss + perceptual_loss + t_gen_adv_loss

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        loss["loss_adv_dis"] = dis_adv_loss
        loss["loss_adv_gen"] = gen_adv_loss
        loss["loss_adv_dis_temp"] = t_dis_adv_loss
        loss["loss_adv_gen_temp"] = t_gen_adv_loss
        loss["loss_content"] = content_loss
        loss["loss_tv"] = tv_loss
        loss["loss_constraint"] = lc_loss
        loss["loss_perceptual"] = perceptual_loss

        return loss

    def __call__(self):
        iteration = 0
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"],
                                     )

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
    parser = argparse.ArgumentParser(description="TCN")
    parser.add_argument('--session', type=str, default='tcn', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
    parser.add_argument('--dist_path', type=Path, help="path containing distance field images")
    parser.add_argument('--pre_path', type=Path, help="pretrained model of ctn")
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
                      args.dist_path,
                      args.pre_path)
    trainer()
