import yaml
import torch
import torch.nn as nn
import argparse
import pprint

from typing import List, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LocalEnhancer, GlobalGenerator, Discriminator, down_sample, Vgg19
from torch.utils.data import DataLoader
from dataset import IllustDataset
from visualize import Visualizer
from utils import session
from loss import Pix2pixHDCalculator


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

        loc_gen = LocalEnhancer(in_ch=in_ch,
                                num_layers=model_config["local_enhancer"]["num_layers"])
        self.loc_gen, self.loc_gen_opt = self._setting_model_optim(loc_gen,
                                                                   model_config["local_enhancer"])

        glo_gen = GlobalGenerator(in_ch=in_ch)
        self.glo_gen, self.glo_gen_opt = self._setting_model_optim(glo_gen,
                                                                   model_config["global_generator"])

        dis = Discriminator(model_config["discriminator"]["in_ch"],
                            model_config["discriminator"]["multi"])
        self.dis, self.dis_opt = self._setting_model_optim(dis,
                                                           model_config["discriminator"])

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.lossfunc = Pix2pixHDCalculator()
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
    def _valid_prepare(dataset, validsize: int, mask: bool) -> List[torch.Tensor]:
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
              v_list: List[torch.Tensor],
              pretrain: bool):

        torch.save(self.loc_gen.state_dict(),
                   f"{self.modeldir}/local_enhancer_{iteration}.pt")
        torch.save(self.glo_gen.state_dict(),
                   f"{self.modeldir}/global_generator_{iteration}.pt")
        torch.save(self.dis.state_dict(),
                   f"{self.modeldir}/discriminator_{iteration}.pt")

        with torch.no_grad():
            if pretrain:
                y = self.loc_gen(v_list[0], pretrain)
            else:
                le = self.loc_gen(v_list[0], pretrain)
                y = self.glo_gen(v_list[0], le)

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _iter(self, data, pretrain: int, epoch: int):
        color, line, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()

        loss = {}

        if self.mask:
            x = torch.cat([line, mask], dim=1)
            t_cat = torch.cat([mask, color], dim=1)
        else:
            x = line
            t_cat = torch.cat([line, color], dim=1)

        if pretrain > epoch:
            y = self.loc_gen(x, pretrain > epoch)

            if self.mask:
                hint = down_sample(mask)
            else:
                hint = down_sample(line)
            t_cat = down_sample(t_cat)
        else:
            le = self.loc_gen(x, pretrain > epoch)
            y = self.glo_gen(x, le)

        y_cat = torch.cat([hint, y], dim=1)

        # discriminate images themselve
        dis_loss = self.loss_config["adv"] * self.lossfunc.dis_loss(self.dis,
                                                                    y_cat.detach(),
                                                                    t_cat)

        self.dis_opt.zero_grad()
        dis_loss.backward()
        self.dis_opt.step()

        adv_gen_loss, fm_loss = self.lossfunc.gen_loss(self.dis,
                                                       y_cat,
                                                       t_cat)

        # gray scale
        fm_loss = self.loss_config["fm"] * fm_loss
        adv_gen_loss = self.loss_config["adv"] * adv_gen_loss
        con_loss = self.loss_config["content"] * self.lossfunc.content_loss(y_cat[:, 3:6, :, :],
                                                                            t_cat[:, 3:6, :, :])
        perceptual_loss = self.loss_config["perceptual"] * self.lossfunc.perceptual_loss(self.vgg,
                                                                                         y_cat[:, 3:6, :, :],
                                                                                         t_cat[:, 3:6, :, :])

        gen_loss = adv_gen_loss + fm_loss + con_loss + perceptual_loss

        self.loc_gen_opt.zero_grad()
        self.glo_gen_opt.zero_grad()
        gen_loss.backward()
        self.loc_gen_opt.step()
        self.glo_gen_opt.step()

        loss["loss_adv_dis"] = dis_loss
        loss["loss_adv_gen"] = adv_gen_loss
        loss["loss_fm"] = fm_loss
        loss["loss_content"] = con_loss
        loss["loss_perceptual"] = perceptual_loss

        return loss

    def __call__(self):
        iteration = 0
        v_list = self._valid_prepare(self.dataset,
                                     self.train_config["validsize"],
                                     self.mask)
        pretrain = self.train_config["pretrain"]

        for epoch in range(self.train_config["epoch"]):
            dataloader = DataLoader(self.dataset,
                                    batch_size=self.train_config["batchsize"],
                                    shuffle=True,
                                    drop_last=True)

            with tqdm(total=len(self.dataset)) as pbar:
                for index, data in enumerate(dataloader):
                    iteration += 1
                    loss_dict = self._iter(data, pretrain, epoch)
                    report_dict = self._build_dict(loss_dict,
                                                   epoch,
                                                   self.train_config["epoch"])

                    pbar.update(self.train_config["batchsize"])
                    pbar.set_postfix(**report_dict)

                    if iteration % self.train_config["snapshot_interval"] == 1:
                        self._eval(iteration,
                                   self.train_config["validsize"],
                                   v_list,
                                   pretrain > epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2pixHD")
    parser.add_argument('--session', type=str, default='pix2pixhd', help="session name")
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
