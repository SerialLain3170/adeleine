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
from bicyclegan import Generator as BicycleGAN
from bicyclegan import LatentEncoder
from dataset import IllustDataset, DanbooruFacesDataset
from visualize import Visualizer
from utils import session, GuidedFilter
from loss import DecomposeLossCalculator


class Trainer:
    def __init__(self,
                 config,
                 outdir,
                 modeldir,
                 data_path,
                 sketch_path,
                 flat_path,
                 pretrain_path=None):

        self.train_config = config["train"]
        self.data_config = config["dataset"]
        model_config = config["model"]
        self.loss_config = config["loss"]

        self.outdir = outdir
        self.modeldir = modeldir

        self.train_type = self.train_config["train_type"]

        if self.train_type == "multi":
            self.dataset = DanbooruFacesDataset(data_path,
                                                sketch_path,
                                                self.data_config["line_method"],
                                                self.data_config["extension"],
                                                self.data_config["train_size"],
                                                self.data_config["valid_size"],
                                                self.data_config["color_space"],
                                                self.data_config["line_space"])

        else:
            self.dataset = IllustDataset(data_path,
                                         sketch_path,
                                         flat_path,
                                         self.data_config["line_method"],
                                         self.data_config["extension"],
                                         self.data_config["train_size"],
                                         self.data_config["valid_size"],
                                         self.data_config["color_space"],
                                         self.data_config["line_space"])
        print(self.dataset)

        flat_gen = Generator(model_config["flat_generator"]["in_ch"],
                             num_layers=model_config["flat_generator"]["num_layers"],
                             attn_type=model_config["flat_generator"]["attn_type"],
                             )
        self.flat_gen, self.flat_gen_opt = self._setting_model_optim(flat_gen,
                                                                     model_config["flat_generator"])

        if self.train_type == "multi":
            weight = torch.load(pretrain_path)
            self.flat_gen.load_state_dict(weight)

        f_dis = Discriminator(model_config["flat_dis"]["in_ch"],
                              model_config["flat_dis"]["multi"])
        self.f_dis, self.f_dis_opt = self._setting_model_optim(f_dis,
                                                               model_config["flat_dis"])

        if self.train_type == "multi":
            bicycle_gen = BicycleGAN(model_config["bicycle_gan"]["in_ch"],
                                     latent_dim=model_config["bicycle_gan"]["l_dim"],
                                     num_layers=model_config["bicycle_gan"]["num_layers"])
            self.b_gen, self.b_gen_opt = self._setting_model_optim(bicycle_gen,
                                                                   model_config["bicycle_gan"])

            latent_enc = LatentEncoder(model_config["encoder"]["in_ch"],
                                       latent_dim=model_config["encoder"]["l_dim"])
            self.l_enc, self.l_enc_opt = self._setting_model_optim(latent_enc,
                                                                   model_config["encoder"])

            b_dis = Discriminator(model_config["bicycle_dis"]["in_ch"],
                                  model_config["bicycle_dis"]["multi"])
            self.b_dis, self.b_dis_opt = self._setting_model_optim(b_dis,
                                                                   model_config["bicycle_dis"])

        self.vgg = Vgg19(requires_grad=False)
        self.vgg.cuda()
        self.vgg.eval()

        self.out_filter = GuidedFilter(r=1, eps=1e-2)
        self.out_filter.cuda()

        self.lossfunc = DecomposeLossCalculator()
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
        c_val, l_val, f_val, m_val = dataset.valid(validsize)
        x_val = torch.cat([l_val, m_val], dim=1)

        return [x_val, l_val, m_val, f_val, c_val]

    @staticmethod
    def _set_requires_grad(net: nn.Module,
                           requires: bool):
        for param in net.parameters():
            param.requires_grad = requires

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
        if self.train_type == "flat":
            torch.save(self.flat_gen.state_dict(),
                       f"{self.modeldir}/flat_{iteration}.pt")
        elif self.train_type == "multi":
            torch.save(self.b_gen.state_dict(),
                       f"{self.modeldir}/bicyclegan_{iteration}.pt")

        with torch.no_grad():
            if self.train_type == "flat":
                y = self.flat_gen(v_list[0], v_list[2])
                y = self.out_filter(v_list[1], y)
            elif self.train_type == "multi":
                flat = self.flat_gen(v_list[0], v_list[2])
                flat = self.out_filter(v_list[1], flat)
                _, y = self.b_gen(flat, v_list[2])
                v_list[3] = flat

        self.visualizer(v_list[1:], y,
                        self.outdir, iteration, validsize)

    def _flatter(self, data):
        color, line, flat, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()
        flat = flat.cuda()

        loss = {}

        x = torch.cat([line, mask], dim=1)

        y = self.flat_gen(x, mask)
        y = self.out_filter(line, y)

        # Discriminator update
        dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.f_dis,
                                                                                y.detach(),
                                                                                flat)

        self.f_dis_opt.zero_grad()
        dis_loss.backward()
        self.f_dis_opt.step()

        # Generator update
        adv_gen_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.f_dis,
                                                                                    y)
        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y, flat)
        perceptual_loss = self.loss_config["perceptual"] * self.lossfunc.perceptual_loss(self.vgg, y, flat)

        gen_loss = adv_gen_loss + content_loss + perceptual_loss

        self.flat_gen_opt.zero_grad()
        gen_loss.backward()
        self.flat_gen_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_content"] = content_loss.item()
        loss["loss_perceptual"] = perceptual_loss.item()

        return loss

    def _multi(self, data):
        color, line, _, mask = data
        color = color.cuda()
        line = line.cuda()
        mask = mask.cuda()

        loss = {}

        x = torch.cat([line, mask], dim=1)

        with torch.no_grad():
            flat = self.flat_gen(x, mask)
            flat = self.out_filter(line, flat)

        z, y = self.b_gen(flat, mask)
        z_y = self.l_enc(torch.cat([y, flat], dim=1))
        z_t = self.l_enc(torch.cat([color, flat], dim=1))
        _, y_t = self.b_gen(flat, mask, z_t)

        # discriminate images themselves
        dis_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.b_dis,
                                                                            y.detach(),
                                                                            color)

        dis_loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.b_dis,
                                                                             y_t.detach(),
                                                                             color)

        self.b_dis_opt.zero_grad()
        dis_loss.backward()
        self.b_dis_opt.step()

        # generator & latent encoder update
        adv_gen_loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.b_dis,
                                                                            y)

        adv_gen_loss += self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.b_dis,
                                                                             y_t)

        # latent constrain
        kl_loss = self.loss_config["kl"] * self.lossfunc.kl_loss(z_t)
        content_loss = self.loss_config["content"] * self.lossfunc.content_loss(y_t, color)

        self.b_gen_opt.zero_grad()
        self.l_enc_opt.zero_grad()
        gen_loss = adv_gen_loss + kl_loss + content_loss
        gen_loss.backward(retain_graph=True)
        if self.loss_config["latent"] > 0:
            self._set_requires_grad(self.l_enc, False)
            latent_loss = self.loss_config["latent"] * self.lossfunc.latent_constrain_loss(z, z_y)
            ms_loss = self.loss_config["ms"] * self.lossfunc.mode_seeking_regularize(y, z)
            enc_loss = latent_loss + ms_loss
            enc_loss.backward()
            self._set_requires_grad(self.l_enc, True)
        self.b_gen_opt.step()
        self.l_enc_opt.step()

        loss["loss_adv_dis"] = dis_loss.item()
        loss["loss_adv_gen"] = adv_gen_loss.item()
        loss["loss_kl"] = kl_loss.item()
        loss["loss_content"] = content_loss.item()
        loss["loss_latent"] = latent_loss.item()
        loss["loss_ms"] = ms_loss.item()

        return loss

    def _iter(self, data):
        if self.train_type == "flat":
            loss = self._flatter(data)
        elif self.train_type == "multi":
            loss = self._multi(data)

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
    parser = argparse.ArgumentParser(description="Decomposer")
    parser.add_argument('--session', type=str, default='decompose', help="session name")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
    parser.add_argument('--flat_path', type=Path, help="path containing selective search images")
    parser.add_argument('--pretrain_path', type=Path, help="path containing pretrain model")
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
                      args.flat_path,
                      args.pretrain_path)
    trainer()
