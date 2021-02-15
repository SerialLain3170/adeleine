import torch
import torch.nn as nn

from typing import List

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class Pix2pixHDCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    @staticmethod
    def feature_matching_loss(fake_feats: List[torch.Tensor],
                              real_feats: List[torch.Tensor]) -> torch.Tensor:
        sum_loss = 0

        d_weight = float(1.0 / 3.0)
        feat_weight = float(4.0 / 7.0)

        for y, t in zip(fake_feats, real_feats):
            sum_loss += d_weight * feat_weight * torch.mean(torch.abs(y-t))

        return sum_loss

    @staticmethod
    def adversarial_disloss(y_list: List[torch.Tensor],
                            t_list: List[torch.Tensor]) -> torch.Tensor:
        sum_loss = 0

        for y, t in zip(y_list, t_list):
            loss = torch.mean(softplus(-t)) + torch.mean(softplus(y))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_genloss(y_list: List[torch.Tensor]) -> torch.Tensor:
        sum_loss = 0

        for y in y_list:
            loss = torch.mean(softplus(-y))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_hingedis(y_list: List[torch.Tensor],
                             t_list: List[torch.Tensor]) -> torch.Tensor:
        sum_loss = 0

        for y, t in zip(y_list, t_list):
            sum_loss += nn.ReLU()(1.0 + y).mean()
            sum_loss += nn.ReLU()(1.0 - t).mean()

        return sum_loss

    @staticmethod
    def adversarial_hingegen(y_list: List[torch.Tensor]) -> torch.Tensor:
        sum_loss = 0

        for y in y_list:
            sum_loss += -y.mean()

        return sum_loss

    def dis_loss(self, discriminator: nn.Module,
                 y: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:

        _, y_outputs = discriminator(y)
        _, t_outputs = discriminator(t)

        return self.adversarial_hingedis(y_outputs, t_outputs)

    def gen_loss(self, discriminator: nn.Module,
                 y: torch.Tensor,
                 t: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        y_feats, y_outputs = discriminator(y)
        t_feats, _ = discriminator(t)

        adv_loss = self.adversarial_hingegen(y_outputs)
        fm_loss = self.feature_matching_loss(y_feats, t_feats)

        return adv_loss, fm_loss

    @staticmethod
    def perceptual_loss(vgg: nn.Module,
                        y: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        y_feat = vgg(y)
        t_feat = vgg(t)

        for y, t in zip(y_feat, t_feat):
            _, c, h, w = y.size()
            sum_loss += torch.mean(torch.abs(y-t)) / (c * h * w)

        return sum_loss
