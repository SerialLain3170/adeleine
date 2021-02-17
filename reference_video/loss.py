import torch
import torch.nn as nn

maeloss = nn.L1Loss()
softplus = nn.Softplus()
downsample = nn.AvgPool2d(8, 4, 2)


class VideoColorizeLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return maeloss(y, t)

    @staticmethod
    def adversarial_dis_loss(discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        y_dis = discriminator(y.detach())
        t_dis = discriminator(t)

        loss = torch.mean(softplus(-t_dis))
        loss += torch.mean(softplus(y_dis))

        return loss

    @staticmethod
    def adversarial_gen_loss(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:
        y_dis = discriminator(y)

        loss = torch.mean(softplus(-y_dis))

        return loss

    @staticmethod
    def total_variation_loss(y: torch.Tensor) -> torch.Tensor:
        _, ch, h, w = y.size()
        loss = maeloss(y[:, :, :-1, :], y[:, :, 1:, :])
        loss += maeloss(y[:, :, :, :-1], y[:, :, :, 1:])

        return loss / (ch * h * w)

    @staticmethod
    def latent_constraint_loss(ysim: torch.Tensor,
                               ymid: torch.Tensor,
                               t: torch.Tensor) -> torch.Tensor:
        t = downsample(t)
        loss = maeloss(ysim, t)
        loss += maeloss(ymid, t)

        return loss

    @staticmethod
    def perceptual_loss(vgg: nn.Module,
                        y: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        y_feat = vgg(y)
        t_feat = vgg(t)

        sum_loss = 0

        for y, t in zip(y_feat, t_feat):
            _, ch, h, w = y.size()
            sum_loss += maeloss(y, t) / (ch * h * w)

        return sum_loss
