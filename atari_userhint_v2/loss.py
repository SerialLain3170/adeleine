import torch
import torch.nn as nn

from torch import autograd

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class LossCalculator:
    def __init__(self):
        pass

    def content_loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    def adversarial_disloss(self, discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        _, fake_list = discriminator(y)
        _, real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
            sum_loss += loss

        return sum_loss

    def adversarial_genloss(self, discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_adv_loss = 0
        sum_fm_loss = 0
        fake_points, fake_list = discriminator(y)
        real_points, _ = discriminator(t)

        for fake in fake_list:
            sum_adv_loss += torch.mean(softplus(-fake))

        d_weight = float(1.0 / 3.0)
        feat_weight = float(4.0 / 7.0)

        for f_feat, r_feat in zip(fake_points, real_points):
            sum_fm_loss += d_weight * feat_weight * maeloss(f_feat, r_feat.detach())

        return sum_adv_loss, sum_fm_loss

    def adversarial_hingedis(self, discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        sum_adv_loss = 0
        _, fake_list = discriminator(y)
        _, real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            sum_adv_loss += nn.ReLU()(1.0 + fake).mean()
            sum_adv_loss += nn.ReLU()(1.0 - real).mean()

        return sum_adv_loss

    def adversarial_hingegen(self, discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        sum_adv_loss = 0
        sum_fm_loss = 0
        fake_points, fake_list = discriminator(y)
        real_points, _ = discriminator(t)

        for fake in fake_list:
            sum_adv_loss += -fake.mean()

        d_weight = float(1.0 / 3.0)
        feat_weight = float(4.0 / 7.0)

        for f_feat, r_feat in zip(fake_points, real_points):
            sum_fm_loss += d_weight * feat_weight * maeloss(f_feat, r_feat.detach())

        return sum_adv_loss, sum_fm_loss

    def positive_enforcing_loss(self, y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        batch, ch, h, w = y.size()

        for color in range(3):
            perch = y[:, color, :, :]
            mean = torch.mean(perch)
            mean = mean * torch.ones_like(mean)
            loss = torch.mean((perch-mean)**2)
            sum_loss += loss

        return -sum_loss

    def perceptual_loss(self, vgg: nn.Module,
                        y: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        y_vgg = vgg(y)
        t_vgg = vgg(t)

        _, c, h, w = y.size()

        loss = maeloss(y_vgg, t_vgg) / (c * h * w)

        return loss

    def total_variation_loss(self, y: torch.Tensor) -> torch.Tensor:
        _, c, h, w = y.size()

        vertical_loss = torch.mean((torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))**2)
        horizon_loss = torch.mean((torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))**2)

        return (vertical_loss + horizon_loss) / (c * h * w)

    def gradient_penalty(self,
                         discriminator: nn.Module,
                         t: torch.Tensor
                         ) -> torch.Tensor:

        _, t_dis = discriminator(t)

        (grad_real,) = autograd.grad(
            outputs=t_dis[0].sum(), inputs=t, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        grad_penalty / 2.0 + 0 * t_dis[0][0]

        return grad_penalty
