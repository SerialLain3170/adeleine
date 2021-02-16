import torch
import torch.nn as nn

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class WhiteBoxLossCalculator:
    def __init__(self):
        pass

    def content_loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    def adversarial_disloss(self, discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
            sum_loss += loss

        return sum_loss

    def adversarial_genloss(self, discriminator: nn.Module,
                            y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            loss = torch.mean(softplus(-fake))
            sum_loss += loss

        return sum_loss

    def adversarial_hingedis(self, discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            sum_loss += nn.ReLU()(1.0 + fake).mean()
            sum_loss += nn.ReLU()(1.0 - real).mean()

        return sum_loss

    def adversarial_hingegen(self, discriminator: torch.Tensor,
                             y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            sum_loss += -fake.mean()

        return sum_loss

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
