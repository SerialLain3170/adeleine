import torch
import torch.nn as nn

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class DiverseColorizeLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:

        return torch.mean(torch.abs(y - t))

    @staticmethod
    def latent_constrain_loss(y: torch.Tensor,
                              t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    @staticmethod
    def adversarial_disloss(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_genloss(discriminator: nn.Module,
                            y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            loss = torch.mean(softplus(-fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_hingedis(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            sum_loss += nn.ReLU()(1.0 + fake).mean()
            sum_loss += nn.ReLU()(1.0 - real).mean()

        return sum_loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            sum_loss += -fake.mean()

        return sum_loss

    @staticmethod
    def positive_enforcing_loss(y: torch.Tensor) -> torch.Tensor:
        sum_loss = 0
        batch, ch, h, w = y.size()

        for color in range(3):
            perch = y[:, color, :, :]
            mean = torch.mean(perch)
            mean = mean * torch.ones_like(mean)
            loss = torch.mean((perch-mean)**2)
            sum_loss += loss

        return -sum_loss

    @staticmethod
    def kl_loss(y: torch.Tensor) -> torch.Tensor:
        x_2 = torch.pow(y, 2)
        loss = torch.mean(x_2)

        return loss

    @staticmethod
    def mode_seeking_regularize(y: torch.Tensor,
                                z: torch.Tensor) -> torch.Tensor:
        batchsize = y.size(0)
        index = torch.randperm(batchsize).cuda()

        lz = torch.mean(torch.abs(y-y[index, :])) / (torch.mean(torch.abs(z-z[index, :])) + 1e-9)
        loss = 1 / (lz + 1e-5)

        return loss