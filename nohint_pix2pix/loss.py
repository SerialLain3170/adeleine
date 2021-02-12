import torch
import numpy as np
import torch.nn as nn

from torch import autograd
from torch.autograd import Variable

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class Pix2pixCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - t))

    @staticmethod
    def adversarial_disloss(discriminator: nn.Module,
                            y: torch.Tensor,
                            t: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)
        real = discriminator(t)

        loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))

        return loss

    @staticmethod
    def adversarial_genloss(discriminator: nn.Module,
                            y: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        loss = torch.mean(softplus(-fake))

        return loss

    @staticmethod
    def adversarial_hingedis(discriminator: nn.Module,
                             y: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        real = discriminator(t)

        loss = nn.ReLU()(1.0 + fake).mean()
        loss += nn.ReLU()(1.0 - real).mean()

        return loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:

        fake = discriminator(y)
        loss = -fake.mean()

        return loss

    @staticmethod
    def gradient_penalty(discriminator: nn.Module,
                         t: torch.Tensor,
                         center="zero") -> torch.Tensor:

        alpha = torch.cuda.FloatTensor(np.random.random(size=t.shape))
        epsilon = torch.rand(t.size()).cuda()
        interpolates = alpha * t + ((1 - alpha) * (t + 0.5 * t.std() * epsilon))
        interpolates = Variable(interpolates, requires_grad=True)

        d_interpolates = discriminator(interpolates)

        fake = Variable(torch.cuda.FloatTensor(t.shape[0], 1, 8, 8).fill_(1.0), requires_grad=False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if center == "one":
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        elif center == "zero":
            gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean()

        return gradient_penalty
