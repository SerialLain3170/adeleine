import torch
import torch.nn as nn

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class SCFTLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def gram_matrix(y: torch.Tensor) -> torch.Tensor:
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    @staticmethod
    def content_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y-t))

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
        fake = discriminator(y)
        real = discriminator(t)

        fake_loss = nn.ReLU()(1.0 + fake).mean()
        real_loss = nn.ReLU()(1.0 - real).mean()

        return fake_loss + real_loss

    @staticmethod
    def adversarial_hingegen(discriminator: nn.Module,
                             y: torch.Tensor) -> torch.Tensor:
        fake = discriminator(y)

        return -fake.mean()

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

    def style_and_perceptual_loss(self,
                                  vgg: nn.Module,
                                  y: torch.Tensor,
                                  t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        sum_loss = 0
        y_list = vgg(y)
        t_list = vgg(t)

        for index, (y_feat, t_feat) in enumerate(zip(y_list, t_list)):
            batch, ch, h, w = y_feat.size()

            loss = maeloss(y_feat, t_feat)
            sum_loss += loss

            if index == 3:
                gram_y = self.gram_matrix(y_feat)
                gram_t = self.gram_matrix(t_feat)
                style_loss = maeloss(gram_y, gram_t)

        return 0.01 * sum_loss, 50.0 * style_loss
