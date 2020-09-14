import numpy as np
import torch


def first_making(l: torch.Tensor,
                 m: torch.Tensor,
                 c: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    validsize = l.size(0)
    l = l[1].squeeze(1).repeat(validsize, 1, 1, 1)
    m = m[1].squeeze(1).repeat(validsize, 1, 1, 1)
    c = c[1].squeeze(1).repeat(validsize, 1, 1, 1)

    return l, m, c


def noise_generate(m: torch.Tensor,
                   latent_dim: int) -> torch.Tensor:

    batchsize = m.size(0)
    z = np.random.normal(size=(batchsize, latent_dim))
    z = torch.cuda.FloatTensor(z)

    return z