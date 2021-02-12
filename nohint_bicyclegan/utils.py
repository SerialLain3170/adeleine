import numpy as np
import torch
import shutil
import datetime

from pathlib import Path


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


def session(session_name: str) -> (Path, Path, Path):
    session_path = Path("session") / Path(session_name)
    if session_path.exists():
        dt = datetime.datetime.now()
        dt = dt.strftime('%m%d-%H%M-%S%f')[:-4]
        session_name = f"{session_name}.{dt}"
        session_path = Path("session") / Path(session_name)

    modeldir_path = session_path / "ckpts"
    outdir_path = session_path / "vis"
    outdir_fix_path = session_path / "vis_fix"

    modeldir_path.mkdir(exist_ok=True, parents=True)
    outdir_path.mkdir(exist_ok=True, parents=True)
    outdir_fix_path.mkdir(exist_ok=True, parents=True)

    shutil.copy("param.yaml", session_path)

    return outdir_path, outdir_fix_path, modeldir_path
