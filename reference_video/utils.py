import torch
import shutil
import datetime

from pathlib import Path


def session(session_name):
    session_path = Path("session") / Path(session_name)
    if session_path.exists():
        dt = datetime.datetime.now()
        dt = dt.strftime('%m%d-%H%M-%S%f')[:-4]
        session_name = f"{session_name}.{dt}"
        session_path = Path("session") / Path(session_name)

    modeldir_path = session_path / "ckpts"
    outdir_path = session_path / "vis"

    modeldir_path.mkdir(exist_ok=True, parents=True)
    outdir_path.mkdir(exist_ok=True, parents=True)

    shutil.copy("param.yaml", session_path)

    return outdir_path, modeldir_path


def sum_totensor(l_x, l_y0, l_y1, c_x, c_y0, c_y1, d_x, d_y0, d_y1):
    l_x = l_x.cuda()
    l_y0 = l_y0.cuda()
    l_y1 = l_y1.cuda()
    c_x = c_x.cuda()
    c_y0 = c_y0.cuda()
    c_y1 = c_y1.cuda()
    d_x = d_x.cuda()
    d_y0 = d_y0.cuda()
    d_y1 = d_y1.cuda()

    return l_x, l_y0, l_y1, c_x, c_y0, c_y1, d_x, d_y0, d_y1


def movie_prepare(l_x, l_y0, l_y1, y, y0, y1):
    y = torch.cat([l_x, y], dim=1).unsqueeze(2)
    y0 = torch.cat([l_y0, y0], dim=1).unsqueeze(2)
    y1 = torch.cat([l_y1, y1], dim=1).unsqueeze(2)

    return torch.cat([y0, y, y1], dim=2)
