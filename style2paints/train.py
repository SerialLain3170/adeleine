import torch
import torch.nn as nn
import argparse

from pathlib import Path
from tqdm import tqdm
from model import Style2Paint, Discriminator
from torch.utils.data import DataLoader
from dataset import IllustDataset, LineCollator

maeloss = nn.L1Loss()
softplus = nn.Softplus()


def content_loss(y, t):
    return 10 * maeloss(y, t)


def adversarial_disloss(discriminator, y, t):
    fake = discriminator(y)
    real = discriminator(t)

    return torch.mean(softplus(-real)) + torch.mean(softplus(fake))


def adversarial_genloss(discriminator, y):
    fake = discriminator(y)

    return torch.mean(softplus(-fake))


def adversarial_hingedis(discriminator, y, t):
    fake = discriminator(y)
    real = discriminator(t)

    fake_loss = nn.ReLU()(1.0 + fake).mean()
    real_loss = nn.ReLU()(1.0 - real).mean()

    return fake_loss + real_loss


def adversarial_hingegen(discriminator, y):
    fake = discriminator(y)

    return -fake.mean()


def train(epochs, batchsize, iterations, data_path, modeldir):
    # Dataset Definition
    dataset = IllustDataset(data_path)
    print(dataset)
    collator = LineCollator()

    # Model & Optimizer Definition
    model = Style2Paint()
    model.cuda()
    model.train()
    gen_opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    discriminator = Discriminator()
    discriminator.cuda()
    discriminator.train()
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    iteration = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                collate_fn=collator,
                                drop_last=True)
        progress_bar = tqdm(dataloader)

        for index, data in enumerate(progress_bar):
            iteration += 1
            color, line = data
            y = model(line, color)
            loss = adversarial_disloss(discriminator, y.detach(), color)

            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

            y = model(line, color)
            loss = adversarial_genloss(discriminator, y)
            loss += maeloss(y, color)

            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

            if iteration % iterations == 0:
                torch.save(model.state_dict(), f"{modeldir}/model_{iteration}.pt")

            print(f"iteration: {iteration} Loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style2Paint")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--i", type=int, default=2000, help="the number of iterations")

    args = parser.parse_args()

    data_path = Path("danbooru-images")
    modeldir = Path("modeldir")
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.b, args.i, data_path, modeldir)
