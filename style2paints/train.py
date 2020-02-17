import torch
import torch.nn as nn
import argparse

from pathlib import Path
from tqdm import tqdm
from model import Style2Paint, Discriminator
from torch.utils.data import DataLoader
from dataset import IllustDataset, LineCollator
from evaluation import Visualizer

maeloss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


class Style2paintsLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return 10 * torch.mean(torch.abs(y-t))

    @staticmethod
    def adversarial_disloss(discriminator, y, t):
        sum_loss = 0
        fake_list = discriminator(y)
        real_list = discriminator(t)

        for fake, real in zip(fake_list, real_list):
            loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_genloss(discriminator, y):
        sum_loss = 0
        fake_list = discriminator(y)

        for fake in fake_list:
            loss = torch.mean(softplus(-fake))
            sum_loss += loss

        return sum_loss

    @staticmethod
    def adversarial_hingedis(discriminator, y, t):
        fake = discriminator(y)
        real = discriminator(t)

        fake_loss = nn.ReLU()(1.0 + fake).mean()
        real_loss = nn.ReLU()(1.0 - real).mean()

        return fake_loss + real_loss

    @staticmethod
    def adversarial_hingegen(discriminator, y):
        fake = discriminator(y)

        return -fake.mean()

    @staticmethod
    def positive_enforcing_loss(y):
        sum_loss = 0
        batch, ch, h, w = y.size()

        for color in range(3):
            perch = y[:, color, :, :]
            mean = torch.mean(perch)
            mean = mean * torch.ones_like(mean)
            loss = torch.mean((perch-mean)**2)
            sum_loss += loss

        return -sum_loss


def train(epochs,
          interval,
          batchsize,
          validsize,
          data_path,
          sketch_path,
          extension,
          img_size,
          outdir,
          modeldir,
          learning_rate):

    # Dataset Definition
    dataset = IllustDataset(data_path, sketch_path, extension)
    c_valid, l_valid = dataset.valid(validsize)
    print(dataset)
    collator = LineCollator(img_size)

    # Model & Optimizer Definition
    model = Style2Paint(attn_type="adain")
    model.cuda()
    model.train()
    gen_opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    discriminator = Discriminator()
    discriminator.cuda()
    discriminator.train()
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss function definition
    lossfunc = Style2paintsLossCalculator()

    # Visualizer definition
    visualizer = Visualizer()

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
            loss = 0.01 * lossfunc.adversarial_disloss(discriminator, y.detach(), color)

            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

            y = model(line, color)
            loss = 0.01 * lossfunc.adversarial_genloss(discriminator, y)
            loss += maeloss(y, color)
            loss += 0.001 * lossfunc.positive_enforcing_loss(y)

            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

            if iteration % interval == 1:
                torch.save(model.state_dict(), f"{modeldir}/model_{iteration}.pt")

                with torch.no_grad():
                    y = model(l_valid, c_valid)

                c = c_valid.detach().cpu().numpy()
                l = l_valid.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                visualizer(l, c, y, outdir, iteration, validsize)

            print(f"iteration: {iteration} Loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style2Paint")
    parser.add_argument("--e", type=int, default=1000, help="The number of epochs")
    parser.add_argument("--i", type=int, default=2000, help="The interval of snapshot")
    parser.add_argument("--b", type=int, default=4, help="batch size")
    parser.add_argument('--v', type=int, default=12, help="valid size")
    parser.add_argument('--ext', type=str, default='.jpg', help="extension of training images")
    parser.add_argument('--size', type=int, default=224, help="size of training images")
    parser.add_argument('--outdir', type=Path, default='outdir', help="output directory")
    parser.add_argument('--modeldir', type=Path, default='modeldir', help="model output directory")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate of Adam")
    parser.add_argument('--data_path', type=Path, help="path containing color images")
    parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, args.data_path, args.sketch_path,
          args.ext, args.size, outdir, modeldir, args.lr)
