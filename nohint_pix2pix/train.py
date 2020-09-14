import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import argparse

from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from dataset import IllustDataset
from visualize import Visualizer
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


class Trainer:
	def __init__(self,
				 config,
				 outdir,
				 modeldir,
				 data_path,
				 sketch_path,
				 ):

		self.train_config = config["train"]
		self.data_config = config["dataset"]
		model_config = config["model"]
		self.loss_config = config["loss"]

		self.outdir = outdir
		self.modeldir = modeldir
		self.mask = self.train_config["mask"]

		self.dataset = IllustDataset(data_path,
									sketch_path,
									self.data_config["extension"],
									self.data_config["train_size"],
									self.data_config["valid_size"],
									self.data_config["color_space"],
									self.data_config["line_space"])
		print(self.dataset)

		if self.mask:
			in_ch = 6
		else:
			in_ch = 3

		gen = Generator(in_ch=in_ch)
		self.gen, self.gen_opt = self._setting_model_optim(gen,
														   model_config["generator"])

		dis = Discriminator()
		self.dis, self.dis_opt = self._setting_model_optim(dis,
														   model_config["discriminator"])

		self.lossfunc = Pix2pixCalculator()
		self.visualizer = Visualizer(self.data_config["color_space"])

	@staticmethod
	def _setting_model_optim(model: nn.Module,
							config: Dict):
		model.cuda()
		if config["mode"] == "train":
			model.train()
		elif config["mode"] == "eval":
			model.eval()

		optimizer = torch.optim.Adam(model.parameters(),
									lr=config["lr"],
									betas=(config["b1"], config["b2"]))

		return model, optimizer

	@staticmethod
	def _valid_prepare(dataset, validsize: int, mask: bool) -> List[torch.Tensor]:
		c_val, l_val, m_val = dataset.valid(validsize)

		if mask:
			x_val = torch.cat([l_val, m_val], dim=1)
		else:
			x_val = l_val

		return [x_val, l_val, m_val, c_val]

	def _eval(self,
			  iteration: int,
			  validsize: int,
			  v_list: List[torch.Tensor]):
		torch.save(self.gen.state_dict(),
				   f"{self.modeldir}/model_{iteration}.pt")

		with torch.no_grad():
			y = self.gen(v_list[0])

		self.visualizer(v_list[1:], y,
						self.outdir, iteration, validsize)

	def _iter(self, data):
		color, line, mask = data
		color = color.cuda()
		line = line.cuda()
		mask = mask.cuda()

		if self.mask:
			y = self.gen(torch.cat([line, mask], dim=1))
		else:
			y = self.gen(line)

		# discriminate images themselve
		loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingedis(self.dis,
																y.detach(),
																color)
		loss += self.loss_config["gp"] * self.lossfunc.gradient_penalty(self.dis,
																		color,
																		center=self.loss_config["center"])

		self.dis_opt.zero_grad()
		loss.backward()
		self.dis_opt.step()

		loss = self.loss_config["adv"] * self.lossfunc.adversarial_hingegen(self.dis,
																y)

		# gray scale
		loss += self.loss_config["content"] * self.lossfunc.content_loss(y,
																		color)

		self.gen_opt.zero_grad()
		loss.backward()
		self.gen_opt.step()

	def __call__(self):
		iteration = 0
		v_list = self._valid_prepare(self.dataset,
									 self.train_config["validsize"],
									 self.mask)

		for epoch in range(self.train_config["epoch"]):
			dataloader = DataLoader(self.dataset,
									batch_size=self.train_config["batchsize"],
									shuffle=True,
									drop_last=True)
			progress_bar = tqdm(dataloader)

			for index, data in enumerate(progress_bar):
				iteration += 1
				self._iter(data)

				if iteration % self.train_config["snapshot_interval"] == 1:
					self._eval(iteration,
							   self.train_config["validsize"],
							   v_list,
							   )


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Style2Paint")
	parser.add_argument('--outdir', type=Path, default='outdir', help="output directory")
	parser.add_argument('--modeldir', type=Path, default='modeldir', help="model output directory")
	parser.add_argument('--data_path', type=Path, help="path containing color images")
	parser.add_argument('--sketch_path', type=Path, help="path containing sketch images")
	args = parser.parse_args()

	outdir = args.outdir
	outdir.mkdir(exist_ok=True)

	modeldir = args.modeldir
	modeldir.mkdir(exist_ok=True)

	with open("param.yaml", "r") as f:
		config = yaml.safe_load(f)

	trainer = Trainer(config,
					  outdir,
					  modeldir,
					  args.data_path,
					  args.sketch_path,
					  )
	trainer()