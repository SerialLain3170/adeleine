import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, initializers, serializers
import numpy as np
import os
import argparse
from model import Generator, Discriminator, VGG

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def calc_content_loss(fake, real):
    _, c,w,h = fake.shape
    return F.mean_squared_error(fake, real) / (c*w*h)

parser = argparse.ArgumentParser(description="Colorization")
parser.add_argument("--epoch",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",type=16,type=int,help="batchsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--testsize",default=2,type=int,help="testsize")
parser.add_argument("--aw",default=10.0,type=float,help="the weight of adversarial loss")
parser.add_argument("--pw",default=10.0,type=float,help="the weight of penalty loss")
parser.add_argument("--dw",default=0.001,type=float,help="epsilon drift")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
testsize = args.testsize
adver_weight = args.aw
penalty_weight = args.pw
epsilon_drift = args.dw

output_dir = "./output/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0
    for batch in range(0,Ntrain,batchsize):
        line_box = []
        color_box = []
        for index in range(batchsize):


        line = xp.array(line_box).astype(xp.float32)
        color = xp.array(color_box).astype(xp.float32)
        line = chainer.as_variable(line)
        color = chainer.as_variable(color)

        fake = generator(line)
        y_dis = discriminator(fake)
        t_dis = discriminator(color)
        dis_loss = F.mean(F.softplus(y_dis)) + F.mean(F.softplus(-t_dis))

        fake.unchain_backward()

        rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
        x_perturbed = Variable(cuda.to_gpu(rnd_x*fake + (1.0-rnd_x)*color))

        y_perturbed, _  = discriminator(x_perturbed)
        grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))

        dis_loss += penalty_weight * loss_grad

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        fake = generator(line)
        y_dis = discriminator(fake)
        gen_loss = adver_weight * F.mean(F.softplus(-y_dis))
        fake_feat = vgg(fake)
        real_feat = vgg(color)
        vgg_loss = calc_content_loss(fake_feat, real_feat)
        gen_loss += vgg_loss

        vgg.cleargrads()
        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        vgg_opt.update()
        gen_loss.unchain_backward()

        sum_gen_loss += gen_loss.data.get()
        sum_dis_loss += dis_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("generator.model",generator)
    
    print("epoch : {}".format(epoch))
    print("Generator loss : {}".format(sum_gen_loss))
    print("Discriminator loss : {}".format(sum_dis_loss))
