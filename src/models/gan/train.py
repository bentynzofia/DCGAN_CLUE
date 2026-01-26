from torch import nn

from .generator import Generator
from .discriminator import Discriminator


class DCGANTrainer:
    def __init__(self, netG: Generator, netD: Discriminator):
        self.netG = netG
        self.netD = netD


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

