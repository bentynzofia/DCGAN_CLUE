import os
import torch
import torch.nn as nn

from src.data import get_mnist_loaders
from src.models.gan import (Generator, Discriminator,
                            DCGenerator, DCDiscriminator,
                            DCGANTrainer, weights_init)

data_loader, _ = get_mnist_loaders(batch_size=32)

G, D = DCGenerator(), DCDiscriminator()
# G, D = Generator(), Discriminator()

if isinstance(G, DCGenerator) and isinstance(D, DCDiscriminator):
    G.apply(weights_init)
    D.apply(weights_init)

    sample_dir = 'DCGAN_results/training_samples'
    save_dir = 'DCGAN_results/saves'
else:
    sample_dir = 'GAN_results/training_samples'
    save_dir = 'GAN_results/saves'

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

trainer = DCGANTrainer(G, D, g_optimizer, d_optimizer)
trainer.train(data_loader, epochs=20, sample_dir=sample_dir, save_dir=save_dir)
