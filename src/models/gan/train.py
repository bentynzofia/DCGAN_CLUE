import torch
from torch import nn
import os

from .generator import Generator
from .discriminator import Discriminator

from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pylab
import numpy as np


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_training_curves(epochs, save_dir, current_epoch, d_losses, g_losses, real_scores, fake_scores):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    pylab.xlim(0, epochs + 1)
    plt.plot(range(1, current_epoch + 1), d_losses[:current_epoch], label='Discriminator Loss')
    plt.plot(range(1, current_epoch + 1), g_losses[:current_epoch], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')

    plt.subplot(1, 2, 2)
    pylab.xlim(0, epochs + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, current_epoch + 1), fake_scores[:current_epoch], label='Fake Score D(G(z))')
    plt.plot(range(1, current_epoch + 1), real_scores[:current_epoch], label='Real Score D(x)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Discriminator Scores')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


class DCGANTrainer:
    def __init__(self, generator: Generator, discriminator: Discriminator,
                 g_optimizer, d_optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.G = generator
        self.D = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device

        self.G.to(device)
        self.D.to(device)

        self.criterion = nn.BCELoss()

    def train_discriminator(self, real_images):
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(self.device)
        outputs = self.D(real_images)
        d_loss_real = self.criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, self.G.latent_size).to(self.device)
        fake_images = self.G(z)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        outputs = self.D(fake_images.detach())
        d_loss_fake = self.criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator(self, batch_size):
        z = torch.randn(batch_size, self.G.latent_size).to(self.device)
        fake_images = self.G(z)
        labels = torch.ones(batch_size, 1).to(self.device)

        outputs = self.D(fake_images)
        g_loss = self.criterion(outputs, labels)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss, fake_images

    def train_epoch(self, epochs, data_loader, epoch, total_step, d_losses, g_losses, real_scores, fake_scores):
        for i, (images, _) in enumerate(data_loader):
            real_images = images.to(self.device)
            batch_size = real_images.size(0)

            d_loss, real_score, fake_score = self.train_discriminator(real_images)

            g_loss, fake_images = self.train_generator(batch_size)

            d_losses[epoch] = d_losses[epoch] * (i / (i + 1.)) + d_loss.item() * (1. / (i + 1.))
            g_losses[epoch] = g_losses[epoch] * (i / (i + 1.)) + g_loss.item() * (1. / (i + 1.))
            real_scores[epoch] = real_scores[epoch] * (i / (i + 1.)) + real_score.mean().item() * (1. / (i + 1.))
            fake_scores[epoch] = fake_scores[epoch] * (i / (i + 1.)) + fake_score.mean().item() * (1. / (i + 1.))

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch + 1, epochs, i + 1, total_step,
                              d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        return real_images, fake_images

    def train(self, data_loader, epochs=2, save_dir='save', sample_dir='samples'):
        d_losses = np.zeros(epochs)
        g_losses = np.zeros(epochs)
        real_scores = np.zeros(epochs)
        fake_scores = np.zeros(epochs)
        total_step = len(data_loader)

        for epoch in range(epochs):
            self.G.train()
            self.D.train()

            real_images, fake_images = self.train_epoch(
                epochs, data_loader, epoch, total_step, d_losses, g_losses, real_scores, fake_scores
            )

            if (epoch + 1) == 1:
                save_image(denorm(real_images.data[:25]),
                           os.path.join(sample_dir, 'real_images.png'),
                           nrow=5, normalize=True)

            fake_images_for_saving = fake_images
            if fake_images.dim() == 2 and fake_images.shape[1] == 784:
                fake_images_for_saving = fake_images.view(-1, 1, 28, 28)

            save_image(denorm(fake_images_for_saving.data[:25]),
                       os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)),
                       nrow=5, normalize=True)

            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses[:epoch + 1])
            np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses[:epoch + 1])
            np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores[:epoch + 1])
            np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores[:epoch + 1])

            plot_training_curves(epochs, save_dir, epoch + 1, d_losses, g_losses, real_scores, fake_scores)

            if (epoch + 1) % 10 == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(save_dir, 'G_epoch_{}.pth'.format(epoch + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(save_dir, 'D_epoch_{}.pth'.format(epoch + 1)))

            print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch + 1, epochs,
                          d_losses[epoch], g_losses[epoch],
                          real_scores[epoch], fake_scores[epoch]))

        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G_final.pth'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, 'D_final.pth'))

        print('Training completed')
