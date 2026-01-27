import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from src.models.gan import GANInverter, DCGANInverter
from src.data import get_mnist_loaders
from src.models.gan import Generator, DCGenerator


def plot_single_reconstruction(original, reconstructed, label, loss, filename):
    original_np = original.numpy()
    reconstructed_np = reconstructed.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_np, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title(f"Original Image\nLabel: {label}")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_np, cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title("GAN Inversion Reconstruction")
    axes[1].axis('off')

    difference = np.abs(original_np - reconstructed_np)
    im_diff = axes[2].imshow(difference, cmap='hot')
    axes[2].set_title(f"Absolute Difference\n{loss}: {losses[-1]:.6f}")
    axes[2].axis('off')

    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    print(f"Saved inversion result in {filename}.png'")
    plt.close(fig)


def plot_inversion_loss(losses, loss, filename):
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(losses)
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel(f"{loss} Loss")
    ax2.set_title("GAN Inversion Loss Curve")
    ax2.grid(True, alpha=0.3)
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    print(f"Saved inversion loss curve in {filename}.png")
    plt.close(fig2)


def plot_batch_reconstructions(original_batch, reconstructed_batch, filename):
    original_grid = vutils.make_grid(original_batch, nrow=8, normalize=True, scale_each=True)
    recon_grid = vutils.make_grid(reconstructed_batch, nrow=8, normalize=True, scale_each=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Images")
    axes[0].axis('off')

    axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Reconstructed Images")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150)
    print(f"Saved batch reconstructions '{filename}'")
    plt.close(fig)


data_loader, _ = get_mnist_loaders(batch_size=32)
random_images, random_labels = next(iter(data_loader))

generator = DCGenerator()
# generator = Generator(latent_size=64, image_size=784, hidden_size=256)

checkpoint = torch.load('save/G_final.pth', map_location='cpu')  # for GAN change path to 'save_GAN/G_final.pth'
generator.load_state_dict(checkpoint)

if isinstance(generator, DCGenerator):
    loss = 'BCE'
    inverter = DCGANInverter(generator)
    img = random_images[0].unsqueeze(0)
    z_batch, reconstructions, losses = inverter.invert_batch(
        random_images, use_regularization=True
    )

    plot_batch_reconstructions(random_images, reconstructions, filename="dcgan_batch_inversion_result")

    plot_single_reconstruction(random_images[0].squeeze(),
                               reconstructions[0].view(28, 28).cpu(),
                               random_labels[0], loss,
                               filename="dcgan_inversion_result")

    plot_inversion_loss(losses, loss, filename="dcgan_inversion_loss_curve")
else:
    loss = 'MSE'
    inverter = GANInverter(generator)
    img = random_images[0].view(-1)
    z, reconstructed, losses = inverter.invert(img)
    plot_single_reconstruction(random_images[0].squeeze(),
                               reconstructed[0].view(28, 28).cpu(),
                               random_labels[0], loss,
                               filename="gan_inversion_result.png")

    plot_inversion_loss(losses, loss, filename="gan_inversion_loss_curve")
