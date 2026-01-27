import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.gan import GANInverter
from src.data import get_mnist_loaders
from src.models.gan import Generator

generator = Generator(latent_size=64, image_size=784, hidden_size=256)
checkpoint = torch.load('/data/G_final.pth', map_location='cpu')

data_loader, _ = get_mnist_loaders(batch_size=32)
random_image, random_label = next(iter(data_loader))

inverter = GANInverter(generator)

img = random_image[0].view(-1)
z, reconstructed, losses = inverter.invert(img)

original_np = random_image[0].squeeze().numpy()
reconstructed_np = reconstructed.view(28, 28).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original_np, cmap='gray', vmin=-1, vmax=1)
axes[0].set_title(f"Original Image\nLabel: {random_label[0]}")
axes[0].axis('off')

axes[1].imshow(reconstructed_np, cmap='gray', vmin=-1, vmax=1)
axes[1].set_title("GAN Inversion Reconstruction")
axes[1].axis('off')

difference = np.abs(original_np - reconstructed_np)
im_diff = axes[2].imshow(difference, cmap='hot')
axes[2].set_title(f"Absolute Difference\nMSE: {losses[-1]:.6f}")
axes[2].axis('off')

plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('gan_inversion_result.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'gan_inversion_result.png'")
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(losses)
ax2.set_xlabel("Optimization Step")
ax2.set_ylabel("MSE Loss")
ax2.set_title("GAN Inversion Loss Curve")
ax2.grid(True, alpha=0.3)
plt.savefig('drugie.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'drugie.png'")
plt.close(fig2)
