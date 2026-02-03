import os
import random
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Seed set to: {seed}")


def plot_bnn_predictions(img, label, tag, mean_probs, std_probs, pred_class, confidence, uncertainty):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    title = f"Pred: {pred_class} | Conf: {confidence:.2f} | Uncert: {uncertainty:.2f}"
    if label is not None:
        title = f"Pred: {pred_class} | True: {label} | Conf: {confidence:.2f} | Uncert: {uncertainty:.2f}"
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), mean_probs.cpu().numpy(), yerr=std_probs.cpu().numpy(), capsize=5)
    plt.title("Class Probabilities w/ Uncertainty")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(range(10))

    plt.tight_layout()
    plt.savefig(f"bnn_prediction_{tag}.png")


def plot_reconstructions(reconstruction, filename):
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    recon_grid = vutils.make_grid(reconstruction, nrow=8, normalize=True, scale_each=True)
    recon_np = recon_grid.permute(1, 2, 0).detach().cpu().numpy()

    axes.imshow(recon_np)
    axes.set_title("Reconstructed Image")
    axes.axis('off')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150)
    print(f"Saved reconstruction to '{filename}.png'")
    plt.close(fig)


def plot_single_reconstruction(original, reconstructed, label, loss, filename, losses):
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


def plot_batch_reconstructions(original_batch, reconstructed_batch, filename, title0="Original Images", title1="Reconstructed Images"):
    original_grid = vutils.make_grid(original_batch, nrow=8, normalize=True, scale_each=True)
    recon_grid = vutils.make_grid(reconstructed_batch, nrow=8, normalize=True, scale_each=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(title0)
    axes[0].axis('off')

    axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(title1)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150)
    print(f"Saved batch reconstructions '{filename}'")
    plt.close(fig)


def plot_three_way_comparison(img_orig, img_init, img_final, stats_orig, stats_init, stats_final, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    images = [img_orig, img_init, img_final]
    stats_list = [stats_orig, stats_init, stats_final]
    titles = ["Original Image", "Initial Inversion", "Final Inversion"]

    for i in range(3):
        grid = vutils.make_grid(images[i], normalize=True, scale_each=True)
        axes[i].imshow(grid.permute(1, 2, 0).cpu().numpy())

        # Build the title string
        s = stats_list[i]
        stat_text = f"{titles[i]}\n\nPred: {s['pred']}\nConf: {s['conf']:.2f}, Unc: {s['unc']:.4f}"

        # pad=30 ensures the title is well above the image
        axes[i].set_title(stat_text, fontsize=11, pad=15)
        axes[i].axis('off')

    # Adjust layout to prevent clipping
    plt.subplots_adjust(top=0.8, wspace=0.2)
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    print(f"Saved 3-way comparison to '{filename}.png'")
    plt.close(fig)

def plot_progression(history_images, history_stats, target, filename):
    n_steps = len(history_images)
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 3, 4))

    if n_steps == 1: axes = [axes]

    for i, (img, stats) in enumerate(zip(history_images, history_stats)):
        grid = vutils.make_grid(img, normalize=True, scale_each=True)
        img_np = grid.permute(1, 2, 0).numpy()

        axes[i].imshow(img_np)
        axes[i].set_title(f"Step {stats['step']}\nTarger: {target} Pred: {stats['pred']}\nConf: {stats['conf']:.6f}\nUnc: {stats['unc']:.6f}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', bbox_inches='tight')


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