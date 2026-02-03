"""
Loads a random MNIST image, gets certainty values from Bnn, finds initial GAN inversion (z_codeA).
Then run inversion with the z_codeA as a starting point, this time the loss is calculated solely with values obtained
by Bnn.

Plots original image vs initial inversion vs final inversion.
"""
import torch

from src.data import get_mnist_loaders
from src.utils import set_seed, plot_three_way_comparison
from src.models.gan import DCGenerator, GanBnnPipeline
from src.models.bnn import BayesianClassifier

set_seed(42)
samples = 10
path = "DCGAN_BNN_pipeline"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Bnn
bnn = BayesianClassifier.load('BNN_results/saves/bnn.pth').to(device)
bnn.eval()  # Might make the Bnn more deterministic, but not sure TODO: check

# Load Generator
generator = DCGenerator().to(device)
checkpoint = torch.load('DCGAN_results/saves/G_final.pth')
generator.load_state_dict(checkpoint)

# Set up inversion
inverter = GanBnnPipeline(generator)

# Load data
data_loader, _ = get_mnist_loaders(batch_size=32)
random_images, random_labels = next(iter(data_loader))
img, label = random_images[0:1].to(device), random_labels[0:1].to(device)

# Bnn result
with torch.no_grad():
    bnn_output = torch.stack([torch.softmax(bnn(img.view(1, -1)), dim=1) for _ in range(samples)])
    mean_probs = bnn_output.mean(dim=0)
    std_probs = bnn_output.std(dim=0)

pred_class = mean_probs.argmax(dim=1).item()
confidence = mean_probs[0, pred_class].item()
class_uncertainty = std_probs[0, pred_class].item()

stats_orig = {
    'pred': pred_class,
    'conf': confidence,
    'unc': class_uncertainty
}

print(f"Original Image is Class {pred_class} | Confidence: {confidence:.4f} | Uncertainty: {class_uncertainty:.4f}")

# Initial inversion
z_codeA, reconstructionA = inverter.invert(img)

# Latent space search wrt class certainty
z_codeB, reconstructionB, _, _, results = inverter.steer(bnn, label, z_init=z_codeA)

# Plotting
plot_three_way_comparison(
    img,
    reconstructionA,
    reconstructionB,
    stats_orig,
    results['initial'],
    results['final'],
    f"{path}/complete_pipeline_comparison"
)
