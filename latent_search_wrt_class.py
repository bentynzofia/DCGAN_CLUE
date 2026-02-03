"""
Search the latent space of Generator, starting from random init,
to find an image that yields low uncertainty in the defined label when put through BNN
"""
import torch

from src.utils import set_seed, plot_reconstructions, plot_progression
from src.models.gan import DCGenerator, GanBnnPipeline
from src.models.bnn import BayesianClassifier

set_seed(42)
samples = 10
path = "DCGAN_BNN_pipeline"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnn = BayesianClassifier.load('BNN_results/saves/bnn.pth').to(device)
bnn.eval()

generator = DCGenerator()
checkpoint = torch.load('DCGAN_results/saves/G_final.pth')
generator.load_state_dict(checkpoint)

inverter = GanBnnPipeline(generator)

z_code, reconstruction, history_images, history_stats, _ = inverter.steer(bnn=bnn, target_class_idx=2,
                                                                          track_history=True)

plot_progression(history_images, history_stats, 2, 'from_random')

# TODO: why sampled z gives good results form the start?
#    And more so, why the stats are so low for such a nice image? why not predict the 5, even we search for 2
