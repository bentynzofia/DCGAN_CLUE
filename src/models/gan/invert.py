import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GANInverter:
    def __init__(self, generator, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator
        self.device = device
        self.generator.to(device)
        self.generator.eval()

    def invert(self, target_image, num_steps=1000, lr=0.02, latent_dim=64):
        target_image = target_image.to(self.device)
        z = torch.randn(1, latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([z], lr=lr)
        mse_loss = nn.MSELoss()
        losses = []

        for step in range(num_steps):
            optimizer.zero_grad()

            generated = self.generator(z)

            loss = mse_loss(generated, target_image)
            latent_reg = 0.001 * torch.norm(z)
            loss += latent_reg
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")

        with torch.no_grad():
            reconstructed = self.generator(z)

        return z.detach(), reconstructed.detach(), losses


class GanBnnPipeline:
    def __init__(self, generator, prior='normal', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator.to(device).eval()
        self.prior = prior
        self.device = device
        self.latent_dim = self._get_latent_dim()

    def _get_latent_dim(self):
        for m in self.generator.modules():
            if isinstance(m, nn.Linear): return m.in_features
            if isinstance(m, nn.ConvTranspose2d): return m.in_channels
        return 100

    def _initialize_z(self, batch_size, z_init=None):
        if z_init is not None:
            return z_init.clone().detach().to(self.device).requires_grad_(True)

        if self.prior == 'uniform':
            z = torch.rand(batch_size, self.latent_dim, device=self.device) * 2 - 1
        else:
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
        return z.requires_grad_(True)

    def _get_bnn_loss(self, generated, bnn, target_label, bnn_samples):
        bnn_input = generated.view(generated.size(0), -1)
        samples = torch.stack([torch.softmax(bnn(bnn_input), dim=1) for _ in range(bnn_samples)])

        mean_probs = samples.mean(dim=0)
        uncertainty = samples.std(dim=0).mean()

        class_loss = F.nll_loss(torch.log(mean_probs + 1e-8), target_label)
        return class_loss + (2.0 * uncertainty), mean_probs, uncertainty

    def invert(self, target_images, num_steps=1000, lr=0.02):  # Pixel loss only
        z = self._initialize_z(target_images.size(0))
        optimizer = optim.Adam([z], lr=lr)
        target_norm = (target_images.to(self.device) + 1) / 2.0

        for step in range(num_steps):
            optimizer.zero_grad()
            generated_norm = (self.generator(z) + 1) / 2.0
            loss = F.binary_cross_entropy(generated_norm, target_norm)

            loss += 0.01 * torch.norm(z)  # L2

            loss.backward()
            optimizer.step()
            if step % 200 == 0: print(f"Invert Step {step}, Loss: {loss.item():.4f}")

        return z.detach(), self.generator(z).detach()

    def steer(self, bnn, target_class_idx, track_history=False, z_init=None, num_steps=500, lr=0.01, bnn_samples=15):  # BNN loss only
        batch_size = 1
        z = self._initialize_z(batch_size, z_init)
        optimizer = optim.Adam([z], lr=lr)

        history_images = []
        history_stats = []

        results = {}

        if not isinstance(target_class_idx, torch.Tensor):
            target_label = torch.tensor([target_class_idx], device=self.device)
        else:
            target_label = target_class_idx.to(self.device)

        for step in range(num_steps):
            optimizer.zero_grad()
            generated = self.generator(z)

            total_loss, mean_probs, uncertainty = self._get_bnn_loss(generated, bnn, target_label, bnn_samples)
            if step == 0:
                results['initial'] = {
                    'img': generated.detach().cpu(),
                    'pred': mean_probs.argmax(dim=1).item(),
                    'conf': mean_probs[0, target_class_idx].item(),
                    'unc': uncertainty.item()
                }

            loss = total_loss + 0.01 * torch.norm(z)  # L2

            loss.backward()
            optimizer.step()

            if track_history:
                if step % 10 == 0 or step == num_steps - 1:
                    history_images.append(generated.detach().cpu())
                    current_pred = mean_probs.argmax(dim=1).item()
                    history_stats.append({
                        'step': step,
                        'conf': mean_probs[0, target_class_idx].item(),
                        'unc': uncertainty.item(),
                        'pred': current_pred
                    })

            if step % 100 == 0:
                prob = mean_probs[0, target_label.item()].item()
                print(f"Steer Step {step} | Conf: {prob:.6f} | Unc: {uncertainty.item():.6f}")

        final_gen = self.generator(z).detach()
        _, final_mean, final_unc = self._get_bnn_loss(final_gen, bnn, target_label, bnn_samples)

        results['final'] = {
            'img': final_gen.cpu(),
            'pred': final_mean.argmax(dim=1).item(),
            'conf': final_mean[0, target_class_idx].item(),
            'unc': final_unc.item()
        }
        return z.detach(), self.generator(z).detach(), history_images, history_stats, results
