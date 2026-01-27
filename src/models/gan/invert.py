import torch
import torch.nn as nn
import torch.optim as optim


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


class DCGANInverter:
    def __init__(self, generator, prior='uniform', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator
        self.prior = prior  # 'normal' or 'uniform'
        self.device = device
        self.generator.to(device)
        self.generator.eval()

    def invert(self, target_images, num_steps=1000, lr=0.02,
               use_regularization=False, gamma1=1.0, gamma2=1.0):
        """
        target_images: Tensor of shape (B, C, H, W)
        Following Algorithm 1 and Section 2.3 of the paper
        """
        target_images = target_images.to(self.device)
        B, latent_dim = target_images.shape[0], self.get_latent_dim()

        # Step 2: Initialize from prior (Algorithm 1)
        if self.prior == 'uniform':
            # Start near 0, not full uniform
            z = torch.randn(B, latent_dim, device=self.device) * 0.1
        else:  # normal
            # Start with small variance
            z = torch.randn(B, latent_dim, device=self.device) * 0.1

        z.requires_grad_(True)

        optimizer = optim.Adam([z], lr=lr)
        losses = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Generate images
            generated = self.generator(z)

            # PAPER'S LOSS: Binary Cross-Entropy (Eqn 1)
            # Convert from [-1, 1] (tanh) to [0, 1] for BCE
            target_norm = (target_images + 1) / 2.0
            generated_norm = (generated + 1) / 2.0

            # BCE loss
            loss = -torch.mean(
                target_norm * torch.log(generated_norm + 1e-8) +
                (1 - target_norm) * torch.log(1 - generated_norm + 1e-8)
            )

            # PAPER'S REGULARIZATION (Section 2.3, Eqn 5) - only if requested
            if use_regularization and self.prior == 'normal':
                mu_z = torch.mean(z)
                sigma_z = torch.std(z)
                reg_loss = gamma1 * (mu_z ** 2) + gamma2 * ((sigma_z - 1) ** 2)
                loss = loss + reg_loss

            loss.backward()
            optimizer.step()

            # PAPER'S CLIPPING for uniform prior (Section 2.3)
            if use_regularization and self.prior == 'uniform':
                with torch.no_grad():
                    z.data = torch.clamp(z.data, -1, 1)

            losses.append(loss.item())

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")

        with torch.no_grad():
            reconstructed = self.generator(z)

        return z.detach(), reconstructed.detach(), losses

    def get_latent_dim(self):
        """Infer latent dimension from generator"""
        # Check first layer of generator
        for module in self.generator.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
            elif hasattr(module, 'l1'):  # For DCGAN structure
                return module.l1[0].in_features
        return 100

    # ADDITIONAL: Batch inversion helper
    def invert_batch(self, target_images, **kwargs):
        """Wrapper to ensure batch inversion"""
        return self.invert(target_images, **kwargs)
