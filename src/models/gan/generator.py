from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_size=64, image_size=784, hidden_size=256):
        super(Generator, self).__init__()
        self.latent_size = latent_size

        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class DCGenerator(nn.Module):
    def __init__(self, latent_size=100, image_size=28, channels=1):
        super(DCGenerator, self).__init__()
        self.latent_size = latent_size
        self.image_size = image_size
        self.channels = channels

        self.fc1 = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 6272),  # 128 * 7 * 7 = 6272
            nn.BatchNorm1d(6272),
            nn.ReLU(inplace=True)
        )

        self.reshape = lambda x: x.view(-1, 128, 7, 7)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.tanh = nn.Tanh()

    def forward(self, z):
        # z shape: (batch_size, latent_size)

        x = self.fc1(z)  # (batch_size, 1024)
        x = self.fc2(x)  # (batch_size, 6272)

        x = self.reshape(x)

        x = self.conv1(x)  # (batch_size, 64, 14, 14)
        x = self.conv2(x)  # (batch_size, channels, 28, 28)

        x = self.tanh(x)

        return x
