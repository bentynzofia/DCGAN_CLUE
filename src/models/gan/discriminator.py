from torch import nn


class Discriminator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),

            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCDiscriminator(nn.Module):
    def __init__(self, image_size=28, channels=1):
        super(DCDiscriminator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten = nn.Flatten()  # from (128, 7, 7) to 6272

        self.fc1 = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, 28, 28)

        x = self.conv1(x)  # (batch_size, 64, 14, 14)
        x = self.conv2(x)  # (batch_size, 128, 7, 7)

        x = self.flatten(x)  # (batch_size, 128*7*7)

        x = self.fc1(x)  # (batch_size, 1024)
        x = self.fc2(x)  # (batch_size, 1)

        return x
