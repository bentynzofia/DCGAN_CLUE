from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=64, root='data'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
