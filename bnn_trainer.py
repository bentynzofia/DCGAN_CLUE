import torch.nn as nn
import torch.optim as optim

from src.data import get_mnist_loaders
from src.models.bnn import BayesianClassifier, BNNTrainer

train_loader, test_loader = get_mnist_loaders()

model = BayesianClassifier(input_dim=784, hidden_dims=400, num_classes=10)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = BNNTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
)

losses = trainer.train(train_loader, epochs=2)
