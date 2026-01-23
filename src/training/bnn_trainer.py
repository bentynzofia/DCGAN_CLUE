import torch.nn as nn
import torch.optim as optim

from ..data import train_loader
from ..models.bnn import BayesianClassifier, BNNTrainer

model = BayesianClassifier(input_dim=784, hidden_dims=[100], num_classes=10)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = BNNTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
)

losses = trainer.train(train_loader, epochs=10)
