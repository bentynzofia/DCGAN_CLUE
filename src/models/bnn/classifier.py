import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


@variational_estimator
class BayesianClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        self.blinear1 = BayesianLinear(input_dim, hidden_dims)
        self.relu = nn.ReLU()
        self.blinear2 = BayesianLinear(hidden_dims, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.relu(self.blinear1(x))
        x = self.blinear2(x)
        return x

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.network[0].in_features,
                'hidden_dims': [module.out_features for module in self.network
                                if isinstance(module, nn.Linear)][:-1],
                'num_classes': self.network[-1].out_features
            }
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        return model
