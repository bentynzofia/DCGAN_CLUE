from .classifier import BayesianClassifier


class BNNTrainer:
    def __init__(self, model: BayesianClassifier, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, train_loader):
        total_loss = 0

        for data, target in train_loader:
            self.optimizer.zero_grad()

            loss = self.model.sample_elbo(
                inputs=data,
                labels=target,
                criterion=self.criterion,
                sample_nbr=3,
                complexity_cost_weight=1e-6
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(self, train_loader, epochs=10, verbose=True):
        losses = []

        for epoch in range(epochs):
            avg_loss = self.train_epoch(train_loader)
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model.save('BNN_results/saves/bnn.pth')

        return losses
