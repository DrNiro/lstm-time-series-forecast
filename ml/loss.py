from torch import nn, mean, abs, sqrt


class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        loss = mean(abs((yhat - y) / y))
        return loss


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Used to avoid NaN at backprop

    def forward(self, yhat, y):
        return sqrt(self.mse(yhat, y) + self.eps)


def get_loss_fn(name):
    losses = {
        'mse': nn.MSELoss(),
        'rmse': RMSELoss(),
        'mape': MAPELoss()
    }
    return losses[name]
