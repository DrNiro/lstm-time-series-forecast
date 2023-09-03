import datetime
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score


class Session:
    def __init__(self, model, loss_fn, optimizer, epochs, name="LSTM"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.name = name
        self.train_losses = []
        self.val_losses = []
        self.val_score = []

        # If we have a GPU available, we'll set our device to cuda.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model.to(device)

    def train_step(self, x, y):
        # sets model to train mode
        self.model.train()

        # predict
        yhat = self.model(x)

        # computes loss and back propagation
        loss = self.loss_fn(yhat, y)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, val_loader, save=True):
        model_path = './models'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_save_name = f'{datetime.datetime.now().strftime("%m_%d_%H_%M")}_{self.name}.pt'
        save_path = os.path.join(model_path, model_save_name)

        print("Begin training")
        for epoch in range(1, self.epochs + 1):
            batch_losses = []
            for batch in train_loader:
                x_batch = batch['features'].to(self.device)
                y_batch = batch['target'].float().unsqueeze(1).to(self.device)

                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                batch_val_scores = []
                for batch in val_loader:
                    x_val = batch['features'].to(self.device)
                    y_val = batch['target'].float().unsqueeze(1).to(self.device)

                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(yhat, y_val).item()
                    batch_val_losses.append(val_loss)
                    batch_val_scores.append(scores(y_val.cpu(), yhat.cpu()))
                validation_loss = np.mean(batch_val_losses)
                val_score = np.mean(batch_val_scores)
                self.val_losses.append(validation_loss)
                self.val_score.append(val_score)

            if epoch % 10 == 0:
                print(f"[{epoch}/{self.epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

        if save:
            torch.save(self.model.state_dict(), save_path)

    def predict(self, test_loader):
        with torch.no_grad():
            predictions = []
            true_values = []
            for batch in test_loader:
                x_test = batch['features'].to(self.device)
                y_test = batch['target'].to(self.device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().cpu().numpy()[0])
                true_values.append(y_test.detach().cpu().numpy())

        return predictions, true_values

    def load_checkpoint(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

    def plot_validation_score(self):
        plt.plot(self.val_score, label='Validation score')
        plt.legend()
        plt.title('Model performance')
        plt.show()
        plt.close()

    def plot_pred(self, preds, true_vals):
        plt.plot(preds, label="Prediction")
        plt.plot(true_vals, label="True values")
        plt.legend()
        plt.title("Inference graph")
        plt.show()
        plt.close()


def scores(true_values, predictions, type='mae'):
    score_dict = {
        "mae": mae(true_values, predictions),
        "mse": mse(true_values, predictions),
        "mape": mape(true_values, predictions),
        "r2": r2_score(true_values, predictions)
    }
    return score_dict[type]

