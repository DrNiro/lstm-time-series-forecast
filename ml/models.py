import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, bidirectional):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=self.bidirectional)

        # Weights initialization
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Fully connected layer
        self.fc_dim = self.hidden_dim if not self.bidirectional else self.hidden_dim * 2

        self.fc = nn.Linear(self.fc_dim, output_dim)

    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm(x)

        # reshape output of shape (batch_size, seq_length, hidden_size) to fit the FC layer.
        out = out[:, -1, :]

        # FC layer
        out = self.fc(out)

        return out
