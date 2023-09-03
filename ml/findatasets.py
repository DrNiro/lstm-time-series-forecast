import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data_df, target='Close', window_size=17):
        self.features = [
            'Volume', 'MACD', 'rsi', 'diff_1', 'Close',
            'ma7', 'ma21', 'ema12', 'ema26', 'fft_3', 'fft_6', 'fft_9', 'fft_12', 'log', 'bollinger_upper', 'bollinger_lower',
            'Close_lag_15', 'Close_lag_30',
        ]
        self.target = target

        self.data_df = data_df
        self.window_size = window_size

        # target data
        self.target_df = data_df[[target]]
        self.target_df = self.target_df.rename(columns={self.target_df.columns[0]: 'target'})

        self.dataset_length = len(self.target_df) - self.window_size - 1

    def __getitem__(self, index):
        features_list = []

        for feature in self.features:
            window = self.data_df[feature].iloc[index:index + self.window_size].tolist()
            features_list.append(window)

        features = torch.tensor(features_list).float()
        features_t = features.transpose(1, 0)
        sample = {
            'target': self.target_df['target'].iloc[index + self.window_size],
            'features': features_t,
        }
        return sample

    def __len__(self):
        return self.dataset_length
