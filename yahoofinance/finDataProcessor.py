import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ml.util import generate_cyclical_features, generate_lags

date_cfg = {
    'day': (31, 0),
    'month': (12, 0),
    'day_of_week': (7, 0),
    'day_of_year': (365, 0),
    'week_of_year': (54, 0)
}


class FinDataProcessor:
    """
    A class used to preprocess raw data extracted from Yahoo-Finance API, transforming it and extracting features.
    """

    def __init__(self, symbol):
        raw_data_path = f'data/{symbol}/{symbol}_raw_data.csv'
        # load raw data
        self.data_df = pd.read_csv(raw_data_path)

        # generate finance features
        self.generate_features()

        self.data_df = self.data_df.drop(['Date'], axis=1)

    def generate_features(self):
        self.cyclic_dates()
        self.fourier_transformation()
        self.indicators()
        self.rsi()
        self.lags()
        # self.trend()

        # remove rows with NaN values created by rolling windows and shifts
        self.data_df.dropna(how='any', inplace=True)

    def lags(self):
        self.data_df = generate_lags(self.data_df, num_lags=[15, 30], col='Close')

    def cyclic_dates(self):
        for date_kind, (cycle, start_val) in date_cfg.items():
            self.data_df = generate_cyclical_features(self.data_df, date_kind, cycle, start_val, drop_col=True)

    def indicators(self):
        self.data_df['log'] = np.log(self.data_df['Close'])
        self.data_df['diff_1'] = self.data_df['Close'].diff(1)
        self.data_df['typical_price'] = (self.data_df['Close'] + self.data_df['High'] + self.data_df['Low']) / 3
        self.data_df['ma7'] = self.data_df['Close'].rolling(window=7).mean()
        self.data_df['ma21'] = self.data_df['Close'].rolling(window=21).mean()
        self.data_df['ema12'] = self.data_df['Close'].ewm(span=12).mean()
        self.data_df['ema26'] = self.data_df['Close'].ewm(span=26).mean()
        self.data_df['MACD'] = (self.data_df['ema12'] - self.data_df['ema26'])

        # Create Bollinger Bands
        bollinger_window = 20
        num_std = 2
        # rolling_mean = self.data_df['Close'].rolling(bollinger_window).mean()
        self.data_df['sd20'] = self.data_df['typical_price'].rolling(bollinger_window).std()
        # self.data_df['20sd'] = pd.stats.moments.rolling_std(self.data_df['Close'], 20)
        self.data_df['tp_ma20'] = self.data_df['typical_price'].rolling(window=bollinger_window).mean()
        self.data_df['bollinger_upper'] = self.data_df['tp_ma20'] + (self.data_df['sd20'] * num_std)
        self.data_df['bollinger_lower'] = self.data_df['tp_ma20'] - (self.data_df['sd20'] * num_std)

        # Exponential Moving Average
        self.data_df['ema'] = self.data_df['Close'].ewm(com=0.5).mean()

    def fourier_transformation(self):
        # mirror it (make fft periodic)
        N = len(self.data_df['Close'])
        signal_mirror = np.append(self.data_df['Close'], np.flip(self.data_df['Close']))

        # low pass filtering
        for n in [3, 6, 9, 12]:
            fft_list = np.fft.rfft(signal_mirror)
            fft_list[n:-n] = 0
            signal_ifft = np.fft.irfft(np.asarray(fft_list))
            fft = signal_ifft[:N]
            fft_df = pd.DataFrame({f'fft_{n}': fft})
            self.data_df = pd.concat([self.data_df, fft_df], axis=1)

    def rsi(self, window_length=14):
        rsi_df = pd.DataFrame()
        # difference between current and previous value
        rsi_df['diff_1'] = self.data_df['Close'].diff(1)
        # gain and loss calc.
        # negative values count as loss taking the abs value with zero gain,
        rsi_df['gain'] = rsi_df['diff_1'].clip(lower=0)
        # positive values count as gain and zero loss.
        rsi_df['loss'] = rsi_df['diff_1'].clip(upper=0).abs()

        # RS first step - calc gain and loss average window
        rsi_df['avg_gain'] = rsi_df['gain'].rolling(window=window_length).mean()[:window_length+1]
        rsi_df['avg_loss'] = rsi_df['loss'].rolling(window=window_length).mean()[:window_length+1]

        # RS second step - calc following windows based on previous and the current averages
        # Gain averages
        for i, row in enumerate(rsi_df['avg_gain'].iloc[window_length + 1:]):
            prev_avg_gain = rsi_df['avg_gain'].iloc[i + window_length]
            curr_gain = rsi_df['gain'].iloc[i + window_length + 1]
            rsi_df['avg_gain'].iloc[i + window_length + 1] = (prev_avg_gain * (window_length - 1) + curr_gain) / window_length
        # Loss averages
        for i, row in enumerate(rsi_df['avg_loss'].iloc[window_length + 1:]):
            prev_avg_loss = rsi_df['avg_loss'].iloc[i + window_length]
            current_loss = rsi_df['loss'].iloc[i + window_length + 1]
            rsi_df['avg_loss'].iloc[i + window_length + 1] = (prev_avg_loss * (window_length - 1) + current_loss) / window_length

        # RSI
        rsi_df['rs'] = rsi_df['avg_gain'] / rsi_df['avg_loss']
        rsi_df['rsi'] = 100 - (100 / (1.0 + rsi_df['rs']))

        self.data_df['rsi'] = rsi_df['rsi']

    def trend(self, period=90):
        active_ratio = 5 / 7  # Stock market active 5/7 days a week
        period = period * active_ratio // 1
        decomposed_seasonal = sm.tsa.seasonal_decompose(self.data_df["Close"], period=period, extrapolate_trend=1)
        self.data_df['trend'] = decomposed_seasonal.trend
        self.data_df['residual_trend'] = decomposed_seasonal.resid

    def plot_fft(self):
        for n in [3, 6, 9, 12]:
            plt.plot(self.data_df[f'fft_{n}'], label=f'fft_{n}')
        plt.plot(self.data_df['Close'])
        plt.legend()
        plt.show()

    def plot_cyclic(self):
        for time_kind, _ in date_cfg.items():
            plt.plot(self.data_df[f'cyclic_{time_kind}'].iloc[-30:], label=f'cyclic {time_kind}')
            plt.plot(self.data_df[f'sin_{time_kind}'].iloc[-30:], label=f'sin {time_kind}')
            plt.plot(self.data_df[f'cos_{time_kind}'].iloc[-30:], label=f'cos {time_kind}')
            plt.legend()
            plt.show()

    def plot_indicators(self, indicators=()):
        if len(indicators) > 0:
            for indi in indicators:
                plt.plot(self.data_df[indi], label=indi)
            plt.legend()
            plt.show()

    def save_to_csv(self, symbol, override_file=False):
        file_name = f'{symbol}_data.csv'
        data_dir = os.path.join('data', symbol)
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, file_name)
        if not os.path.isfile(path):
            self.data_df.to_csv(path)
            print(f'File successfully saved to {path}.')
        elif override_file:
            self.data_df.to_csv(path, mode='w+')
            print(f'File already exists and successfully replaced at {path}.')
        else:
            print(f'Could not save {path}, check if file already exists, or set override_file to True.')
