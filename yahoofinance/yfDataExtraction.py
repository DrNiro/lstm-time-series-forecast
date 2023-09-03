import pandas as pd
import yfinance as yf
import os

from ml.util import remove_columns

pd.set_option("display.max_rows", 99)
pd.set_option("display.max_columns", 500)


class YfDataExtractor:
    """
    A Class used to load company details by symbol from yahoo-finance API.

    Args:
        symbol: Stock symbol of a company.
    """
    def __init__(self, symbol: str, update_file: bool = True):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.history = self.ticker.history(period="max")
        self.history_df = pd.DataFrame(self.history)

        # generate raw time values
        self.history_df = (
            self.history_df
            .assign(day=self.history_df.index.day)
            .assign(month=self.history_df.index.month)
            .assign(day_of_week=self.history_df.index.dayofweek)
            .assign(day_of_year=self.history_df.index.dayofyear)
            .assign(week_of_year=self.history_df.index.isocalendar().week)
        )

        # remove unwanted columns
        self.history_df = remove_columns(self.history_df, ["Dividends", "Stock Splits"])

        if update_file:
            self.save_to_csv(override_file=True)

    def save_to_csv(self, override_file=False):
        file_name = f'{self.symbol}_raw_data.csv'
        data_dir = os.path.join('data', self.symbol)
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, file_name)
        if not os.path.isfile(path):
            self.history_df.to_csv(path)
            print(f'File successfully saved to {path}.')
        elif override_file:
            self.history_df.to_csv(path, mode='w+')
            print(f'File already exists and successfully replaced at {path}.')
        else:
            print(f'Could not save {path}, check if file already exists, or set override_file to True.')
