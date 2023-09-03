import numpy as np


def generate_cyclical_features(df, col_name, period, start_num=0, drop_col=False):
    kwargs = {
        f'cyclic_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period) * np.cos(2*np.pi*(df[col_name]-start_num)/period),
        f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }

    df = df.assign(**kwargs)
    if drop_col:
        return df.drop(columns=[col_name])
    return df


def generate_lags(df, num_lags: int or [int], col: str):
    df_n = df.copy()
    if isinstance(num_lags, int):
        _iterator = range(1, num_lags + 1)
    else:
        _iterator = num_lags

    for n in _iterator:
        df_n[f"{col}_lag_{n}"] = df_n[col].shift(n)
    return df_n


# pandas util functions
def remove_columns(df, drop_col_list):
    return df.drop(drop_col_list, axis=1)


# split to train, val, test groups
def split_data(data, val_size, test_size):
    train = data[:-test_size]
    test = data[-test_size:]
    val = train[-val_size:]
    train = train[:-val_size]
    return train, val, test