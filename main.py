from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score

from yahoofinance.yfDataExtraction import YfDataExtractor
from yahoofinance.finDataProcessor import FinDataProcessor
from configurations.configuration import get_config
from ml.findatasets import StockDataset
from ml.loss import get_loss_fn
from ml.models import LSTMModel
from ml.training import Session
from ml.util import split_data

SAVE_MODEL = True
LOAD_MODEL = True


if __name__ == '__main__':
    cfg = get_config()
    symbol = cfg.data.symbol

    if cfg.data.updates:
        # Update data from API
        data_extractor = YfDataExtractor(symbol=symbol)

    # Process data
    data_processor = FinDataProcessor(symbol=symbol)
    data_df = data_processor.data_df

    train_data, val_data, test_data = split_data(data_df, cfg.data.val_size, cfg.data.test_size)

    # Scale data
    scaler = MinMaxScaler()
    scale_features = [feature for feature in train_data.columns]
    train_data.loc[:, scale_features] = scaler.fit_transform(train_data[scale_features])
    val_data.loc[:, scale_features] = scaler.transform(val_data[scale_features])
    test_data.loc[:, scale_features] = scaler.transform(test_data[scale_features])

    # Datasets and Dataloaders setup
    window_size = cfg.data.window_size

    train_ds = StockDataset(train_data, window_size=window_size)
    val_ds = StockDataset(val_data, window_size=window_size)
    test_ds = StockDataset(test_data, window_size=window_size)

    batch_size = cfg.data.batch_size
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)

    # Model config
    bidirectional = cfg.model.bidirectional
    input_dim = len(train_ds.features)
    output_dim = 1
    hidden_dim = cfg.model.hidden_dim
    layer_dim = cfg.model.layer_dim
    dropout = cfg.model.dropout
    n_epochs = cfg.train.epochs
    learning_rate = cfg.optimizer.learning_rate
    weight_decay = cfg.optimizer.weight_decay

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, dropout_prob=dropout, bidirectional=bidirectional)
    loss_fn = get_loss_fn(cfg.train.loss_fn)
    optimizer_close = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    session = Session(model=model, loss_fn=loss_fn, optimizer=optimizer_close, epochs=n_epochs)

    # Train model
    session.train(train_dl, val_dl, save=SAVE_MODEL)
    session.plot_losses()

    # Predict on test
    normalized_predictions, true_vals = session.predict(test_dl)

    # Inverse scaled predicted results
    inverse_scaler = MinMaxScaler()
    inverse_scaler.min_, inverse_scaler.scale_ = scaler.min_[train_data.columns.get_loc(train_ds.target)], scaler.scale_[train_data.columns.get_loc(train_ds.target)]
    inverse_predictions = inverse_scaler.inverse_transform(normalized_predictions)
    inverse_true_vals = inverse_scaler.inverse_transform(true_vals)

    # Plot results
    session.plot_pred(inverse_predictions, inverse_true_vals)
    session.plot_validation_score()

    test_scores = {
        "mae": mae(inverse_true_vals, inverse_predictions),
        "mse": mse(inverse_true_vals, inverse_predictions),
        "mape": mape(inverse_true_vals, inverse_predictions),
        "r2": r2_score(inverse_true_vals, inverse_predictions)
    }

    print("##### Scores:\n", test_scores)
