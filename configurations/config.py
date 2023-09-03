from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
# Stock symbol in market
_C.data.symbol = 'GOOGL'
# Size of test group
_C.data.test_size = 235
# Size of validation group
_C.data.val_size = 235
# Window size for dataset sequence length
_C.data.window_size = 17
# Batch size for dataloader
_C.data.batch_size = 16
# Whether to update data to today's market values (True) or not (False)
_C.data.updates = False

_C.model = CN()
# LSTM bidirectional training
_C.model.bidirectional = False
# LSTM hidden dim - how many neurons in each layer
_C.model.hidden_dim = 64
# LSTM dim - how many lstm layers total
_C.model.layer_dim = 2
# LSTM dropout rate
_C.model.dropout = 0

_C.optimizer = CN()
# Optimizer learning rate
_C.optimizer.learning_rate = 1e-3
# Optimizer weight decay
_C.optimizer.weight_decay = 1e-6

_C.train = CN()
# Loss function name
_C.train.loss_fn = 'rmse'
# Number of epochs in training loop
_C.train.epochs = 150


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return _C.clone()
