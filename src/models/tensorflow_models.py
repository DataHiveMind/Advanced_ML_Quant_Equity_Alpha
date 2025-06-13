from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LSTM,
    Conv1D,
    Flatten,
    GRU,
    Bidirectional,
)


def build_dnn_model(input_shape, config):
    """
    Build a simple feedforward DNN model.
    :param input_shape: tuple, shape of input features (excluding batch dimension)
    :param config: dict, configuration with keys like 'layers', 'activation', 'dropout'
    :return: Keras Model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for units in config.get("layers", [64, 32]):
        model.add(Dense(units, activation=config.get("activation", "relu")))
        if config.get("batch_norm", False):
            model.add(BatchNormalization())
        if config.get("dropout", 0) > 0:
            model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation=config.get("output_activation", "linear")))
    return model


def build_lstm_model(input_shape, config):
    """
    Build an LSTM model for sequence data.
    :param input_shape: tuple, shape of input (timesteps, features)
    :param config: dict, configuration with keys like 'lstm_units', 'dropout'
    :return: Keras Model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(config.get("lstm_units", 64), return_sequences=False))
    if config.get("dropout", 0) > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation=config.get("output_activation", "linear")))
    return model


def build_cnn_model(input_shape, config):
    """
    1D CNN for time series or sequence data.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(
        Conv1D(
            filters=config.get("filters", 32),
            kernel_size=config.get("kernel_size", 3),
            activation=config.get("activation", "relu"),
        )
    )
    if config.get("batch_norm", False):
        model.add(BatchNormalization())
    if config.get("dropout", 0) > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Flatten())
    model.add(Dense(1, activation=config.get("output_activation", "linear")))
    return model


def build_gru_model(input_shape, config):
    """
    GRU-based model for sequence data.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(config.get("gru_units", 64), return_sequences=False))
    if config.get("dropout", 0) > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation=config.get("output_activation", "linear")))
    return model


def build_bidirectional_lstm_model(input_shape, config):
    """
    Bidirectional LSTM for sequence data.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(config.get("lstm_units", 64), return_sequences=False)))
    if config.get("dropout", 0) > 0:
        model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation=config.get("output_activation", "linear")))
    return model
