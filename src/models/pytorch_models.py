import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    """
    A customizable LSTM model for regression or classification tasks.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        output_size=1,
        bidirectional=False,
    ):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CustomDNN(nn.Module):
    """
    A customizable feedforward neural network (DNN).
    """

    def __init__(self, input_size, hidden_layers=[64, 32], dropout=0.0, output_size=1):
        super(CustomDNN, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
