# models/lstm.py

import torch.nn as nn


class BaselineLSTM(nn.Module):
    """Many-to-many-vector LSTM: the final hidden state after reading the
    `input_window_size`-day input window is mapped through one linear layer
    straight to all `output_window_size` future CIR values at once (a
    "direct multi-output" strategy -- no separate model per horizon day, no
    per-step teacher forcing during training).

    input: (batch, input_window_size, n_features) -> output: (batch, output_window_size)

    A ReLU is applied across the FULL LSTM output sequence (not just the
    final time step) before dropout and before slicing out the last time
    step.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, window_size, hidden_size)
        out = self.relu(out)
        out = self.dropout(out)
        last_step = out[:, -1, :]      # final time step's hidden output: (batch, hidden_size)
        return self.fc(last_step)
