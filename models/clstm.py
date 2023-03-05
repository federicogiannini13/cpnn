import torch
from torch import nn
import numpy as np


class cLSTMLinear(nn.Module):
    def __init__(
        self,
        input_size,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
    ):
        super(cLSTMLinear, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.h0 = np.zeros((1, self.hidden_size))
        self.c0 = np.zeros((1, self.hidden_size))

        # LAYERS
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm.to(self.device)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.to(self.device)

    def forward(self, x, prev_h):
        input_f = x.to(self.device)

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)

        out_h, initial_states = self.lstm(
            input_f,
            (
                self._build_initial_state(x, self.h0),
                self._build_initial_state(x, self.c0),
            ),
        )

        out = self.linear(out_h)

        return out, out_h, initial_states

    def _build_initial_state(self, x, state):
        s = torch.from_numpy(np.tile(state, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)

    def update_initial_states(self, initial_states):
        self.h0 = initial_states[0][0, -1:, :].detach().numpy()
        self.c0 = initial_states[1][0, -1:, :].detach().numpy()