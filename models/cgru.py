import torch
from torch import nn
import numpy as np


class cGRULinear(nn.Module):
    def __init__(
        self,
        input_size=2,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=50,
    ):
        super(cGRULinear, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.h0 = np.zeros((1, self.hidden_size))

        # LAYERS
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.gru.to(self.device)

        self.linear = nn.Linear(hidden_size, output_size)
        self.linear.to(self.device)

    def forward(self, x, prev_h, train=False):
        input_f = x.to(self.device)

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)
        out_h, _ = self.gru(input_f, self._build_initial_state(x))
        out = self.linear(out_h)

        return out, out_h

    def _build_initial_state(self, x):
        s = torch.from_numpy(np.tile(self.h0, (1, x.size()[0], 1))).float()
        s.requires_grad = True
        return s.to(self.device)
