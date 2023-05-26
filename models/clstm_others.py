import torch
from torch import nn

from models.clstm import cLSTMLinear


class cLSTMLinearCombination(cLSTMLinear):
    def __init__(
        self,
        input_size,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
    ):
        super(cLSTMLinearCombination, self).__init__(
            input_size, device, hidden_size, output_size, batch_size
        )
        self.linear_combination = nn.Linear(hidden_size, hidden_size)
        self.linear_combination.to(self.device)

    def forward(self, x, prev_h):
        outputs = []
        for inp in prev_h:
            outputs.append(self.linear_combination(inp))
        output = outputs[0]
        for o in outputs[1:]:
            output = torch.add(output, o)
        return cLSTMLinear.forward(self, x, output)


class cLSTMLinearCombinationDifferentWeights(cLSTMLinear):
    def __init__(
        self,
        input_size,
        n_columns,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
    ):
        super(cLSTMLinearCombinationDifferentWeights, self).__init__(
            input_size, device, hidden_size, output_size, batch_size
        )
        self.linear_combination = [
            nn.Linear(hidden_size, hidden_size) for i in range(0, n_columns)
        ]
        for l in self.linear_combination:
            l.to(self.device)

    def forward(self, x, prev_h):
        outputs = []
        for i, inp in enumerate(prev_h):
            outputs.append(self.linear_combination[i](inp))
        output = outputs[0]
        for o in outputs[1:]:
            output = torch.add(output, o)
        return cLSTMLinear.forward(self, x, output)


class cLSTMDouble(nn.Module):
    def __init__(
        self,
        input_size,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
    ):
        super(cLSTMDouble, self).__init__()

        # PARAMETERS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        # LAYERS
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm1.to(self.device)
        self.lstm2 = nn.LSTM(hidden_size, output_size, num_layers=1, batch_first=True)
        self.lstm2.to(self.device)

    def forward(self, x, prev_h):
        input_f = x

        if prev_h is not None:
            input_f = torch.cat((x, prev_h), dim=2)  # (B, L, I+H)

        h0_1 = torch.zeros(
            1, x.size()[0], self.hidden_size, device=self.device, requires_grad=True
        )
        c0_1 = torch.zeros(
            1, x.size()[0], self.hidden_size, device=self.device, requires_grad=True
        )
        out_h1, _ = self.lstm1(input_f, (h0_1, c0_1))

        h0_2 = torch.zeros(
            1, x.size()[0], self.output_size, device=self.device, requires_grad=True
        )
        c0_2 = torch.zeros(
            1, x.size()[0], self.output_size, device=self.device, requires_grad=True
        )
        out_h2, _ = self.lstm2(out_h1, (h0_2, c0_2))

        return out_h2, out_h1
