import torch
from torch import nn

from models.cgru import cGRULinear


class cGRULinearCombination(cGRULinear):
    def __init__(
        self,
        input_size,
        device=torch.device("cpu"),
        hidden_size=128,
        output_size=2,
        batch_size=128,
    ):
        super(cGRULinearCombination, self).__init__(
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
        return cGRULinear.forward(self, x, output)
