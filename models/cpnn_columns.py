import numpy as np
import torch
from torch import nn
import torch.quantization

from models.cgru_others import cGRULinearCombination
from models.clstm import cLSTMLinear
from models.clstm_others import cLSTMDouble, cLSTMLinearCombinationDifferentWeights


class cPNNColumns(torch.nn.Module):
    """
    Class that implements the list of single cPNN columns.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr=0.01,
        combination=False,
        remember_states=False,
        many_to_one=False,
        quantize=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        column_class: default: cLSTMLinear
            The class that implements the single column's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of columns' Adam Optimizer.
        combination: bool, default: False.
            If True each column combines all previous columns.
            If False each column considers only the last column.
        remember_states: bool, default: False
            If True the initial h0 is set as h1 of the previous sequence.
        many_to_one: bool, default: False
            If True each column is a many to one model.
        quantize: bool, default: False
            If True, after a concept drift, the column is quantized.
        kwargs:
            Parameters of column_class.
        """
        super(cPNNColumns, self).__init__()
        kwargs["device"] = (
            torch.device("cpu") if device is None else torch.device(device)
        )
        kwargs["remember_states"] = remember_states
        kwargs["many_to_one"] = many_to_one
        self.device = kwargs["device"]
        self.column_class = column_class
        self.column_args = kwargs
        self.lr = lr
        self.combination = combination
        self.quantize = quantize

        self.columns = torch.nn.ModuleList([column_class(**kwargs)])
        self.column_args["input_size"] = self.columns[0].input_size + self.columns[0].hidden_size
        self.optimizers = [self._create_optimizer()]
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _create_optimizer(self, column_id=-1):
        return torch.optim.Adam(self.columns[column_id].parameters(), lr=self.lr)

    def forward(self, x, column_id=None, train=False):
        if column_id is None:
            column_id = len(self.columns) - 1
        out = None
        if len(self.columns) <= 2 or not self.combination:
            prev_h = None
            for i in range(0, column_id + 1):
                out, prev_h = self.columns[i](x, prev_h, train=train)
        else:
            prev_h = None
            prev_h_list = []
            for i in range(0, 2):
                out, prev_h = self.columns[i](x, prev_h)
                prev_h_list.append(prev_h)
            for i in range(2, len(self.columns)):
                out, prev_h = self.columns[i](x, prev_h_list, train=train)
                prev_h_list.append(prev_h)
        return out

    def get_hidden(self, x, column_id=None):
        x = self.convert_to_tensor_dataset(x).to(self.device)

        if len(self.columns) == 1:
            return None

        if column_id is None:
            column_id = len(self.columns) - 2

        out_h = None
        for i in range(0, column_id + 1):
            _, out_h = self.columns[i](x, out_h)

        return out_h.detach().numpy()

    def add_new_column(self):
        """
        It adds a new column to the cPNN architecture, after a concept drift.
        Weights of previous columns are frozen.
        It also adds a new optimizer.
        """
        for param in self.columns[-1].parameters():
            param.requires_grad = False
        if self.quantize:
            last_column = self.columns[-1]
            self.columns = self.columns[:-1]
            conf = {}
            if "lstm" in self.column_class.__name__.lower():
                conf = {nn.LSTM, nn.Linear}
            if "gru" in self.column_class.__name__.lower():
                conf = {nn.GRU, nn.Linear}
            self.columns.append(torch.quantization.quantize_dynamic(
                last_column, conf, dtype=torch.qint8
            ))
        if len(self.columns) < 2 or not self.combination:
            self.columns.append(self.column_class(**self.column_args))
        else:
            self.column_args["n_columns"] = len(self.columns)
            if "lstm" in self.column_class.__name__.lower():
                # TODO differentWeights
                self.columns.append(
                    cLSTMLinearCombinationDifferentWeights(**self.column_args)
                )
            elif "gru" in self.column_class.__name__.lower():
                self.columns.append(cGRULinearCombination(**self.column_args))
        self.optimizers.append(self._create_optimizer())
