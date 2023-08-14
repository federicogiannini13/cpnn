import torch

from models.clstm_others import cLSTMDouble
from models.cpnn import cPNN
from models.cpnn_columns import cPNNColumns


class cPNNExp(cPNN):
    def __init__(
        self,
        column_class=cLSTMDouble,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        stride: int = 1,
        first_label_kappa: int = None,
        train_epochs: int = 10,
        train_verbose: bool = False,
        concepts_boundaries: list = None,
        **kwargs,
    ):
        super(cPNNExp, self).__init__(column_class, device, lr, seq_len, stride, first_label_kappa, train_epochs,
                                      train_verbose, concepts_boundaries, **kwargs)
        self.columns = cPNNColumnsExp(
            column_class, device, lr, seq_len, stride, False, **kwargs
        )


class cPNNColumnsExp(cPNNColumns):
    def forward(self, x, column_id=None):
        if column_id is None:
            column_id = len(self.columns) - 1

        prev_h = None
        for i in range(0, column_id):
            out, prev_h = self.columns[i](x, prev_h)

        if len(self.columns) > 1:
            out, _ = self.columns[-1](torch.zeros(x.size()), prev_h)
        else:
            out, _ = self.columns[-1](x, prev_h)

        return out
