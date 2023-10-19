import torch
from models.clstm import cLSTMLinear


class cPNNColumns(torch.nn.Module):
    """
    Class that implements the list of single cPNN columns.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr=0.01,
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
        kwargs:
            Parameters of column_class.
        """
        super(cPNNColumns, self).__init__()
        kwargs["device"] = (
            torch.device("cpu") if device is None else torch.device(device)
        )
        self.device = kwargs["device"]
        self.column_class = column_class
        self.column_args = kwargs
        self.lr = lr

        self.columns = torch.nn.ModuleList([column_class(**kwargs)])
        self.column_args["input_size"] = (
            self.columns[0].input_size + self.columns[0].hidden_size
        )
        self.optimizers = [self._create_optimizer()]
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _create_optimizer(self, column_id=-1):
        return torch.optim.Adam(self.columns[column_id].parameters(), lr=self.lr)

    def forward(self, x, column_id=None, train=False):
        if column_id is None:
            column_id = len(self.columns) - 1
        out = None
        prev_h = None
        for i in range(0, column_id + 1):
            out, prev_h = self.columns[i](x, prev_h, train=train)
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
        self.columns.append(self.column_class(**self.column_args))
        self.optimizers.append(self._create_optimizer())
