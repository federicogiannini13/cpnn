import numpy as np
import torch
from torch.utils import data as data_utils
from torch.utils.data import DataLoader

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
        seq_len=5,
        stride=1,
        loss_on_seq=False,
        combination=False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        column_class: default: iLSTM
            The class that implements the single column's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of columns' Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        stride: int, default: 1.
            The length of sliding window's stride.
        loss_on_seq: bool, default: False.
            If True the loss function is computed on each sequence and then averaged on the batch.
            If False each sample's predictions are averaged to obtain a single value.
        combination: bool, default: False.
            If True each iGIM column combines all previous columns.
            If False each iGIM column takes only last column.
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
        self.seq_len = seq_len
        self.stride = stride
        self.combination = combination

        self.columns = torch.nn.ModuleList([column_class(**kwargs)])
        self.column_args["input_size"] = self.columns[0].input_size + self.columns[0].hidden_size
        self.optimizers = [self._create_optimizer()]
        if not loss_on_seq:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _cut_in_sequences(self, x, y):
        seqs_features = []
        seqs_targets = []
        for i in range(0, len(x), self.stride):
            if len(x) - i >= self.seq_len:
                seqs_features.append(x[i : i + self.seq_len, :].astype(np.float32))
                if y is not None:
                    seqs_targets.append(
                        np.asarray(y[i : i + self.seq_len], dtype=np.int_)
                    )
        return np.asarray(seqs_features), np.asarray(seqs_targets)

    def cut_in_sequences_tensors(self, x, y):
        seqs_features = []
        seqs_targets = []
        for i in range(0, x.size()[0], self.stride):
            if x.size()[0] - i >= self.seq_len:
                seqs_features.append(
                    x[i : i + self.seq_len, :].view(1, self.seq_len, x.size()[1])
                )
                seqs_targets.append(y[i : i + self.seq_len].view(1, self.seq_len))
        seq_features = torch.cat(seqs_features, dim=0)
        seqs_targets = torch.cat(seqs_targets, dim=0)
        return seq_features, seqs_targets

    def load_batch(self, x: np.array, y: np.array = None):
        """
        It transforms the batch in order to be inputted to cPNN, by building the different sequences and
        converting them to tensors.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None.
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        x: torch.Tensor
            The features values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len, n_features)
        y: torch.Tensor
            The target values of the samples in the batc. It has length: batch_size. If y is None it returns None.
        y_seq: torch.Tensor
            The target values of the created sequences. It has shape: (batch_size - seq_len + 1, seq_len). If y is None it returns None.
        """
        batch = self.convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False
        )
        for x, y_seq in batch_loader:  # only to take x and y from loader
            break
        y = torch.tensor(y)
        return x, y, y_seq

    def convert_to_tensor_dataset(self, x, y=None):
        """
        It converts the dataset in order to be inputted to cPNN, by building the different sequences and
        converting them to TensorDataset.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list, default: None
            The target values of the batch. If None only features will be loaded.
        Returns
        -------
        dataset: torch.data_utils.TensorDataset
            The tensor dataset representing the different sequences.
            The features values have shape: (batch_size - seq_len + 1, seq_len, n_features)
            The target values have shape: (batch_size - seq_len + 1, seq_len)
        """
        x, y = self._cut_in_sequences(x, y)
        x = torch.tensor(x)
        if len(y) > 0:
            y = torch.tensor(y).type(torch.LongTensor)
            return data_utils.TensorDataset(x, y)
        return x

    def _create_optimizer(self, column_id=-1):
        return torch.optim.Adam(self.columns[column_id].parameters(), lr=self.lr)

    def forward(self, x, column_id=None, return_initial_states=False):
        if column_id is None:
            column_id = len(self.columns) - 1

        if len(self.columns) <= 2 or not self.combination:
            prev_h = None
            for i in range(0, column_id + 1):
                out, prev_h, initial_states = self.columns[i](x, prev_h)
        else:
            prev_h = None
            prev_h_list = []
            for i in range(0, 2):
                out, prev_h, initial_states = self.columns[i](x, prev_h)
                prev_h_list.append(prev_h)
            for i in range(2, len(self.columns)):
                out, prev_h, initial_states = self.columns[i](x, prev_h_list)
                prev_h_list.append(prev_h)
        if not return_initial_states:
            return out
        return out, initial_states

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
        if len(self.columns) < 2 or not self.combination:
            self.columns.append(self.column_class(**self.column_args))
        else:
            self.column_args["n_columns"] = len(self.columns)
            if self.column_class.__name__.startswith("iLSTM"):
                # TODO differentWeights
                self.columns.append(
                    cLSTMLinearCombinationDifferentWeights(**self.column_args)
                )
            elif self.column_class.__name__.startswith("iGRU"):
                self.columns.append(cGRULinearCombination(**self.column_args))
        self.optimizers.append(self._create_optimizer())

    def update_initial_states(self, initial_states):
        self.columns[-1].update_initial_states(initial_states)
