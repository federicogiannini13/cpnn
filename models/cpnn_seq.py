import numpy as np
import torch
from torch.utils.data import DataLoader

from models.clstm import cLSTMLinear
from models.cpnn_columns import cPNNColumns
from models.utils_seq import (
    get_accuracy_from_pred,
    get_kappa_temporal_from_pred,
    get_kappa_from_pred,
    loss_many_to_many,
)


class cPNNSeq:
    """
    Class that implements the list of single cPNN columns. The loss function is computed over each sequence.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        stride: int = 1,
        first_label_kappa: int = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        column_class: default: iLSTM.
            The class that implements the single column's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of single columns' Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        stride: int, default: 1.
            The length of the sliding window's stride.
        first_label_kappa: int, default: None.
            The label of the last sample before the start of the stream, it is used to compute the kappa_temporal.
            If None a random label is generated.
        kwargs:
            Parameters of column_class.
        """
        self.columns = cPNNColumns(column_class, device, lr, **kwargs)
        if first_label_kappa is None:
            self.first_label_kappa = np.random.randint(0, 2)
        else:
            self.first_label_kappa = first_label_kappa

    def test_then_train(
        self,
        x: np.array,
        y: list,
        epochs: int = 10,
        column_id: int = None,
        verbose: bool = True,
    ) -> (dict, dict):
        """
        It tests cPNN on a single batch, and then it performs the training.
        It computes the losses on single sequences, and then it averages them on the batch.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: list
            The target values of the batch.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.
        epochs: int, default: 10.
            Training epochs to perform on the batch.
        verbose: bool.
            True if you want to print the metrics of all epochs.

        Returns
        -------
        perf_test: dict
            The dictionary representing test's performance.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
        perf_train: dict
            The dictionary representing training's performance.
            For each metric the dict contains a list of epochs' values.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
        """
        batch = self.columns.convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False
        )
        for x, y in batch_loader:  # only to take x and y from loader
            break

        # TEST
        perf_test = {}
        with torch.no_grad():
            predictions = self.columns(x, column_id)
            perf_test["accuracy"] = get_accuracy_from_pred(predictions, y).item()
            perf_test["kappa_temporal"] = get_kappa_temporal_from_pred(
                predictions, y, self.first_label_kappa
            )
            perf_test["kappa"] = get_kappa_from_pred(predictions, y)
            perf_test["loss"] = loss_many_to_many(
                predictions, y, self.columns.criterion
            ).item()
            if verbose:
                print(
                    "Test accuracy: ",
                    perf_test["accuracy"],
                    ", Test loss: ",
                    perf_test["loss"],
                    sep="",
                )

        # TRAIN
        perf_train = {
            "accuracies": [],
            "losses": [],
            "kappas": [],
            "kappas_temporal": [],
        }
        for e in range(1, epochs + 1):
            predictions = self.columns(x)
            loss = loss_many_to_many(predictions, y, self.columns.criterion)
            self.columns.optimizers[-1].zero_grad()
            loss.backward()
            self.columns.optimizers[-1].step()
            perf_train["losses"].append(loss.item())
            pred = self.columns(x)
            perf_train["accuracies"].append(get_accuracy_from_pred(pred, y).item())
            perf_train["kappas"].append(get_kappa_from_pred(pred, y).item())
            perf_train["kappas_temporal"].append(
                get_kappa_temporal_from_pred(pred, y, self.first_label_kappa).item()
            )
            if verbose:
                print(
                    "Training epoch ",
                    e,
                    "/",
                    epochs,
                    ". accuracy: ",
                    perf_train["acccuracies"][-1],
                    ", loss:",
                    perf_train["losses"][-1],
                    sep="",
                    end="\r",
                )
        if verbose:
            print()
            print()
        self.first_label_kappa = y[-1, -1]
        return perf_test, perf_train

    def add_new_column(self):
        """
        It adds a new column to the cPNN architecture, after a concept drift.
        """
        self.columns.add_new_column()
