import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

from models.cpnn_columns import cPNNColumns
from models.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    kappa_temporal,
    get_samples_outputs,
    get_pred_from_outputs,
    kappa_temporal_score,
)
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from models.clstm import (
    cLSTMLinear,
)


class cPNN:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        column_class=cLSTMLinear,
        device=None,
        lr: float = 0.01,
        seq_len: int = 5,
        stride: int = 1,
        first_label_kappa: int = None,
        train_epochs: int = 10,
        train_verbose: bool = False,
        initial_task_id: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        column_class: default: cLSTMLinear.
            The class that implements the column.
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
        train_epochs: int, default: 10.
            The training epochs to perform in learn_many method.
        train_verbose: bool, default:False.
            True if, during the learn_many execution, you want to print the metrics after each training epoch.
        initial_task_id: int, default: 1.
            The id of the first task.
        kwargs:
            Parameters of column_class.
        """
        self.columns_args = kwargs
        self.columns_args["column_class"] = column_class
        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns = cPNNColumns(**self.columns_args)
        self.seq_len = seq_len
        self.stride = stride
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.samples_cont = 0
        self.columns_perf = [{"kappa": 0.0, "cont": 0}]
        self.task_ids = [initial_task_id]

        if first_label_kappa is not None:
            self.first_label_kappa = torch.tensor([first_label_kappa]).view(1)
        else:
            self.first_label_kappa = torch.randint(0, 2, (1,)).view(1)

    def get_seq_len(self):
        return self.seq_len

    def set_initial_task(self, task):
        self.task_ids = [task]

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

    def _cut_in_sequences_tensors(self, x, y):
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

    def _convert_to_tensor_dataset(self, x, y=None):
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

    def _load_batch(self, x: np.array, y: np.array = None):
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
        batch = self._convert_to_tensor_dataset(x, y)
        batch_loader = DataLoader(
            batch, batch_size=batch.tensors[0].size()[0], drop_last=False
        )
        y_seq = None
        for x, y_seq in batch_loader:  # only to take x and y from loader
            break
        y = torch.tensor(y)
        return x, y, y_seq

    def add_new_column(self, task_id: int = None):
        """
        It adds a new column to the cPNN architecture, after a concept drift.

        Parameters
        ----------
        task_id: int, default: None
            The id of the new task. If None it increments the last one.
        """
        self.columns.add_new_column()
        self.columns_perf.append({"kappa": 0.0, "cont": 0})
        if task_id is None:
            self.task_ids.append(self.task_ids[-1] + 1)
        else:
            self.task_ids.append(task_id)

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains cPNN on a single batch.
        It stores in columns_perf the prequential evaluation performance (using the kappa score).
        It computes the loss after averaging each sample's predictions.
        *ONLY FOR BATCH LEARNER*

        Parameters
        ----------
        x: numpy.array or list
            The features values of the batch.
        y: np.array or list
            The target values of the batch.

        Returns
        -------
        perf_train: dict
            The dictionary representing training's performance. Each key contains the list representing all the epochs' performances.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
            For each metric the dict contains a list of epochs' values.
        """
        # update column_perf
        pred = self.predict_many(x)
        self.columns_perf[-1]["kappa"] = (
            self.columns_perf[-1]["kappa"] * self.columns_perf[-1]["cont"]
            + cohen_kappa_score(pred, y)
        ) / (self.columns_perf[-1]["cont"] + 1)
        self.columns_perf[-1]["cont"] += 1

        x = np.array(x)
        y = list(y)
        first_batch = False
        x, y, y_seq = self._load_batch(x, y)
        if first_batch:
            y = y[self.seq_len - 1 :]

        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }
        for e in range(1, self.train_epochs + 1):
            perf_epoch = self._fit(x, y)
            if self.train_verbose:
                print(
                    "Training epoch ",
                    e,
                    "/",
                    self.train_epochs,
                    ". accuracy: ",
                    perf_epoch["accuracies"],
                    ", loss:",
                    perf_epoch["losses"],
                    sep="",
                    end="\r",
                )
            for k in perf_epoch:
                perf_train[k].append(perf_epoch[k])
        if self.train_verbose:
            print()
            print()
        self.samples_cont += x.size()[0]

        return perf_train

    def predict_many(self, x: np.array, column_id: int = None):
        """
        It performs prediction on a single batch.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the batch.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.

        Returns
        -------
        predictions: numpy.array
            The 1D numpy array (with length batch_size) containing predictions of all samples.
        """
        x = np.array(x)
        if x.shape[0] < self.get_seq_len():
            return np.array([None] * x.shape[0])
        first_train = False
        x = self._convert_to_tensor_dataset(x).to(self.columns.device)
        with torch.no_grad():
            outputs = self.columns(x, column_id)
            outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate(
                    [np.array([None for _ in range(self.seq_len - 1)]), pred], axis=0
                )
            return pred

    def get_n_columns(self):
        return len(self.columns.columns)

    def test_many(self, x: np.array, y: np.array, column_id: int = None) -> dict:
        """
        It tests cPNN on a single batch, by computing the metrics after averaging each data point's predictions.
        *ONLY FOR BATCH LEARNER*

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: numpy.array
            The target values of the batch.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.

        Returns
        -------
        perf_test: dict
            The dictionary representing test's performance.
            The following metrics are computed: accuracy, kappa, kappa_temporal.
        """
        if x.shape[0] < self.seq_len:
            return {k: None for k in ["accuracy", "kappa", "kappa_temporal"]}

        y_pred = self.predict_many(x, column_id)
        perf = {
            "accuracy": accuracy_score(y, y_pred),
            "kappa": cohen_kappa_score(y, y_pred),
            "kappa_temporal": kappa_temporal_score(y, y_pred, self.first_label_kappa),
        }

        if self.train_verbose:
            print(f"Test accuracy: {perf['accuracy']}")
        return perf

    def test_then_train(
        self,
        x: np.array,
        y: np.array,
        column_id: int = None,
    ) -> tuple:
        """
        It tests cPNN on a single batch, and then it performs the training.
        It computes the loss after averaging each sample's predictions.
        *ONLY FOR BATCH LEARNER*

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: numpy.array
            The target values of the batch.
        column_id: int, default: None.
            The id of the column to use for test. If None the last column is used.

        Returns
        -------
        perf_test: dict
            The dictionary representing test's performance on the batch.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
        perf_test_single_pred: dict
            The dictionary representing test's performance on the batch by predicting of data point's label individually.
            The following metrics are computed: accuracy, kappa, kappa_temporal.
        perf_train: dict
            The dictionary representing training's performance on the batch.
            For each metric the dict contains a list of epochs' values.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
        """
        perf_test = self.test_many(x, y, column_id)
        perf_train = self.learn_many(x, y)
        self.first_label_kappa = torch.tensor(y[-1]).view(1)
        return perf_test, perf_train

    def pretraining(
        self, x: np.array, y: list, epochs: int = 100, batch_size: int = 128
    ) -> dict:
        """
        It performs the pretraining on a pretraining set.
        *ONLY FOR BATCH LEARNER*

        Parameters
        ----------
        x: numpy.array
            The features values of the set.
        y: list
            The target values of the set.
        epochs: int, default: 100.
            The number of training epochs to perform on the set.
        batch_size: int, default: 128.
            The training batch size.

        Returns
        -------
        perf_train: dict
            The dictionary representing training's performance.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
            For each metric the dict contains a list of shape (epochs, n_batches) where n_batches is the training
            batches number.
        """
        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }

        x = torch.tensor(x)
        y = torch.tensor(y).type(torch.LongTensor)
        data = data_utils.TensorDataset(x, y)
        loader = DataLoader(data, batch_size=batch_size, drop_last=False)
        print("Pretraining")
        for e in range(1, epochs + 1):
            for k in perf_train:
                perf_train[k].append([])
            for id_batch, (x, y) in enumerate(loader):
                print(
                    f"{id_batch+1}/{len(loader)} batch of {e}/{epochs} epoch", end="\r"
                )
                x, y_seq = self._cut_in_sequences_tensors(x, y)
                perf_batch = self._fit(x, y)
                for k in perf_batch:
                    perf_train[k][-1].append(perf_batch[k])
        print()
        print()
        return perf_train

    def _fit(self, x, y):
        x, y = x.to(self.columns.device), y.to(self.columns.device)
        outputs = self.columns(x, train=True)
        outputs = get_samples_outputs(outputs)
        loss = customized_loss(outputs, y, self.columns.criterion)
        self.columns.optimizers[-1].zero_grad()
        loss.backward()
        self.columns.optimizers[-1].step()
        outputs = self.columns(x)
        outputs = get_samples_outputs(outputs)
        perf_train = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, y).item(),
            "kappa": cohen_kappa(outputs, y, device=self.columns.device).item(),
            "kappa_temporal": kappa_temporal(outputs, y, self.first_label_kappa).item(),
        }
        return perf_train

    def get_hidden(self, x, column_id=None):
        return self.columns.get_hidden(x, column_id)
