import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

from models.cpnn_columns import cPNNColumns
from models.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    kappa_temporal,
    get_samples_outputs,
    get_pred_from_outputs, kappa_temporal_score,
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
        concepts_boundaries: list = None,
        combination: bool = False,
        remember_initial_states: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        column_class: default: iLSTM.
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
            Training epochs to perform in learn_many method.
        train_verbose: bool, default:False.
            True if, during the learn_many execution, you want to print the metrics after each training epoch.
        concepts_boundaries: list, default:None.
            If not None it represents the boundaries of each concept (its last sample's index).
            It is used to automatically add a new column after a concept drift.
        combination: bool, default: False.
            If True each cPNN column combines all previous columns.
            If False each cPNN column takes only last column.
        remember_initial_states: bool, default: False.
            If True model's initial states of batch n are initialized using model's final states of batch n-1.
            If False all batches' model's initial states are initialized as zeros.
        kwargs:
            Parameters of column_class.
        """
        self.columns_args = kwargs
        self.columns_args["column_class"] = column_class
        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns_args["stride"] = stride
        self.columns_args["combination"] = combination
        self.columns_args["seq_len"] = seq_len
        self.columns = cPNNColumns(
            **self.columns_args
        )
        self._seq_len = seq_len
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.concept_boundaries = concepts_boundaries
        self.samples_cont = 0
        self.remember_initial_states = remember_initial_states
        self.previous_data_points_anytime = None
        if first_label_kappa is not None:
            self.first_label_kappa = torch.tensor([first_label_kappa]).view(1)
        else:
            self.first_label_kappa = torch.randint(0, 2, (1,)).view(1)

    def get_seq_len(self):
        return self._seq_len

    def add_new_column(self):
        """
        It adds a new column to the cPNN architecture, after a concept drift.
        """
        self.reset_previous_data_points_anytime()
        self.columns.add_new_column()

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains cPNN on a single batch.
        It computes the loss after averaging each sample's predictions.
        Before performing the training, if concept_boundaries was provided during the constructor method, it
        automatically adds a new column after concept drift.

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
        if self.concept_boundaries is not None and len(self.concept_boundaries) > 0:
            if self.samples_cont >= self.concept_boundaries[0]:
                print("New column added")
                self.add_new_column()
                self.concept_boundaries = self.concept_boundaries[1:]

        x = np.array(x)
        y = list(y)
        x, y, y_seq = self.columns.load_batch(x, y)

        perf_train = {
            "accuracy": [],
            "loss": [],
            "kappa": [],
            "kappa_temporal": [],
        }
        for e in range(1, self.train_epochs + 1):
            perf_epoch = self._train_batch(x, y)
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
        It performs prediction on a single batch. It uses the last column of cPNN architecture.

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
        x = self.columns.convert_to_tensor_dataset(x).to(self.columns.device)
        with torch.no_grad():
            pred, _= get_pred_from_outputs(get_samples_outputs(self.columns(x, column_id)))
            return pred.detach().cpu().numpy()

    def predict_one (self, x : np.array, column_id: int = None, previous_data_points: np.array = None):
        """
        It performs prediction on a single data point using the last column of cPNN architecture.

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        Returns
        -------
        prediction : int
            The predicted int label of x.
        """
        x = np.array(x)
        x = x.reshape(1, -1)
        if previous_data_points is not None:
            self.previous_data_points_anytime = previous_data_points
        if self.previous_data_points_anytime is None:
            self.previous_data_points_anytime = x
            return None
        if len(self.previous_data_points_anytime) != self._seq_len - 1:
            self.previous_data_points_anytime = np.concatenate([self.previous_data_points_anytime, x])
            return None
        self.previous_data_points_anytime = np.concatenate([self.previous_data_points_anytime, x])
        x = self.columns.convert_to_tensor_dataset(self.previous_data_points_anytime).to(self.columns.device)
        self.previous_data_points_anytime = self.previous_data_points_anytime[1:]
        with torch.no_grad():
            pred, _ = get_pred_from_outputs(self.columns(x, column_id)[0])
            return int(pred[-1].detach().cpu().numpy())

    def get_n_columns(self):
        return len(self.columns.columns)

    def reset_previous_data_points_anytime(self):
        self.previous_data_points_anytime = None

    def test_many_with_single_pred(self, x: np.array, y: np.array,  column_id: int = None) -> dict:
        """
        It tests cPNN on a single batch, by computing the metrics after averaging each data point's predictions.
        Each prediction is made on the single data point individually.

        Parameters
        ----------
        x: numpy.array
            The features values of batch.
        y: numpy.array
            The real int label of the batch.
        column_id: int, default: None.
            The id of the column to use. If None the last column is used.

        Returns
        -------
        perf: dict
            A dictionary containing the evaluated metrics.
        """
        y_pred = [self.predict_one(x_, column_id=column_id) for x_ in x]
        y = np.array([y[i] for i in range(len(y_pred)) if y_pred[i] is not None])
        y_pred = np.array([y_ for y_ in y_pred if y_ is not None])
        if len(y_pred) == 0:
            return {k: None for k in ["accuracy", "kappa", "kappa_temporal"]}
        return {
            "accuracy" : accuracy_score(y, y_pred),
            "kappa" : cohen_kappa_score(y, y_pred),
            "kappa_temporal" : kappa_temporal_score(y, y_pred, self.first_label_kappa)
        }

    def test_many(self, x: np.array, y: np.array, column_id: int = None) -> dict:
        """
        It tests cPNN on a single batch, by computing the metrics after averaging each data point's predictions.

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
        if x.shape[0] < self._seq_len:
            return {k: None for k in ["accuracy", "kappa", "kappa_temporal"]}

        y_pred = self.predict_many(x, column_id)
        perf = {
            "accuracy": accuracy_score(y, y_pred),
            "kappa": cohen_kappa_score(y, y_pred),
            "kappa_temporal": kappa_temporal_score(y, y_pred, self.first_label_kappa)
        }

        if self.train_verbose:
            print(f"Test accuracy: {perf['accuracy']}")
        return perf

    def test_then_train(
        self,
        x: np.array,
        y: np.array,
        column_id: int = None,
    ) -> (dict, dict, dict):
        """
        It tests cPNN on a single batch, and then it performs the training.
        It computes the loss after averaging each sample's predictions.

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
        perf_test_single_pred = self.test_many_with_single_pred(x, y)
        perf_test = self.test_many(x, y, column_id)
        perf_train = self.learn_many(x, y)
        self.first_label_kappa = torch.tensor(y[-1]).view(1)
        return perf_test, perf_test_single_pred, perf_train

    def pretraining(
        self, x: np.array, y: list, epochs: int = 100, batch_size: int = 128
    ) -> dict:
        """
        It performs the pretraining on a pretraining set.

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
                x, y_seq = self.columns.cut_in_sequences_tensors(x, y)
                perf_batch = self._train_batch(x, y)
                for k in perf_batch:
                    perf_train[k][-1].append(perf_batch[k])
        print()
        print()
        return perf_train

    def _train_batch(self, x, y):
        x, y = x.to(self.columns.device), y.to(self.columns.device)
        outputs = get_samples_outputs(self.columns(x))
        loss = customized_loss(outputs, y, self.columns.criterion)
        self.columns.optimizers[-1].zero_grad()
        loss.backward()
        self.columns.optimizers[-1].step()
        out, initial_states = self.columns(x, return_initial_states=True)
        if self.remember_initial_states:
            self.columns.update_initial_states(initial_states)
        outputs = get_samples_outputs(out)
        perf_train = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, y).item(),
            "kappa": cohen_kappa(outputs, y, device=self.columns.device).item(),
            "kappa_temporal": kappa_temporal(outputs, y, self.first_label_kappa).item(),
        }
        return perf_train

    def get_hidden(self, x, column_id=None):
        return self.columns.get_hidden(x, column_id)

