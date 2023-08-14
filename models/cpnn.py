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
        anytime_learner: bool = False,
        loss_on_seq: bool = False,
        remember_states: bool = False,
        quantize: bool = False,
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
            In case of anytime_learner=False, the training epochs to perform in learn_many method.
        train_verbose: bool, default:False.
            True if, during the learn_many execution, you want to print the metrics after each training epoch.
        concepts_boundaries: list, default:None.
            If not None it represents the boundaries of each concept (its last sample's index).
            It is used to automatically add a new column after a concept drift.
        combination: bool, default: False.
            If True each cPNN column combines all previous columns.
            If False each cPNN column takes only last column.
        anytime_learner: bool, default: False.
            If True the model learns data point by data point by data point.
            Otherwise, il learns batch by batch.
        loss_on_seq: bool, default: False.
            In case of anytime_learner = False, if True the model considers only past temporal dependencies. The model
            is a many_to_one model and each data point's prediction is associated to the first sequence in which
            it appears.
            If False, the model considers both past and future temporal dependencies.The model is a many_to_many model
            and each data point's prediction is the average prediction between all the sequences in which it appears.
        remember_states: bool, default: False
            In case of anytime learner and cGRU, if True the initial h0 is set as h1 of the previous sequence.
        quantize: bool, default: False
            If True, after a concept drift, the column is quantized.
        kwargs:
            Parameters of column_class.
        """
        self.anytime_learner = anytime_learner
        if self.anytime_learner:
            self.loss_on_seq = True
            self.many_to_one = True
            self.remember_states = remember_states
        else:
            self.loss_on_seq = loss_on_seq
            if loss_on_seq:
                self.many_to_one = True
            else:
                self.many_to_one = False
            self.remember_states = False

        self.columns_args = kwargs
        self.columns_args["column_class"] = column_class
        self.columns_args["device"] = device
        self.columns_args["lr"] = lr
        self.columns_args["combination"] = combination
        self.columns_args["remember_states"] = self.remember_states
        self.columns_args["many_to_one"] = self.many_to_one
        self.columns_args["quantize"] = quantize
        self.columns = cPNNColumns(**self.columns_args)
        self.seq_len = seq_len
        self.stride = stride
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.concept_boundaries = concepts_boundaries
        self.samples_cont = 0
        self.previous_data_points_anytime_inference = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_batch_train = None
        self.previous_data_points_batch_test = None

        if first_label_kappa is not None:
            self.first_label_kappa = torch.tensor([first_label_kappa]).view(1)
        else:
            self.first_label_kappa = torch.randint(0, 2, (1,)).view(1)

    def get_seq_len(self):
        return self.seq_len

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

    def add_new_column(self):
        """
        It adds a new column to the cPNN architecture, after a concept drift.
        """
        self.reset_previous_data_points()
        self.columns.add_new_column()

    def learn_one(self, x:np.array, y: np.array, previous_data_points: np.array = None):
        """
        It trains cPNN on a single data point.
        Before performing the training, if concept_boundaries was provided during the constructor method, it
        automatically adds a new column after concept drift.
        *ONLY FOR ANYTIME LEARNER*

        Parameters
        ----------
        x: numpy.array or list
            The features values of the single data point.
        y: numpy.array or list
            The target value of the single data point.
        previous_data_points: numpy.array, default: None.
            The features value of the data points preceding x in the sequence.
            If None, it uses the last seq_len-1 points seen during the last calls of the method.
            It returns None if the model has not seen yet seq_len-1 data points and previous_data_points is None.
        """
        if not self.anytime_learner:
            warnings.warn(
                "The model is a batch learner, it cannot learn from a single data point.\n" +
                "Call learn_many on a batch containing a single sequence"
            )
            return None
        if self.concept_boundaries is not None and len(self.concept_boundaries) > 0:
            if self.samples_cont >= self.concept_boundaries[0]:
                print("New column added")
                self.add_new_column()
                self.concept_boundaries = self.concept_boundaries[1:]

        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1, -1)
        if previous_data_points is not None:
            self.previous_data_points_anytime_train = previous_data_points
        if self.previous_data_points_anytime_train is None:
            self.previous_data_points_anytime_train = x
            return None
        if len(self.previous_data_points_anytime_train) != self.seq_len - 1:
            self.previous_data_points_anytime_train = np.concatenate(
                [self.previous_data_points_anytime_train, x])
            return None
        self.previous_data_points_anytime_train = np.concatenate([self.previous_data_points_anytime_train, x])
        x, y, _ = self._load_batch(self.previous_data_points_anytime_train, y)
        self._fit(x, y.view(-1))
        self.previous_data_points_anytime_train = self.previous_data_points_anytime_train[1:]

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains cPNN on a single batch.
        It computes the loss after averaging each sample's predictions.
        Before performing the training, if concept_boundaries was provided during the constructor method, it
        automatically adds a new column after concept drift.
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
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n" +
                "Loop on learn_one method to learn from multiple data points"
            )
            return {}
        if self.concept_boundaries is not None and len(self.concept_boundaries) > 0:
            if self.samples_cont >= self.concept_boundaries[0]:
                print("New column added")
                self.add_new_column()
                self.concept_boundaries = self.concept_boundaries[1:]

        x = np.array(x)
        y = list(y)
        first_batch = False
        if self.loss_on_seq:
            if self.previous_data_points_batch_train is None:
                first_batch = True
            else:
                x = np.concatenate([x, self.previous_data_points_batch_train], axis=0)
                self.previous_data_points_batch_train = x[-(self.seq_len-1):]
        x, y, y_seq = self._load_batch(x, y)
        if first_batch:
            y = y[self.seq_len - 1:]

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
        if self.anytime_learner:
            if self.anytime_learner:
                warnings.warn(
                    "The model is an anytime learner, it cannot predict a batch of data.\n" +
                    "Loop on predict_one method to predict on multiple data points"
                )
                return None
        x = np.array(x)
        if x.shape[0] < self.get_seq_len():
            return np.array([None] * x.shape[0])
        first_train = False
        if self.loss_on_seq:
            if self.previous_data_points_batch_train is not None:
                x = np.concatenate([x, self.previous_data_points_batch_train], axis=0)
                self.previous_data_points_batch_train = x[-(self.seq_len-1):]
            else:
                first_train = True
        x = self._convert_to_tensor_dataset(x).to(self.columns.device)
        with torch.no_grad():
            outputs = self.columns(x, column_id)
            if not self.loss_on_seq:
                outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate([np.array([None for _ in range(self.seq_len-1)]), pred], axis=0)
            return pred

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
        x = np.array(x).reshape(1, -1)
        if previous_data_points is not None:
            self.previous_data_points_anytime_inference = previous_data_points
        if self.previous_data_points_anytime_inference is None:
            self.previous_data_points_anytime_inference = x
            return None
        if len(self.previous_data_points_anytime_inference) != self.seq_len - 1:
            self.previous_data_points_anytime_inference = np.concatenate([self.previous_data_points_anytime_inference, x])
            return None
        self.previous_data_points_anytime_inference = np.concatenate([self.previous_data_points_anytime_inference, x])
        x = self._convert_to_tensor_dataset(self.previous_data_points_anytime_inference).to(self.columns.device)
        self.previous_data_points_anytime_inference = self.previous_data_points_anytime_inference[1:]
        with torch.no_grad():
            if not self.loss_on_seq:
                pred, _ = get_pred_from_outputs(self.columns(x, column_id)[0])
            else:
                pred, _ = get_pred_from_outputs(self.columns(x, column_id))
            return int(pred[-1].detach().cpu().numpy())

    def get_n_columns(self):
        return len(self.columns.columns)

    def reset_previous_data_points(self):
        self.previous_data_points_batch_train = None
        self.previous_data_points_anytime_train = None
        self.previous_data_points_anytime_inference = None

    def test_many_anytime(self, x: np.array, y: np.array, column_id: int = None) -> dict:
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
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n" +
                "You cannot call this method"
            )
            return {}
        if x.shape[0] < self.seq_len:
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
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n" +
                "You cannot call this method"
            )
            return ()
        perf_test_single_pred = self.test_many_anytime(x, y)
        perf_test = self.test_many(x, y, column_id)
        perf_train = self.learn_many(x, y)
        self.first_label_kappa = torch.tensor(y[-1]).view(1)
        return perf_test, perf_test_single_pred, perf_train

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
        if self.anytime_learner:
            warnings.warn(
                "The model is an anytime learner, it cannot learn from batch.\n" +
                "You cannot call this method"
            )
            return {}

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
        if not self.loss_on_seq:
            outputs = get_samples_outputs(outputs)
        loss = customized_loss(outputs, y, self.columns.criterion)
        self.columns.optimizers[-1].zero_grad()
        loss.backward()
        self.columns.optimizers[-1].step()
        outputs = self.columns(x)
        if not self.loss_on_seq:
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

