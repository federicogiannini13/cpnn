import torch
import numpy as np

from models.cgru_others import cGRULinearCombination
from models.utils import (
    customized_loss,
    accuracy,
    cohen_kappa,
    kappa_temporal,
    get_samples_outputs,
    get_pred_from_outputs,
)
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from models.clstm import (
    cLSTMLinear,
)
from models.clstm_others import cLSTMLinearCombination, cLSTMLinearCombinationDifferentWeights, cLSTMDouble


class cPNNModules(torch.nn.Module):
    """
    Class that implements the list of single cPNN modules.
    """

    def __init__(
        self,
        model_class=cLSTMDouble,
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
        model_class: default: iLSTM
            The class that implements the single module's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of single modules' Adam Optimizer.
        seq_len: int, default: 5.
            The length of the sliding window that builds the single sequences.
        stride: int, default: 1.
            The length of sliding window's stride.
        loss_on_seq: bool, default: False.
            If True the loss function is computed on each sequence and then averaged on the batch.
            If False each sample's predictions are averaged to obtain a single value.
        combination: bool, default: False.
            If True each iGIM module combines all previous modules.
            If False each iGIM module takes only last module.
        kwargs:
            Parameters of model_class.
        """
        super(cPNNModules, self).__init__()
        kwargs["device"] = (
            torch.device("cpu") if device is None else torch.device(device)
        )
        self.device = kwargs["device"]
        self.model_class = model_class
        self.model_args = kwargs
        self.lr = lr
        self.seq_len = seq_len
        self.stride = stride
        self.combination = combination

        self.models = torch.nn.ModuleList([model_class(**kwargs)])
        self.model_args["input_size"] += self.model_args["hidden_size"]
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

    def _create_optimizer(self, module_id=-1):
        return torch.optim.Adam(self.models[module_id].parameters(), lr=self.lr)

    def forward(self, x, module_id=None, return_initial_states=False):
        if module_id is None:
            module_id = len(self.models) - 1

        if len(self.models) <= 2 or not self.combination:
            prev_h = None
            for i in range(0, module_id + 1):
                out, prev_h, initial_states = self.models[i](x, prev_h)
        else:
            prev_h = None
            prev_h_list = []
            for i in range(0, 2):
                out, prev_h, initial_states = self.models[i](x, prev_h)
                prev_h_list.append(prev_h)
            for i in range(2, len(self.models)):
                out, prev_h, initial_states = self.models[i](x, prev_h_list)
                prev_h_list.append(prev_h)
        if not return_initial_states:
            return out
        return out, initial_states

    def get_hidden(self, x, module_id=None):
        x = self.convert_to_tensor_dataset(x).to(self.device)

        if len(self.models) == 1:
            return None

        if module_id is None:
            module_id = len(self.models) - 2

        out_h = None
        for i in range(0, module_id + 1):
            _, out_h = self.models[i](x, out_h)

        return out_h.detach().numpy()

    def add_new_module(self):
        """
        It adds a new module to the cPNN architecture, after a concept drift.
        Weights of previous modules are frozen.
        It also adds a new optimizer.
        """
        for param in self.models[-1].parameters():
            param.requires_grad = False
        if len(self.models) < 2 or not self.combination:
            self.models.append(self.model_class(**self.model_args))
        else:
            self.model_args["n_modules"] = len(self.models)
            if self.model_class.__name__.startswith("iLSTM"):
                # TODO differentWeights
                self.models.append(
                    cLSTMLinearCombinationDifferentWeights(**self.model_args)
                )
            elif self.model_class.__name__.startswith("iGRU"):
                self.models.append(cGRULinearCombination(**self.model_args))
        self.optimizers.append(self._create_optimizer())

    def update_initial_states(self, initial_states):
        self.models[-1].update_initial_states(initial_states)


class cPNN:
    """
    Class that implements all the cPNN structure.
    """

    def __init__(
        self,
        model_class=cLSTMLinear,
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
        model_class: default: iLSTM.
            The class that implements the single module's architecture.
        device: default: None.
            Torch's device, if None its value is set to 'cpu'.
        lr: float, default: 0.01.
            The learning rate value of single modules' Adam Optimizer.
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
            It is used to automatically add a new module after a concept drift.
        combination: bool, default: False.
            If True each cPNN module combines all previous modules.
            If False each cPNN module takes only last module.
        remember_initial_states: bool, default: False.
            If True model's initial states of batch n are initialized using model's final states of batch n-1.
            If False all batches' model's initial states are initialized as zeros.
        kwargs:
            Parameters of model_class.
        """
        self.modules = cPNNModules(
            model_class, device, lr, seq_len, stride, False, combination, **kwargs
        )
        self.train_epochs = train_epochs
        self.train_verbose = train_verbose
        self.concept_boundaries = concepts_boundaries
        self.samples_cont = 0
        self.remember_initial_states = remember_initial_states
        if first_label_kappa is not None:
            self.first_label_kappa = torch.tensor([first_label_kappa]).view(1)
        else:
            self.first_label_kappa = torch.randint(0, 2, (1,)).view(1)

    def add_new_module(self):
        """
        It adds a new module to the cPNN architecture, after a concept drift.
        """
        self.modules.add_new_module()

    def learn_many(self, x: np.array, y: np.array) -> dict:
        """
        It trains cPNN on a single batch.
        It computes the loss after averaging each sample's predictions.
        Before performing the training, if concept_boundaries was provided during the constructor method, it
        automatically adds a new module after concept drift.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: np.array
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
                print("New module added")
                self.add_new_module()
                self.concept_boundaries = self.concept_boundaries[1:]

        y = list(y)
        x, y, y_seq = self.modules.load_batch(x, y)

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

    def predict_many(self, x: np.array):
        """
        It performs prediction on a single batch. It uses the last module of cPNN architecture.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.

        Returns
        -------
        predictions: numpy.array
            The 1D numpy array (with length batch_size) containing predictions of all samples.
        """
        x = self.modules.convert_to_tensor_dataset(x).to(self.modules.device)
        with torch.no_grad():
            pred, _ = get_pred_from_outputs(get_samples_outputs(self.modules(x)))
            return pred.detach().cpu().numpy()

    def test_many(self, x: np.array, y: np.array, module_id: int = None):
        """
        It tests cPNN on a single batch.
        It computes the metrics after averaging each sample's predictions.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: numpy.array
            The target values of the batch.
        module_id: int, default: -1.
            The id of the module to use. If None the last module is used.

        Returns
        -------
        perf_test: dict
            The dictionary representing test's performance.
            The following metrics are computed: accuracy, loss, kappa, kappa_temporal.
        """
        y = list(y)
        x, y, y_seq = self.modules.load_batch(x, y)
        x, y, y_seq = (
            x.to(self.modules.device),
            y.to(self.modules.device),
            y_seq.to(self.modules.device),
        )

        perf_test = {}
        with torch.no_grad():
            outputs = get_samples_outputs(self.modules(x, module_id))
            perf_test["accuracy"] = accuracy(outputs, y).item()
            perf_test["kappa"] = cohen_kappa(outputs, y).item()
            perf_test["kappa_temporal"] = kappa_temporal(
                outputs, y, self.first_label_kappa
            ).item()
            perf_test["loss"] = customized_loss(
                outputs, y, self.modules.criterion
            ).item()
            if self.train_verbose:
                print(
                    "Test accuracy: ",
                    perf_test["accuracy"],
                    ", Test loss: ",
                    perf_test["loss"],
                    sep="",
                )

        return perf_test

    def test_then_train(
        self,
        x: np.array,
        y: np.array,
        module_id: int = None,
    ) -> (dict, dict):
        """
        It tests cPNN on a single batch, and then it performs the training.
        It computes the loss after averaging each sample's predictions.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        y: numpy.array
            The target values of the batch.
        module_id: int, default: None.
            The id of the module to use for test. If None the last module is used.

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
        perf_test = self.test_many(x, y, module_id)
        perf_train = self.learn_many(x, y)
        self.first_label_kappa = torch.tensor(y[-1]).view(1)
        return perf_test, perf_train

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
                x, y_seq = self.modules.cut_in_sequences_tensors(x, y)
                perf_batch = self._train_batch(x, y)
                for k in perf_batch:
                    perf_train[k][-1].append(perf_batch[k])
        print()
        print()
        return perf_train

    def _train_batch(self, x, y):
        x, y = x.to(self.modules.device), y.to(self.modules.device)
        outputs = get_samples_outputs(self.modules(x))
        loss = customized_loss(outputs, y, self.modules.criterion)
        self.modules.optimizers[-1].zero_grad()
        loss.backward()
        self.modules.optimizers[-1].step()
        out, initial_states = self.modules(x, return_initial_states=True)
        if self.remember_initial_states:
            self.modules.update_initial_states(initial_states)
        outputs = get_samples_outputs(out)
        perf_train = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, y).item(),
            "kappa": cohen_kappa(outputs, y, device=self.modules.device).item(),
            "kappa_temporal": kappa_temporal(outputs, y, self.first_label_kappa).item(),
        }
        return perf_train

    def get_hidden(self, x, module_id=None):
        return self.modules.get_hidden(x, module_id)


class cPNNModulesExp(cPNNModules):
    def forward(self, x, module_id=None):
        if module_id is None:
            module_id = len(self.models) - 1

        prev_h = None
        for i in range(0, module_id):
            out, prev_h = self.models[i](x, prev_h)

        if len(self.models) > 1:
            out, _ = self.models[-1](torch.zeros(x.size()), prev_h)
        else:
            out, _ = self.models[-1](x, prev_h)

        return out


class cPNNExp(cPNN):
    def __init__(
        self,
        model_class=cLSTMDouble,
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
        super(cPNNExp, self).__init__(
            model_class,
            device,
            lr,
            seq_len,
            stride,
            first_label_kappa,
            train_epochs,
            train_verbose,
            concepts_boundaries,
            **kwargs,
        )
        self.modules = cPNNModulesExp(
            model_class, device, lr, seq_len, stride, False, **kwargs
        )
