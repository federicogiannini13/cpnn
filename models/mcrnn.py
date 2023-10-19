import pickle
import warnings

from models.cpnn import cPNN
from models.cpnn_columns import cPNNColumns
import numpy as np
import torch

from models.utils import get_pred_from_outputs, get_samples_outputs


class mcRNN(cPNN):
    def __init__(self, **kwargs):
        super(mcRNN, self).__init__(**kwargs)
        self.frozen_columns = []
        self.all_columns = [self.columns]

    def add_new_column(self, task_id=None):
        """
        It adds a new columns to the cPNN architecture, after a concept drift.

        Parameters
        ----------
        task_id: int, default: None
            The id of the new task. If None it increments the last one.
        """
        self.frozen_columns.append(pickle.loads(pickle.dumps(self.columns)))
        self.columns = cPNNColumns(**self.columns_args)
        self.all_columns = self.frozen_columns + [self.columns]
        self.reset_previous_data_points()
        if task_id is None:
            self.task_ids.append(self.task_ids[-1] + 1)
        else:
            self.task_ids.append(task_id)

    def predict_many(self, x: np.array, column_id: int = -1):
        """
        It performs prediction on a single batch. It uses the last column of cPNN architecture.

        Parameters
        ----------
        x: numpy.array
            The features values of the batch.
        column_id: int, default: -1.
            The id of the column to use. If not specified, the last column is used.

        Returns
        -------
        predictions: numpy.array
            The 1D numpy array (with length batch_size) containing predictions of all samples.
        """
        x = np.array(x)
        if x.shape[0] < self.get_seq_len():
            return np.array([None] * x.shape[0])
        first_train = False
        x = self._convert_to_tensor_dataset(x).to(self.all_columns[column_id].device)
        with torch.no_grad():
            outputs = self.all_columns[column_id](x)
            outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate(
                    [np.array([None for _ in range(self.seq_len - 1)]), pred], axis=0
                )
            return pred

    def get_n_columns(self):
        return len(self.all_columns)
