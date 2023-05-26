import pickle

from models.cpnn import cPNN
from models.cpnn_columns import cPNNColumns
import numpy as np
import torch

from models.utils import get_pred_from_outputs, get_samples_outputs


class mcLSTM(cPNN):
    def __init__(self, **kwargs):
        super(mcLSTM, self).__init__(**kwargs)
        self.frozen_columns = []
        self.all_columns = [self.columns]

    def add_new_column(self):
        """
        It adds a new columns to the cPNN architecture, after a concept drift.
        """
        self.frozen_columns.append(pickle.loads(pickle.dumps(self.columns)))
        self.columns = cPNNColumns(**self.columns_args)
        self.all_columns = self.frozen_columns + [self.columns]
        self.reset_previous_data_points_anytime()
    
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
        x = self.all_columns[column_id].convert_to_tensor_dataset(x).to(self.all_columns[column_id].device)
        with torch.no_grad():
            pred, _= get_pred_from_outputs(get_samples_outputs(self.all_columns[column_id](x)))
            return pred.detach().cpu().numpy()

    def predict_one (self, x : np.array, column_id: int = -1, previous_data_points: np.array = None):
        """
        It performs prediction on a single data point using the last column of cPNN architecture.

        Parameters
        ----------
        x: numpy.array
            The features values of the single data point.
        column_id: int, default: -1.
            The id of the column to use. If not specified, the last column is used.
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
        x = self.all_columns[column_id].convert_to_tensor_dataset(self.previous_data_points_anytime)\
            .to(self.all_columns[column_id].device)
        self.previous_data_points_anytime = self.previous_data_points_anytime[1:]
        with torch.no_grad():
            pred, _ = get_pred_from_outputs(self.all_columns[column_id](x)[0])
            return int(pred[-1].detach().cpu().numpy())

    def get_n_columns(self):
        return len(self.all_columns)