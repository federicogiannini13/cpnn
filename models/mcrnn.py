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

    def add_new_column(self):
        """
        It adds a new columns to the cPNN architecture, after a concept drift.
        """
        self.frozen_columns.append(pickle.loads(pickle.dumps(self.columns)))
        self.columns = cPNNColumns(**self.columns_args)
        self.all_columns = self.frozen_columns + [self.columns]
        self.reset_previous_data_points()
    
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
        x = self._convert_to_tensor_dataset(x).to(self.all_columns[column_id].device)
        with torch.no_grad():
            outputs = self.all_columns[column_id](x)
            if not self.loss_on_seq:
                outputs = get_samples_outputs(outputs)
            pred, _ = get_pred_from_outputs(outputs)
            pred = pred.detach().cpu().numpy()
            if first_train:
                return np.concatenate([np.array([None for _ in range(self.seq_len-1)]), pred], axis=0)
            return pred

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
            self.previous_data_points_anytime_inference = previous_data_points
        if self.previous_data_points_anytime_inference is None:
            self.previous_data_points_anytime_inference = x
            return None
        if len(self.previous_data_points_anytime_inference) != self.seq_len - 1:
            self.previous_data_points_anytime_inference = np.concatenate([self.previous_data_points_anytime_inference, x])
            return None
        self.previous_data_points_anytime_inference = np.concatenate([self.previous_data_points_anytime_inference, x])
        x = self._convert_to_tensor_dataset(self.previous_data_points_anytime_inference)\
            .to(self.all_columns[column_id].device)
        self.previous_data_points_anytime_inference = self.previous_data_points_anytime_inference[1:]
        with torch.no_grad():
            if not self.loss_on_seq:
                pred, _ = get_pred_from_outputs(self.all_columns[column_id](x)[0])
            else:
                pred, _ = get_pred_from_outputs(self.all_columns[column_id](x))
            return int(pred[-1].detach().cpu().numpy())

    def get_n_columns(self):
        return len(self.all_columns)