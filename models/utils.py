import torch
import numpy as np
from sklearn.metrics import accuracy_score


def get_pred_from_outputs(outputs):
    """
    Given the values assigned by the model to the different classes, it returns classes probabilities and
    model's prediction.

    Parameters
    ----------
    outputs: torch.Tensor
        A tensor containing, for each element, the values assigned by the model to the different classes.

    Returns
    -------
    predictions: torch.Tensor
        A tensor containing, for each element, the target predicted by the model.
    probabilities: torch.Tensor
        A tensor containing, for each element, the probabilities assigned by the model to the different classes.
    """
    probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.argmax(dim=1), probs


def accuracy(outputs, targets, reduction="mean"):
    """
    Given the model's outputs and the real targets it computes the accuracy.

    Parameters
    ----------
    outputs: torch.Tensor
        A tensor containing, for each element, the values assigned by the model to the different classes.
    targets: torch.Tensor
        A tensor containing, for each element, the real classes values.
    reduction: str, default: 'mean'
        If "mean" the accuracy is calculated over the given batch.
        Otherwise, it only returns a tensor containing 1 where the model's prediction is correct and 0
        where is not correct.
    Returns
    -------
    accuracy: torch.Tensor
        The accuracy tensor. If reduction is "mean", to obtain the value, you have to call the method .item().
    """
    if len(list(outputs.size())) > 1:
        predictions, _ = get_pred_from_outputs(outputs)
    else:
        predictions = outputs
    acc = (predictions == targets).float()
    if reduction == "mean":
        acc = torch.sum(acc) / targets.size(0)
    return acc


def _pe_kappa(outputs, targets, device=torch.device("cpu")):
    if len(list(outputs.size())) > 1:
        predictions, _ = get_pred_from_outputs(outputs)
    else:
        predictions = outputs
    ones = torch.ones(predictions.size()[0], device=device)
    zeros = torch.zeros(predictions.size()[0], device=device)
    pos = (targets == ones).float().sum()
    pos_ = (predictions == ones).float().sum()
    neg = (targets == zeros).float().sum()
    neg_ = (predictions == zeros).float().sum()
    return (pos * pos_ + neg * neg_) / (predictions.size()[0] * predictions.size()[0])


def cohen_kappa(outputs, targets, device=torch.device("cpu")):
    """
    Given the model's outputs and the real targets it computes the Cohen Kappa.

    Parameters
    ----------
    outputs: torch.Tensor
        A tensor containing, for each element, the values assigned by the model to the different classes.
    targets: torch.Tensor
        A tensor containing, for each element, the real classes values.
    Returns
    -------
    kappa: torch.Tensor
        The kappa tensor. To obtain the value, you have to call the method .item().
    """
    model_accuracy = accuracy(outputs, targets, reduction="mean")
    p_e = _pe_kappa(outputs, targets, device=device)
    model_kappa = (model_accuracy - p_e) / (1 - p_e)
    model_kappa = torch.nan_to_num(model_kappa, nan=0, posinf=0, neginf=0)
    return model_kappa


def kappa_temporal_score(y_true: np.array, y_pred: np.array, first_label=None):
    """
    Parameters
    ----------
    y_true: np.array
        The numpy array containing the real labels of the batch.
    y_pred: np.array
        The numpy array containing the predicted labels of the batch.
    first_label: int, default: None
        It represents the real label of the last sample before the current batch.
        It is used, for the naive approach to predict the first data point's label of the current batch.
        If None, a random label is generated.
    Returns
    -------
    kappa_temporal : float
        The evaluated kappa temporal on the batch
    """
    if first_label is None:
        first_label = np.random.randint(0, 2, (1,))
    naive_predictions = np.concatenate([first_label, y_true[:-1]])
    naive_accuracy = accuracy_score(y_true, naive_predictions)
    model_accuracy = accuracy_score(y_true, y_pred)
    model_kappa = (model_accuracy - naive_accuracy) / (1 - naive_accuracy)
    model_kappa = np.nan_to_num(model_kappa, nan=0, posinf=0, neginf=0)
    return model_kappa


def kappa_temporal(outputs, targets, first_label=None):
    """
    Given the model's outputs and the real targets it computes the Kappa Temporal.

    Parameters
    ----------
    outputs: torch.Tensor
        A tensor containing, for each element, the values assigned by the model to the different classes.
    targets: torch.Tensor
        A tensor containing, for each element, the real label.
    first_label: int, default: None.
        It represents the real label of the last sample before the current stream.
        It is used, for the naive approach to predict the first element of the current stream.
        If None, a random label is generated.
    Returns
    -------
    kappa: torch.Tensor
        The kappa temporal tensor. To obtain the value, you have to call the method .item().
    """
    if first_label is None:
        first_label = torch.randint(0, 2, (1,))
    if len(list(outputs.size())) > 1:
        predictions, _ = get_pred_from_outputs(outputs)
    else:
        predictions = outputs
    naive_predictions = torch.cat([first_label, targets[:-1]])
    naive_accuracy = accuracy(naive_predictions, targets)
    model_accuracy = accuracy(predictions, targets)
    model_kappa = (model_accuracy - naive_accuracy) / (1 - naive_accuracy)
    model_kappa = torch.nan_to_num(model_kappa, nan=0, posinf=0, neginf=0)
    return model_kappa


@torch.enable_grad()
def customized_loss(predictions, y, criterion):
    return criterion(predictions, y)


def get_samples_outputs(outputs):
    """
    It transforms the 3D tensor representing sequences' outputs in a 2D tensor.
    The outputs of each sample are averaged over the different sequences, in order to obtain a tensor with shape
    (batch_size, n_classes).

    Parameters
    ----------
    outputs: torch.Tensor
        The 3D tensor with shape (n_sequences, seq_len, n_classes). It represents classes outputs for each element
        of each sequence.

    Returns
    -------
    samples_outputs: torch.Tensor
        The tensor with shape (batch_size, n_classes) containing, for every sample, an output for each class.
    """
    outputs_flipped = torch.fliplr(outputs)
    num_sub_batches, sub_batch_size, _ = outputs_flipped.size()
    offsets = range((sub_batch_size - 1), -num_sub_batches, -1)

    diagonals = [
        torch.mean(torch.diagonal(outputs_flipped, offset), axis=-1).view(
            1, outputs.size()[2]
        )
        for offset in offsets
    ]

    samples_outputs = torch.cat(diagonals, dim=0)
    return samples_outputs
