import torch

from models.utils import accuracy, _pe_kappa


@torch.enable_grad()
def loss_many_to_many(predictions, y, criterion):
    loss = []
    for i in range(0, predictions.size()[1]):
        pred = predictions[:, i, :]
        loss.append(criterion(pred, y[0:, i]).reshape(predictions.size()[0], 1))
    # B x seq_len tensor:
    loss = torch.cat(loss, dim=1)
    # mean over seq
    loss = torch.sum(loss, dim=1) / predictions.size()[1]
    # calculate mean over batch
    return torch.sum(loss) / predictions.size()[0]


def get_accuracy_from_pred(predictions, y, batch_reduction=True):
    with torch.no_grad():
        acc = []
        for i in range(0, predictions.size()[1]):
            if len(list(predictions.size())) > 2:
                pred = predictions[:, i, :]
            else:
                pred = predictions[:, i]
            acc.append(
                accuracy(pred, y[0:, i], reduction="none").reshape(
                    predictions.size()[0], 1
                )
            )
        # B x seq_len tensor:
        acc = torch.cat(acc, dim=1)
        # mean over seq
        acc = torch.sum(acc, dim=1) / predictions.size()[1]
        if batch_reduction:
            # calculate mean over batch
            acc = torch.sum(acc) / predictions.size()[0]
        return acc


def get_kappa_from_pred(predictions, y, reduction="mean"):
    with torch.no_grad():
        accuracy = get_accuracy_from_pred(predictions, y, False)
        p_e = []
        for i in range(0, predictions.size()[0]):
            pred = predictions[i, :, :]
            targets = y[i, :]
            p_e.append(_pe_kappa(pred, targets).view(1))
        p_e = torch.cat(p_e)
        kappa = (accuracy - p_e) / (torch.ones(p_e.size()[0]) - p_e)
        kappa = torch.nan_to_num(kappa, nan=0, posinf=0, neginf=0)
        if reduction == "mean":
            kappa = kappa.sum() / kappa.size()[0]
        return kappa


def get_kappa_temporal_from_pred(predictions, y, first_label, reduction="mean"):
    with torch.no_grad():
        accuracy = get_accuracy_from_pred(predictions, y, False)
        predictions_naive = torch.cat(
            [torch.tensor(first_label).view(1), y[0, :-1]]
        ).view(1, y.size()[1])
        predictions_naive = torch.cat([predictions_naive, y[:-1, :]])
        accuracy_naive = get_accuracy_from_pred(
            predictions_naive, y, batch_reduction=False
        )
        kappa = (accuracy - accuracy_naive) / (
            torch.ones(accuracy.size()[0]) - accuracy_naive
        )
        kappa = torch.nan_to_num(kappa, nan=0, posinf=0, neginf=0)
        if reduction == "mean":
            kappa = kappa.sum() / kappa.size()[0]
        return kappa
