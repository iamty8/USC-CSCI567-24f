import numpy as np
import torch
import torch.nn.functional as F


def swapped_prediction(logits, targets):
    loss = 0
    for view in range(2):
        for other_view in np.delete(range(2), view):
            loss += cross_entropy_loss(logits[other_view], targets[view])
    return loss / 2


def cross_entropy_loss(preds, targets):
    preds = F.log_softmax(preds / 0.1, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)
