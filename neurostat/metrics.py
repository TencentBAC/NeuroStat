"""Evaluation metrics: AUROC, AUPR, MCC, Balanced Accuracy, TPR@FPR=5%."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)


def AUROC(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = list(neg_list) + list(pos_list)
    fpr, tpr, _ = roc_curve(labels, preds)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))


def AUPR(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = list(neg_list) + list(pos_list)
    precision, recall, _ = precision_recall_curve(labels, preds)
    return precision.tolist(), recall.tolist(), float(auc(recall, precision))


def MCC(neg_list, pos_list, threshold: float = 0.5) -> float:
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = list(neg_list) + list(pos_list)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(matthews_corrcoef(labels, pred_labels))


def Balanced_Accuracy(neg_list, pos_list, threshold: float = 0.5) -> float:
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = list(neg_list) + list(pos_list)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(balanced_accuracy_score(labels, pred_labels))


def TPR_at_FPR5(neg_list, pos_list) -> float:
    """TPR at FPR = 5% (linearly interpolated)."""
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = list(neg_list) + list(pos_list)
    fpr, tpr, _ = roc_curve(labels, preds)

    valid_indices = np.where(fpr <= 0.05)[0]
    if not valid_indices.size:
        return 0.0

    i_max = valid_indices[-1]
    if i_max == len(fpr) - 1:
        return float(tpr[i_max])

    fpr_low, fpr_high = fpr[i_max], fpr[i_max + 1]
    tpr_low, tpr_high = tpr[i_max], tpr[i_max + 1]
    ratio = (0.05 - fpr_low) / (fpr_high - fpr_low)
    return float(tpr_low + ratio * (tpr_high - tpr_low))
