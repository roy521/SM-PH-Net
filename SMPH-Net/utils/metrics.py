import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def compute_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    probs = 1 / (1 + np.exp(-preds))
    auc = roc_auc_score(labels, probs)
    preds_bin = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds_bin)
    f1 = f1_score(labels, preds_bin)
    return acc, auc, f1
