import numpy as np

def mrr_at_k(y_true, y_pred, k=10):
    rr = []
    for t, pred_list in zip(y_true, y_pred):
        try:
            rank = pred_list[:k].index(t) + 1
            rr.append(1.0 / rank)
        except ValueError:
            rr.append(0.0)
    return float(np.mean(rr))

def accuracy(y_true, y_pred):
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
