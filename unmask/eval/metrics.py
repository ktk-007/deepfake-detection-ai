"""Video-level metrics. Video score = mean of the per-frame fake probabilities."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve


def video_scores(frame_probs, labels, keys):
    """-> (video keys, video labels, video scores) with score = mean frame fake-probability."""
    acc, lab = defaultdict(list), {}
    for p, y, k in zip(frame_probs, labels, keys):
        acc[k].append(float(p))
        lab[k] = int(y)
    vids = sorted(acc)
    return vids, np.array([lab[v] for v in vids]), np.array([np.mean(acc[v]) for v in vids])


def eer(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def binary_metrics(y, s) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    out = dict(n=len(y), n_fake=int(y.sum()), n_real=int((1 - y).sum()))
    if len(set(y.tolist())) < 2:
        return out | dict(auc=float("nan"), ap=float("nan"), eer=float("nan"),
                          logloss=float("nan"), acc=float(((s >= 0.5) == y).mean()))
    sc = np.clip(s, 1e-6, 1 - 1e-6)
    return out | dict(
        auc=float(roc_auc_score(y, s)),
        ap=float(average_precision_score(y, s)),
        eer=eer(y, s),
        logloss=float(log_loss(y, sc, labels=[0, 1])),
        acc=float(((s >= 0.5) == y).mean()),
    )
