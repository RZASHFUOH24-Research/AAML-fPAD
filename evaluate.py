"""
Evaluation routines (ISO/IEC 30107-3 metrics).

Score used for ranking and thresholding: cosine similarity to the LIVE centre.
Reported:
  * AUC (threshold-free)
  * APCER / BPCER / HTER (= ACER) with the argmax rule  (cos_live > cos_spoof)
  * APCER / BPCER / HTER with the threshold rule cos_live >= tau, where tau is
    calibrated on the VALIDATION set at the Equal Error Rate (EER) point.
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from visualize import plot_confusion_matrix


@torch.no_grad()
def extract(model, criterion, dataloader, device):
    """Run the model in eval mode. Returns (L2-normalised features, cosines [N,2], labels)."""
    model.eval()
    feats, cosines, labels = [], [], []
    for imgs, y in dataloader:
        f = model(imgs.to(device))
        cosines.append(criterion.cosine(f).cpu())
        feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
        labels.append(torch.as_tensor(y))
    return (torch.cat(feats).numpy(), torch.cat(cosines).numpy(), torch.cat(labels).numpy())


def find_threshold_eer(y_true, scores):
    """Threshold on `scores` (live = positive) where FAR ~= FRR."""
    fpr, tpr, thr = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    finite = np.isfinite(thr)
    idx = np.argmin(np.abs(fnr[finite] - fpr[finite]))
    return float(thr[finite][idx])


def error_rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    apcer = fp / max(tn + fp, 1)     # attacks accepted as live
    bpcer = fn / max(tp + fn, 1)     # live rejected as spoof
    hter = (apcer + bpcer) / 2.0     # HTER = ACER
    return {"APCER": float(apcer), "BPCER": float(bpcer), "HTER": float(hter)}, cm


def evaluate(model, criterion, val_loader, test_loader, device, save_dir=None):
    # --- calibrate tau on validation ---
    _, val_cos, val_y = extract(model, criterion, val_loader, device)
    tau = find_threshold_eer(val_y, val_cos[:, 1])

    # --- test ---
    _, test_cos, test_y = extract(model, criterion, test_loader, device)
    score = test_cos[:, 1]                                   # cos(theta_real)

    auc = float(roc_auc_score(test_y, score))

    pred_argmax = (test_cos[:, 1] > test_cos[:, 0]).astype(int)
    m_argmax, cm_argmax = error_rates(test_y, pred_argmax)

    pred_tau = (score >= tau).astype(int)
    m_tau, cm_tau = error_rates(test_y, pred_tau)

    results = {
        "AUC": auc,
        "tau_val_EER": tau,
        "argmax": m_argmax,
        "threshold_tau": m_tau,
    }

    print(f"AUC: {auc:.4f}")
    print(f"[argmax rule]        APCER: {m_argmax['APCER']:.4f}  BPCER: {m_argmax['BPCER']:.4f}  HTER: {m_argmax['HTER']:.4f}")
    print(f"[threshold tau={tau:.4f}] APCER: {m_tau['APCER']:.4f}  BPCER: {m_tau['BPCER']:.4f}  HTER: {m_tau['HTER']:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(cm_argmax, os.path.join(save_dir, "confusion_matrix_argmax.png"),
                              title="Confusion Matrix (Test, argmax)")
        plot_confusion_matrix(cm_tau, os.path.join(save_dir, "confusion_matrix_tau.png"),
                              title="Confusion Matrix (Test, threshold tau)")
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results
