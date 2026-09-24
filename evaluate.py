"""
Evaluation routine: runs the trained model on the test set, plots the
confusion matrix, and reports APCER / BPCER / ACER (ISO/IEC 30107-3 metrics).
"""

import os
import numpy as np
import torch

from visualize import plot_confusion_matrix


def evaluate(model, criterion, dataloader, device, save_dir=None):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            features = model(imgs)
            _, cosine = criterion(features, labels)

            preds = torch.argmax(cosine, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm_save_path = os.path.join(save_dir, "confusion_matrix.png") if save_dir else None
    cm = plot_confusion_matrix(y_true, y_pred, save_path=cm_save_path)

    TN, FP, FN, TP = cm.ravel()

    apcer = FP / (TN + FP + 1e-6)   # attack misclassified as real
    bpcer = FN / (TP + FN + 1e-6)   # bona fide misclassified as spoof
    acer = (apcer + bpcer) / 2

    print(f"APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACER: {acer:.4f}")
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}
