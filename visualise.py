"""
Plotting utilities: raw feature scatter, t-SNE embedding visualisation,
and confusion-matrix display.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_features(epoch_features, epoch_labels, epoch, save_path=None):
    features_tensor = torch.cat(epoch_features, dim=0).cpu().detach().numpy()
    labels_tensor = torch.cat(epoch_labels, dim=0).cpu().detach().numpy()

    plt.figure(figsize=(8, 8))
    if features_tensor.shape[1] >= 2:
        plt.scatter(features_tensor[labels_tensor == 1, 0], features_tensor[labels_tensor == 1, 1],
                    c='blue', label='Real', alpha=0.5)
        plt.scatter(features_tensor[labels_tensor == 0, 0], features_tensor[labels_tensor == 0, 1],
                    c='red', label='Spoof', alpha=0.5)
    else:
        plt.scatter(np.arange(len(features_tensor))[labels_tensor == 1], features_tensor[labels_tensor == 1, 0],
                    c='blue', label='Real', alpha=0.5)
        plt.scatter(np.arange(len(features_tensor))[labels_tensor == 0], features_tensor[labels_tensor == 0, 0],
                    c='red', label='Spoof', alpha=0.5)

    plt.title(f'Epoch {epoch + 1} - Feature Scatter')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_tsne(epoch_features, epoch_labels, epoch, save_path=None, perplexity=30):
    features_tensor = torch.cat(epoch_features, dim=0).cpu().detach().numpy()
    labels_tensor = torch.cat(epoch_labels, dim=0).cpu().detach().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_tensor)

    plt.figure(figsize=(8, 8))
    plt.scatter(features_2d[labels_tensor == 1, 0], features_2d[labels_tensor == 1, 1],
                c='blue', label='Real', alpha=0.5)
    plt.scatter(features_2d[labels_tensor == 0, 0], features_2d[labels_tensor == 0, 1],
                c='red', label='Spoof', alpha=0.5)
    plt.title(f'Epoch {epoch + 1} - t-SNE Feature Visualization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Spoof", "Real"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return cm
