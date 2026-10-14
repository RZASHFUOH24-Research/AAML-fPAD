"""
Plotting utilities: t-SNE of normalised embeddings, cosine-similarity histograms
(live-centre score), and confusion-matrix display.

Group codes used for plots: 1 = Live, 0 = Known spoof (training), 2 = Unseen attack (test).
"""

import matplotlib
matplotlib.use("Agg")   # figures are only saved to disk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

GROUPS = {
    1: ("Live", "blue"),
    0: ("Known spoof", "red"),
    2: ("Unseen attack", "green"),
}


def _subsample(features, groups, max_points, seed=42):
    if len(features) <= max_points:
        return features, groups
    idx = np.random.RandomState(seed).choice(len(features), max_points, replace=False)
    return features[idx], groups[idx]


def plot_tsne(features, groups, save_path, title="t-SNE of normalised embeddings",
              perplexity=30, max_points=3000):
    features, groups = _subsample(np.asarray(features), np.asarray(groups), max_points)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42).fit_transform(features)

    plt.figure(figsize=(8, 8))
    for g, (name, color) in GROUPS.items():
        m = groups == g
        if m.any():
            plt.scatter(emb[m, 0], emb[m, 1], c=color, label=name, alpha=0.5, s=10)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_cosine_hist(cos_real, groups, save_path, tau=None,
                     title="Cosine similarity to the live centre"):
    cos_real, groups = np.asarray(cos_real), np.asarray(groups)
    plt.figure(figsize=(8, 5))
    for g, (name, color) in GROUPS.items():
        m = groups == g
        if m.any():
            plt.hist(cos_real[m], bins=50, range=(-1, 1), alpha=0.5, color=color,
                     label=name, density=True)
    if tau is not None:
        plt.axvline(tau, color="black", linestyle="--", label=f"tau = {tau:.3f}")
    plt.xlabel(r"$\cos(\theta_{real})$")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix (Test)"):
    """`cm` is a 2x2 confusion matrix with label order [Spoof, Real]."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Spoof", "Real"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
