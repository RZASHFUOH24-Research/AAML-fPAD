"""
Main training script for AAML (Asymmetric Angular Margin Learning) on the
Leave-One-Attack-Type-Out (LOAO) face presentation-attack-detection protocol.

Edit `config.py` to change paths and hyper-parameters, then simply run:
    python train.py
"""

import os
import random

import numpy as np
import torch

import config
from dataset import build_loao_loaders
from evaluate import evaluate, extract
from losses import AsymmetricAngularMarginLoss
from models import EnhancedResNet
from transforms_utils import get_test_transform, get_train_transform
from visualize import plot_cosine_hist, plot_tsne


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_fold(left_out, save_dir):
    set_seed(config.RANDOM_SEED)
    os.makedirs(save_dir, exist_ok=True)
    device = config.DEVICE

    # ---- Data ----
    train_loader, val_loader, test_loader, train_eval_loader, info = build_loao_loaders(
        root_dir=config.ROOT_DIR,
        live_folder_name=config.LIVE_FOLDER_NAME,
        spoof_folders=config.SPOOF_FOLDERS,
        left_out_attack=left_out,
        test_live_fraction=config.TEST_LIVE_FRACTION,
        val_fraction=config.VAL_FRACTION,
        train_transform=get_train_transform(),
        test_transform=get_test_transform(),
        batch_size=config.BATCH_SIZE,
        random_seed=config.RANDOM_SEED,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    print(f"\n===== Fold: left-out attack = {left_out} =====")
    print("Split info:", info)

    # ---- Model / loss / optimiser / scheduler ----
    model = EnhancedResNet(feature_dim=config.FEATURE_DIM, pretrained=config.PRETRAINED).to(device)
    criterion = AsymmetricAngularMarginLoss(
        feature_dim=config.FEATURE_DIM, scale=config.SCALE, margin=config.MARGIN
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # ---- Training loop ----
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(imgs)
            loss, _ = criterion(features, labels)
            loss.backward()
            optimizer.step()
            criterion.renormalize_centers()      # project centres back onto the unit sphere

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % config.SAVE_EVERY == 0 or epoch == config.EPOCHS - 1:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                },
                os.path.join(save_dir, f"epoch_{epoch + 1}.pth"),
            )

    # ---- Final evaluation (model after the last epoch; no checkpoint selection on test) ----
    print("\n---- Evaluation ----")
    results = evaluate(model, criterion, val_loader, test_loader, device, save_dir=save_dir)

    # ---- Visualisation: eval mode, no augmentation, L2-normalised embeddings ----
    print("Generating visualisation plots ...")
    tr_f, tr_cos, tr_y = extract(model, criterion, train_eval_loader, device)
    te_f, te_cos, te_y = extract(model, criterion, test_loader, device)

    # groups: 1 = live (train+test), 0 = known spoof (train), 2 = unseen attack (test)
    te_groups = np.where(te_y == 1, 1, 2)
    feats = np.concatenate([tr_f, te_f])
    groups = np.concatenate([tr_y, te_groups])
    cos_real = np.concatenate([tr_cos[:, 1], te_cos[:, 1]])

    plot_tsne(feats, groups, os.path.join(save_dir, "tsne.png"),
              title=f"t-SNE (unseen attack: {left_out})", max_points=config.TSNE_MAX_POINTS)
    plot_cosine_hist(cos_real, groups, os.path.join(save_dir, "cosine_hist.png"),
                     tau=results["tau_val_EER"],
                     title=f"Cosine similarity to live centre (unseen attack: {left_out})")
    return results


def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    if config.RUN_ALL_FOLDS:
        all_results = {}
        for attack in config.SPOOF_FOLDERS:
            all_results[attack] = run_fold(attack, os.path.join(config.MODEL_SAVE_DIR, attack))
        print("\n===== Summary over folds =====")
        for k in ("AUC",):
            print(f"mean {k}: {np.mean([r[k] for r in all_results.values()]):.4f}")
        for rule in ("argmax", "threshold_tau"):
            print(f"mean HTER ({rule}): "
                  f"{np.mean([r[rule]['HTER'] for r in all_results.values()]):.4f}")
    else:
        run_fold(config.LEFT_OUT_ATTACK, config.MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
