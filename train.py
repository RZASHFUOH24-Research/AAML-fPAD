"""
Main training script for AAML (Asymmetric Angular Margin Learning) on the
Leave-One-Spoof-Out (LOSO) face presentation-attack-detection protocol.

Edit `config.py` to change paths and hyper-parameters, then simply run:
    python train.py
"""

import os
import torch

import config
from dataset import build_leave_one_spoof_loaders
from transforms_utils import get_train_transform, get_test_transform
from models import EnhancedResNet50
from losses import AngularLoss
from visualize import plot_features, plot_tsne
from evaluate import evaluate


def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # ---- Data ----
    train_loader, test_loader, info = build_leave_one_spoof_loaders(
        root_dir=config.ROOT_DIR,
        live_folder_name=config.LIVE_FOLDER_NAME,
        spoof_folders=config.SPOOF_FOLDERS,
        left_out_spoof=config.LEFT_OUT_SPOOF,
        test_live_fraction=config.TEST_LIVE_FRACTION,
        train_transform=get_train_transform(),
        test_transform=get_test_transform(),
        batch_size=config.BATCH_SIZE,
        random_seed=config.RANDOM_SEED,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    print("Split info:", info)

    # ---- Model / loss / optimiser ----
    device = config.DEVICE
    model = EnhancedResNet50(feature_dim=config.FEATURE_DIM, pretrained=True).to(device)
    criterion = AngularLoss(feature_dim=config.FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=config.LEARNING_RATE
    )

    # ---- Training loop ----
    for epoch in range(config.EPOCHS):
        model.train()
        all_features, all_labels = [], []
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(imgs)
            loss, _ = criterion(features, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_features.append(features.detach().cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

        # ---- Checkpoint ----
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch + 1}.pth"),
        )

        # ---- Visualisation only on the final epoch ----
        if epoch == config.EPOCHS - 1:
            scatter_path = os.path.join(config.MODEL_SAVE_DIR, f"scatter_epoch_{epoch + 1}.png")
            tsne_path = os.path.join(config.MODEL_SAVE_DIR, f"tsne_epoch_{epoch + 1}.png")

            print(f"Generating visualization plots for epoch {epoch + 1} ...")
            plot_features(all_features, all_labels, epoch, save_path=scatter_path)
            plot_tsne(all_features, all_labels, epoch, save_path=tsne_path)

    # ---- Final test evaluation ----
    print("\n---- Test Evaluation ----")
    evaluate(model, criterion, test_loader, device, save_dir=config.MODEL_SAVE_DIR)


if __name__ == "__main__":
    main()
