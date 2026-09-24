"""
Dataset utilities implementing the Leave-One-Spoof-Out (LOSO) protocol
used for the open-set Face Presentation Attack Detection experiments.
"""

import random
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class PathDataset(Dataset):
    """Simple dataset that loads images from a list of file paths."""

    def __init__(self, paths, labels, transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_leave_one_spoof_loaders(
    root_dir,
    live_folder_name,
    spoof_folders,
    left_out_spoof,
    test_live_fraction=0.2,
    train_transform=None,
    test_transform=None,
    batch_size=32,
    random_seed=42,
    num_workers=0,
    pin_memory=True,
):
    """
    Builds train/test DataLoaders following the Leave-One-Spoof-Out (LOSO) protocol:
    - The `left_out_spoof` folder is used exclusively for testing (unseen attack).
    - All other spoof folders are used for training (known attacks).
    - Live images are split between train and test using `test_live_fraction`.
    """
    root = Path(root_dir)
    live_dir = root / live_folder_name
    if not live_dir.exists():
        raise FileNotFoundError(f"Live folder not found: {live_dir}")

    for sf in spoof_folders:
        if not (root / sf).exists():
            raise FileNotFoundError(f"Spoof folder not found: {(root / sf)}")

    # ---- Live images ----
    live_paths = sorted(
        str(p) for p in live_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )
    train_live, test_live = train_test_split(
        live_paths, test_size=test_live_fraction, random_state=random_seed, shuffle=True
    )

    # ---- Spoof images ----
    spoof_train_paths, spoof_train_labels = [], []
    spoof_test_paths, spoof_test_labels = [], []

    for sf in spoof_folders:
        sf_dir = root / sf
        images = sorted(
            str(p) for p in sf_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
        )
        if sf == left_out_spoof:
            spoof_test_paths.extend(images)
            spoof_test_labels.extend([0] * len(images))
        else:
            spoof_train_paths.extend(images)
            spoof_train_labels.extend([0] * len(images))

    # ---- Combine (label 1 = live, label 0 = spoof) ----
    train_paths = spoof_train_paths + train_live
    train_labels = spoof_train_labels + [1] * len(train_live)

    test_paths = spoof_test_paths + test_live
    test_labels = spoof_test_labels + [1] * len(test_live)

    # ---- Shuffle ----
    random.seed(random_seed)

    combined_train = list(zip(train_paths, train_labels))
    random.shuffle(combined_train)
    train_paths, train_labels = zip(*combined_train) if combined_train else ([], [])

    combined_test = list(zip(test_paths, test_labels))
    random.shuffle(combined_test)
    test_paths, test_labels = zip(*combined_test) if combined_test else ([], [])

    train_dataset = PathDataset(list(train_paths), list(train_labels), transform=train_transform)
    test_dataset = PathDataset(list(test_paths), list(test_labels), transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    info = {
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_train_live": len(train_live),
        "n_test_live": len(test_live),
        "n_train_spoof": len(spoof_train_paths),
        "n_test_spoof": len(spoof_test_paths),
        "left_out_spoof": left_out_spoof,
    }

    return train_loader, test_loader, info
