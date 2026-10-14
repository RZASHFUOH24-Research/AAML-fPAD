"""
Dataset utilities implementing the Leave-One-Attack-Type-Out (LOAO) protocol
used for the open-set Face Presentation Attack Detection experiments.

Labels: 1 = live (bona fide), 0 = spoof (attack).
"""

from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class PathDataset(Dataset):
    """Simple dataset that loads images from a list of file paths."""

    def __init__(self, paths, labels, transform=None):
        assert len(paths) == len(labels)
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _list_images(folder):
    return sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)


def build_loao_loaders(
    root_dir,
    live_folder_name,
    spoof_folders,
    left_out_attack,
    test_live_fraction=0.2,
    val_fraction=0.1,
    train_transform=None,
    test_transform=None,
    batch_size=32,
    random_seed=42,
    num_workers=0,
    pin_memory=True,
):
    """
    Builds the loaders for one LOAO fold:
      - `left_out_attack` is used exclusively for testing (unseen attack).
      - All other attack folders are known attacks used for training/validation.
      - Live images are split into train and test using `test_live_fraction`.
      - `val_fraction` of the training set (stratified) is held out for validation
        (used only to calibrate the threshold tau). It contains NO unseen attack.

    Returns: train_loader, val_loader, test_loader, train_eval_loader, info
      train_eval_loader = training samples with the test transform and no shuffling
      (used only to extract clean embeddings for visualisation).
    """
    if left_out_attack not in spoof_folders:
        raise ValueError(f"left_out_attack '{left_out_attack}' not in spoof_folders {spoof_folders}")

    root = Path(root_dir)
    live_dir = root / live_folder_name
    if not live_dir.exists():
        raise FileNotFoundError(f"Live folder not found: {live_dir}")
    for sf in spoof_folders:
        if not (root / sf).exists():
            raise FileNotFoundError(f"Attack folder not found: {root / sf}")

    # ---- Live ----
    live_paths = _list_images(live_dir)
    train_live, test_live = train_test_split(
        live_paths, test_size=test_live_fraction, random_state=random_seed, shuffle=True
    )

    # ---- Attacks ----
    known_paths, unseen_paths = [], []
    for sf in spoof_folders:
        images = _list_images(root / sf)
        (unseen_paths if sf == left_out_attack else known_paths).extend(images)

    # ---- Train(+val) / test ----
    trainval_paths = known_paths + train_live
    trainval_labels = [0] * len(known_paths) + [1] * len(train_live)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        trainval_paths, trainval_labels,
        test_size=val_fraction, random_state=random_seed, shuffle=True, stratify=trainval_labels,
    )

    test_paths = unseen_paths + test_live
    test_labels = [0] * len(unseen_paths) + [1] * len(test_live)

    train_ds = PathDataset(train_paths, train_labels, transform=train_transform)
    train_eval_ds = PathDataset(train_paths, train_labels, transform=test_transform)
    val_ds = PathDataset(val_paths, val_labels, transform=test_transform)
    test_ds = PathDataset(test_paths, test_labels, transform=test_transform)

    kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    # drop_last avoids a final batch of size 1, which breaks BatchNorm1d in train mode
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    info = {
        "left_out_attack": left_out_attack,
        "n_train": len(train_ds),
        "n_train_live": int(sum(train_labels)),
        "n_train_spoof": len(train_labels) - int(sum(train_labels)),
        "n_val": len(val_ds),
        "n_val_live": int(sum(val_labels)),
        "n_val_spoof": len(val_labels) - int(sum(val_labels)),
        "n_test": len(test_ds),
        "n_test_live": len(test_live),
        "n_test_unseen_spoof": len(unseen_paths),
    }
    return train_loader, val_loader, test_loader, train_eval_loader, info
