"""
Central configuration for the AAML (Asymmetric Angular Margin Learning) project.
Edit the values below directly instead of passing command-line arguments.
"""

import torch

# ----------------------------
# PATHS
# ----------------------------
# Root folder that contains one "live" folder and several attack-type folders
ROOT_DIR = r"..\path\to_folders"

# Name of the folder that holds live (bona-fide) images
LIVE_FOLDER_NAME = "Live"

# Names of the attack-type folders (Leave-One-Attack-Type-Out protocol)
SPOOF_FOLDERS = [
    "Replay",
    "Print",
    "Mask",
    "Makeup",
    "Partial",
]

# Attack type withheld from training and used only for testing (LOAO)
LEFT_OUT_ATTACK = "Partial"

# If True, every folder in SPOOF_FOLDERS is left out in turn (one independent
# training run per fold, saved in MODEL_SAVE_DIR/<attack>/). LEFT_OUT_ATTACK is ignored.
RUN_ALL_FOLDS = False

# Where checkpoints, plots and metrics are saved
MODEL_SAVE_DIR = r"..\path\to_save_the_model"

# ----------------------------
# DATA SPLIT
# ----------------------------
TEST_LIVE_FRACTION = 0.20   # fraction of live images reserved for the test set
VAL_FRACTION = 0.10         # fraction of the TRAINING set (live + known spoof) used for
                            # validation, i.e. for calibrating the threshold tau
RANDOM_SEED = 42

# ----------------------------
# TRAINING HYPER-PARAMETERS
# ----------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FEATURE_DIM = 128
NUM_WORKERS = 0
PIN_MEMORY = True
PRETRAINED = True           # ImageNet-1K pre-trained ResNet-18

# ----------------------------
# AAML LOSS
# ----------------------------
MARGIN = 0.2                # margin m (applied to the live class only)
SCALE = 30.0                # scale factor alpha

# ----------------------------
# LOGGING / VISUALISATION
# ----------------------------
SAVE_EVERY = 10             # save a checkpoint every N epochs (the last epoch is always saved)
TSNE_MAX_POINTS = 3000      # max points per plot for t-SNE

# ----------------------------
# DEVICE
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# IMAGE NORMALISATION (ImageNet stats)
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
