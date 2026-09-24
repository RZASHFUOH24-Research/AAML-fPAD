"""
Central configuration for the AAML (Asymmetric Angular Margin Learning) project.
Edit the values below directly instead of passing command-line arguments.
"""

import torch

# ----------------------------
# PATHS
# ----------------------------
# Root folder that contains one "live" folder and several spoof-type folders
ROOT_DIR = r"C:\Phd(Reza)\Angular-loss\Database\SiW-Mv2 coarse-grained"

# Name of the folder that holds live (bona-fide) images
LIVE_FOLDER_NAME = "Live"

# Names of the spoof-type folders (leave-one-out protocol)
SPOOF_FOLDERS = [
    "Replay",
    "Print",
    "Mask",
    "Makeup",
    "Partial",
]

# Which spoof folder to leave out for testing (Leave-One-Spoof-Out protocol)
LEFT_OUT_SPOOF = "Partial"

# Where checkpoints and result plots are saved
MODEL_SAVE_DIR = r"C:\Phd(Reza)\Angular-loss\SIW-Mv2 Ablation Results\Angular_256\Partial_Out"

# ----------------------------
# DATA SPLIT
# ----------------------------
TEST_LIVE_FRACTION = 0.20   # fraction of live images reserved for the test set
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
