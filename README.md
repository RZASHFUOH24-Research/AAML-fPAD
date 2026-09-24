# AAML: Asymmetric Angular Margin Learning for Open-Set Face Presentation Attack Detection

Official implementation of **"Asymmetric Angular Margin Learning for Open-Set Face Presentation Attack Detection"** (currently under review).

> Face presentation attack detection (PAD) is usually treated as a closed-set binary
> classification problem, limiting generalisation to unseen attacks. This repository
> implements **AAML**, an asymmetric angular-margin discriminative learning framework
> that enforces tight hyperspherical compactness **only** on the live class, while
> allowing spoof representations to remain distributed — better reflecting the
> heterogeneous nature of presentation attacks.

<p align="center">
  <img src="figures/framework.jpg" width="800" alt="AAML framework overview">
</p>

## Highlights

- **Asymmetric angular margin**: margin applied exclusively to the live class, unlike symmetric losses (e.g., ArcFace) that constrain all classes equally.
- **Leave-One-Spoof-Out (LOSO) protocol**: evaluates generalisation to unseen attack types not seen during training.
- State-of-the-art results on **Oulu-NPU**, **CASIA-MFSD**, **Replay-Attack**, **MSU-MFSD**, and **SiW-Mv2**.

## Repository Structure

```
AAML-FacePAD/
├── config.py            # All paths & hyper-parameters (edit this, no CLI args needed)
├── dataset.py            # Leave-One-Spoof-Out dataset / dataloader construction
├── transforms_utils.py   # Train/test image transforms
├── models.py              # EnhancedResNet50 backbone + projection head
├── losses.py              # Angular loss with learnable class centres
├── visualize.py           # Feature scatter, t-SNE, confusion-matrix plots
├── evaluate.py            # APCER / BPCER / ACER evaluation
├── train.py               # Main training entry point
├── requirements.txt
├── assets/                # Figures used in this README
└── results/               # Checkpoints & output plots are saved here (gitignored)
```

## Installation

```bash
git clone https://github.com/<your-username>/AAML-FacePAD.git
cd AAML-FacePAD
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation

Organise your dataset root as one **live** folder plus one folder per **spoof type**:

```
<ROOT_DIR>/
├── Live/
│   ├── img_0001.jpg
│   └── ...
├── Replay/
├── Print/
├── Mask/
├── Makeup/
└── Partial/
```

This layout matches the **SiW-Mv2 coarse-grained** protocol used in the paper
(Replay / Print / Mask / Makeup / Partial). For the fine-grained LOO protocol,
simply list the fine-grained sub-attack folder names in `SPOOF_FOLDERS`.

## Configuration

All settings live in `config.py` — no command-line arguments needed. Edit the values
you need before running training, most importantly:

```python
ROOT_DIR = r"path/to/your/dataset"
LIVE_FOLDER_NAME = "Live"
SPOOF_FOLDERS = ["Replay", "Print", "Mask", "Makeup", "Partial"]
LEFT_OUT_SPOOF = "Partial"          # attack type held out for testing
MODEL_SAVE_DIR = r"path/to/save/checkpoints"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
FEATURE_DIM = 128
```

## Training

```bash
python train.py
```

This will:
1. Build the Leave-One-Spoof-Out train/test split.
2. Train the `EnhancedResNet50` encoder jointly with the learnable angular class centres.
3. Save a checkpoint per epoch to `MODEL_SAVE_DIR`.
4. Generate feature-scatter and t-SNE plots for the final epoch.
5. Run test-set evaluation and report **APCER / BPCER / ACER**, plus a confusion matrix.

## Results

Example results reported in the paper (see manuscript for full tables):

| Dataset | Metric | Value |
|---|---|---|
| Oulu-NPU (complete protocol) | ACER | **0.32%** |
| CASIA-MFSD / Replay-Attack / MSU-MFSD (cross-type) | Avg. AUC | **99.34%** |
| SiW-Mv2 (fine-grained LOO) | Avg. HTER / AUC | **4.67% / 97.40%** |
| SiW-Mv2 (coarse-grained) | Avg. HTER / AUC | **6.25% / 98.01%** |

<p align="center">
  <img src="assets/tsne_comparison.png" width="700" alt="t-SNE comparison across loss functions">
</p>

## Citation

If you use this code, please cite our paper (citation details will be added once published):

```bibtex
@article{sheikhfathollahi2026aaml,
  title   = {Asymmetric Angular Margin Learning for Open-Set Face Presentation Attack Detection},
  author  = {Sheikhfathollahi, Mohammadreza and Parkinson, Simon and Khan, Saad},
  journal = {Under Review},
  year    = {2026}
}
```
