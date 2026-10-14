# AAML: Asymmetric Angular Margin Learning for Open-Set Face Presentation Attack Detection

Official implementation of **"Asymmetric Angular Margin Learning for Open-Set Face Presentation Attack Detection"** (currently under review).

Face presentation attack detection (PAD) is usually treated as a closed-set binary classification problem, limiting generalisation to unseen attacks. This repository implements **AAML**, an asymmetric angular-margin discriminative learning framework that enforces a margin, and hence tight hyperspherical compactness, only on the live class, while leaving the spoof class without an equivalent constraint so that spoof representations remain distributed. This better reflects the heterogeneous nature of presentation attacks.

![AAML framework overview](assets/framework.jpg)

*Figure: Overview of the proposed Leave-One-Attack-Type-Out (LOAO) training and inference framework based on the Asymmetric Angular Margin Loss. During training, face images (live: y = 1, spoof: y = 0) are fed into an EnhancedResNet encoder (φ_I) followed by a projection head (φ_F) to produce 128-dimensional embeddings. The loss employs two learnable centres C0 (Spoof) and C1 (Live), computes the cosine similarity, subtracts an asymmetric margin (m = 0.2) from the live-class logit of live samples only, scales the logits (α = 30) and applies softmax cross-entropy. During inference, the shared encoder maps unseen test samples to the same embedding space, where classification is performed via cosine similarity to the learned centres. Performance is evaluated using AUC and HTER.*

## Highlights

- **Asymmetric margin:** the margin is applied exclusively to the live class, unlike symmetric margin losses (e.g. ArcFace, CosFace) that constrain every class equally. The margin is implemented additively on the cosine: logit = α(cos θ − m).
- **Leave-One-Attack-Type-Out (LOAO) protocol:** evaluates generalisation to attack types not seen during training.
- Reported on Oulu-NPU, CASIA-MFSD, Replay-Attack, MSU-MFSD and SiW-Mv2 (see the paper).

## Repository Structure

```
AAML-FacePAD/
├── config.py             # All paths & hyper-parameters (edit this, no CLI args needed)
├── dataset.py            # LOAO dataset / dataloader construction (train / val / test)
├── transforms_utils.py   # Train/test image transforms
├── models.py             # EnhancedResNet: ResNet-18 encoder + projection head
├── losses.py             # Asymmetric Angular Margin Loss with learnable class centres
├── visualize.py          # t-SNE, cosine-similarity histogram, confusion-matrix plots
├── evaluate.py           # AUC, APCER / BPCER / HTER (argmax rule and threshold rule)
├── train.py              # Main training entry point
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/RZASHFUOH24-Research/AAML-FacePAD.git
cd AAML-FacePAD
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation

Organise your dataset root as one live folder plus one folder per attack type:

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

This layout matches the SiW-Mv2 coarse-grained protocol used in the paper (Replay / Print / Mask / Makeup / Partial). For the fine-grained LOAO protocol, list the fine-grained sub-attack folder names in `SPOOF_FOLDERS`.

`dataset.py` builds one single-dataset LOAO fold: all images are extracted frames, and the live images are split **at image level** into train/test. If your dataset provides subject or video identities, pre-split the folders so that no subject appears in both train and test.

## Configuration

All settings live in `config.py`; no command-line arguments are needed. Most importantly:

```python
ROOT_DIR = r"path/to/your/dataset"
LIVE_FOLDER_NAME = "Live"
SPOOF_FOLDERS = ["Replay", "Print", "Mask", "Makeup", "Partial"]
LEFT_OUT_ATTACK = "Partial"        # attack type held out for testing
RUN_ALL_FOLDS = False              # True: run every attack type as the held-out one
MODEL_SAVE_DIR = r"path/to/save/outputs"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4               # cosine decay
FEATURE_DIM = 128
MARGIN = 0.2                       # m (live class only)
SCALE = 30.0                       # alpha
```

## Training

```bash
python train.py
```

This will:

1. Build the LOAO split: train / validation (known attacks + live, stratified) / test (unseen attack + held-out live).
2. Train the EnhancedResNet encoder jointly with the learnable class centres using the AAML loss (Adam, cosine LR decay; centres are re-normalised after every step).
3. Save checkpoints to `MODEL_SAVE_DIR` (every `SAVE_EVERY` epochs and the last epoch).
4. Calibrate the threshold τ on the validation set at the EER point, then evaluate on the test set and report **AUC**, and **APCER / BPCER / HTER** both with the argmax rule and with the threshold rule (cosine similarity to the live centre ≥ τ). Metrics are saved to `metrics.json`, together with confusion matrices.
5. Save t-SNE (L2-normalised embeddings, eval mode, no augmentation) and cosine-similarity histogram plots, showing live, known spoof and unseen-attack samples.

## Results

Example results reported in the paper (see the manuscript for full tables):

| Dataset | Metric | Value |
|---|---|---|
| Oulu-NPU (complete protocol) | ACER | 0.32% |
| CASIA-MFSD / Replay-Attack / MSU-MFSD (cross-type) | Avg. AUC | 99.34% |
| SiW-Mv2 (fine-grained LOAO) | Avg. HTER / AUC | 4.67% / 97.40% |
| SiW-Mv2 (coarse-grained) | Avg. HTER / AUC | 6.25% / 98.01% |

> The Oulu-NPU and cross-dataset protocols require their own data organisation and are not covered by the single-dataset loader provided here.

![t-SNE comparison across loss functions](assets/tsne_comparison.jpg)

*Figure: t-SNE visualisation of feature embeddings across five leave-one-out protocols. Rows correspond to (a) Binary Cross-Entropy (BCE), (b) Deep SVDD, (c) ArcFace, and (d) the proposed AAML. Blue: live samples, red: known spoofs used for training, green: unseen attacks used for testing. While ArcFace (row c) encourages symmetric compactness, AAML (row d) allows spoof samples to form a more distributed manifold, giving clearer separation between unseen attacks and the live cluster.*

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
