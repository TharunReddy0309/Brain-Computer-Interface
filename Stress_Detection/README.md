# EEG-Based Mental Stress Detection using ShallowConvNet

**Paradigm:** Passive BCI — Mental State Decoding  
**Dataset:** SAM40 (Stress and Mental Arithmetic Dataset)  
**Task:** Binary Classification — *Relax (0)* vs. *Cognitive Stress (1)*  
**Architecture:** ShallowConvNet (Schirrmeister et al., 2017)  
**Framework:** PyTorch

---

## Table of Contents

1. [Overview](#overview)
2. [Neuroscientific Motivation](#neuroscientific-motivation)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Model Architecture](#model-architecture)
7. [Training Strategy](#training-strategy)
8. [Evaluation Protocol](#evaluation-protocol)
9. [Usage](#usage)
10. [Dependencies](#dependencies)
11. [References](#references)

---

## Overview

This module implements a complete end-to-end pipeline for automatic detection of mental stress from scalp EEG signals. The system processes multi-channel EEG recordings, applies a principled artifact removal pipeline, extracts temporal segments via a sliding-window approach, and classifies them using a Shallow Convolutional Neural Network optimised with modern regularisation and data-augmentation techniques.

The pipeline is designed to address three core challenges inherent to EEG-based affective computing:
- **Non-stationarity** of EEG signals across recording sessions and subjects.
- **Class imbalance** arising from unequal segment counts per cognitive state.
- **Overfitting risk** given the high-dimensional, low-sample-count nature of EEG data.

---

## Neuroscientific Motivation

Mental stress induces measurable changes in the oscillatory dynamics of cortical EEG activity. The primary neurophysiological markers exploited in this work are:

| Frequency Band | Range | Relevance to Stress |
|----------------|-------|---------------------|
| **Alpha (α)** | 8–13 Hz | Alpha power decreases (ERD) under cognitive load — strong negative correlate of stress |
| **Beta (β)** | 13–30 Hz | Beta synchronisation (ERS) increases under focused attention and mental arithmetic |
| **Theta (θ)** | 4–8 Hz | Frontal theta increases reflect working memory engagement during stressors |
| **Gamma (γ)** | 30–40 Hz | Associated with high-level cognitive processing under stress conditions |

Arithmetic tasks (Stroop, mental calculation) are established paradigms for inducing sustained cognitive stress in laboratory settings, providing clean ground-truth labels compared to naturalistic stressors.

---

## Dataset

**SAM40 — Stress and Mental Arithmetic Dataset**

| Property | Value |
|----------|-------|
| Subjects | 40 healthy participants |
| Channels | 32 (10-20 International System) |
| Sampling Rate | 128 Hz |
| Tasks | Relax (baseline) / Arithmetic (Stress induction) |
| File Format | `.mat` (MATLAB) |
| Labelling | Filename-based (`relax` / `arithmetic` keywords) |

**Channel layout (32-ch, 10-20 system):**
```
Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz,
O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz
```

---

## Repository Structure

```
Stress_Detection/
│
├── eeg_clean.py          # Stage 1: EEG artifact removal (Bandpass + FastICA)
├── compare.py            # Stage 2: Quality assurance — cleaning validation
├── graph_plot.py         # Stage 3: MNE-based interactive signal visualisation
├── classify.py           # Stage 4: Segmentation → ShallowConvNet → K-Fold CV
│
├── BTP-Evaluation1.pdf   # BTP Mid-Semester Evaluation Report
├── BTP-Evaluvation2.pdf  # BTP End-Semester Evaluation Report
│
└── README.md
```

---

## Preprocessing Pipeline

### Stage 1 — Bandpass Filtering (`eeg_clean.py`)

A 4th-order zero-phase **Butterworth bandpass filter** (0.5–40 Hz) is applied channel-wise using `sosfiltfilt` (SOS cascade, forward-backward pass) to eliminate:
- DC offset and slow drift (< 0.5 Hz).
- High-frequency muscle artefacts (> 40 Hz).

```python
sos = butter(4, [l / nyq, h / nyq], btype='bandpass', output='sos')
filtered = sosfiltfilt(sos, channel_data)
```

### Stage 2 — Artifact Removal via Independent Component Analysis (ICA)

Ocular (EOG) and muscular (EMG) artifacts are suppressed using a custom **FastICA** implementation with deflation-based orthogonalisation:

**Whitening:**

$$\mathbf{Z} = \mathbf{W}_{white} \cdot \mathbf{X}_{centered}, \quad \mathbf{W}_{white} = \frac{\mathbf{E}}{\sqrt{\boldsymbol{\lambda}}}$$

where $\mathbf{E}$ are eigenvectors and $\boldsymbol{\lambda}$ are eigenvalues of the data covariance matrix.

**FastICA (negentropy maximisation, tanh non-linearity):**

$$\mathbf{w}_{i} \leftarrow \mathbb{E}\{\mathbf{z} \cdot g(\mathbf{w}_{i}^T\mathbf{z})\} - \mathbb{E}\{g'(\mathbf{w}_{i}^T\mathbf{z})\}\mathbf{w}_{i}$$

where $g(u) = \tanh(u)$ approximates negentropy.

**Artifact Identification — Kurtosis Criterion:**

$$\text{Component } i \text{ is artifact} \iff |\kappa_4(s_i)| > \tau, \quad \tau = 5.0$$

High excess kurtosis ($\kappa_4 > 5$) indicates non-Gaussian, impulsive sources (blinks, saccades, EMG bursts). Identified components are zeroed before signal reconstruction.

**Reconstruction:**

$$\hat{\mathbf{X}} = \mathbf{A} \cdot \hat{\mathbf{S}} + \boldsymbol{\mu}_X, \quad \mathbf{A} = (\mathbf{W} \cdot \mathbf{W}_{white})^{+}$$

Cleaned files are written to `NEW_PY/<filename>_cleaned.mat`.

### Stage 3 — Cleaning Validation (`compare.py`)

Three signal quality metrics are computed and visualised:

| Metric | Formula |
|--------|---------|
| Overall STD | $\sigma_{global} = \text{std}(\mathbf{X}_{all})$ |
| Mean Channel STD | $\bar{\sigma}_{ch} = \frac{1}{C}\sum_{c=1}^{C}\text{std}(\mathbf{x}_c)$ |
| Peak Amplitude | $A_{peak} = \max|\mathbf{X}|$ |

### Stage 4 — Sliding-Window Segmentation (`classify.py`)

Continuous EEG recordings are converted to fixed-length segments:

| Parameter | Value |
|-----------|-------|
| Window duration | 2.0 seconds |
| Window size (`CHUNK_SIZE`) | 640 samples (@ 320 Hz) |
| Stride | 320 samples (50% overlap) |

Each segment inherits the label of its source recording. Z-score normalisation is applied **per-channel** using only training-set statistics:

$$\hat{x}_{c,t} = \frac{x_{c,t} - \mu_c^{train}}{\sigma_c^{train} + \epsilon}, \quad \epsilon = 10^{-8}$$

---

## Model Architecture

### ShallowConvNet

ShallowConvNet (Schirrmeister et al., 2017) is a biologically motivated CNN that approximates filter-bank Common Spatial Patterns (FBCSP) in a single learnable architecture. It operates directly on raw EEG windows without hand-crafted spectral features.

```
Input: (Batch, 1, C, T)   →   C=32 channels, T=640 time samples
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Temporal Convolution                               │
│  Conv2d(1, 40, kernel=(1, 25), bias=False)          │  → learns band-specific temporal filters
│  Output: (B, 40, 32, 616)                           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Spatial Convolution (Depthwise-style)              │
│  Conv2d(40, 40, kernel=(32, 1), bias=False)         │  → spatial filtering across channels
│  BatchNorm2d(40)                                    │
│  Output: (B, 40, 1, 616)                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Non-linearity + Pooling                            │
│  x² (square activation — power envelope)           │
│  AvgPool2d(kernel=(1,75), stride=(1,15))            │  → temporal averaging / log-power
│  log(clip(x, min=1e-7))                             │
│  BatchNorm2d(40)                                    │
│  Dropout(0.5)                                       │
│  Flatten → (B, D)                                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Fully Connected Classifier                         │
│  Linear(D → 2)                                      │  → Relax / Stress logits
└─────────────────────────────────────────────────────┘
```

**Biological interpretation of each stage:**

| Layer | Signal Processing Equivalent |
|-------|------------------------------|
| Temporal Conv (1 × 25) | Band-pass FIR filter bank |
| Spatial Conv (32 × 1) | Spatial filter (CSP-like, learns inter-channel covariance structure) |
| Square + Pool + Log | Log-band-power feature extraction — approximates Welch PSD features |
| Dropout | Regularisation; prevents reliance on single channels/components |

**Parameter count:** ~18,000 — deliberately compact to reduce overfitting risk on small EEG datasets.

---

## Training Strategy

### Loss Function — Label-Smoothing Cross-Entropy

Hard one-hot targets are replaced with soft targets to prevent overconfidence:

$$\tilde{y}_k = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{K} & k = y \\ \frac{\varepsilon}{K} & k \neq y \end{cases}, \quad \varepsilon = 0.1,\ K = 2$$

$$\mathcal{L}_{LS} = -\sum_{k} \tilde{y}_k \cdot \log p_k$$

Class-balanced weighting $w_k$ is folded into the loss to account for segment count imbalance:

$$w_k = \text{clip}\!\left(\frac{N}{K \cdot N_k},\ \frac{1}{4},\ 4\right)$$

### Data Augmentation — Mixup

With probability 0.5 per batch, Mixup augmentation is applied:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha),\ \alpha=0.3$$

$$\mathcal{L}_{mix} = \lambda \mathcal{L}(\tilde{x}, y_i) + (1-\lambda)\mathcal{L}(\tilde{x}, y_j)$$

Mixup encourages linear behaviour in the latent space and reduces sensitivity to adversarial perturbations.

### Optimiser & Scheduler

| Component | Configuration |
|-----------|--------------|
| Optimiser | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 5 × 10⁻⁴ |
| LR Schedule | Cosine Annealing with Warm Restarts (T₀=50) |
| Early stopping | Patience = 30 epochs |
| Gradient clipping | max_norm = 1.0 |

**Cosine Annealing:**

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{\pi t}{T_0}\right)$$

---

## Evaluation Protocol

**Stratified K-Fold Cross-Validation** (K = 5) is applied over all extracted segments. Stratification ensures each fold maintains the original Relax/Stress class ratio.

> **Note:** Because segments from the same recording can appear in both train and test folds, this measures *within-distribution* (intra-subject) performance. For subject-independent estimates, `StratifiedGroupKFold` with subject IDs as groups (commented in `classify.py`) provides a stricter, more clinically realistic benchmark.

**Reported metrics per fold and aggregated:**

| Metric | Formula |
|--------|---------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| Balanced Accuracy | $\frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$ |
| Macro F1-Score | $\frac{1}{K}\sum_k \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$ |

**Dynamic Threshold Tuning:**  
The decision threshold $\tau$ is swept over [0.20, 0.80] on each validation fold and the value maximising Macro-F1 is selected, mitigating class-imbalance bias from the default 0.5 threshold.

---

## Usage

### Environment Setup

```bash
pip install numpy scipy torch scikit-learn mne matplotlib
```

### Step 1 — Clean Raw EEG

```python
from eeg_clean import clean_eeg
clean_eeg("path/to/raw_sub_N_relax_trial1.mat", sfreq=128.0)
# Output → NEW_PY/raw_sub_N_relax_trial1_cleaned.mat
```

### Step 2 — Validate Cleaning Quality (Optional)

```bash
python compare.py
# Edit the __main__ block to point to your cleaned files
```

### Step 3 — Visualise EEG (Optional)

```bash
python graph_plot.py
# Edit the loadmat path to your cleaned file
```

### Step 4 — Train and Evaluate

```python
# For Google Colab (recommended — GPU acceleration):
# Mount drive, set DATA_DIR, then run all cells in classify.py

# For local execution:
# Comment out drive.mount lines and set:
DATA_DIR = '/path/to/your/filtered_data'
```

```bash
python classify.py
```

### Expected Console Output

```
════════════════════════════════════════════════════════════
5-FOLD CROSS-VALIDATION SUMMARY
════════════════════════════════════════════════════════════
  Accuracy      : XX.XX% ± X.XX%
  Balanced-Acc  : XX.XX% ± X.XX%
  Macro-F1      : X.XXXX ± X.XXXX

  Per-fold breakdown:
  Fold   Acc    Bal-Acc  Macro-F1
  ──────────────────────────────────
  1     XX.XX%  XX.XX%   X.XXXX
  ...
════════════════════════════════════════════════════════════
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Signal processing, ICA, `.mat` I/O |
| `torch` | ShallowConvNet model & training |
| `scikit-learn` | Cross-validation, metrics |
| `mne` | EEG data structure, interactive visualisation |
| `matplotlib` | Quality comparison plots |

---

## References

1. **Schirrmeister, R. T., et al.** (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420. https://doi.org/10.1002/hbm.23730

2. **Hyvärinen, A., & Oja, E.** (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430.

3. **Zhang, H., et al.** (2018). mixup: Beyond empirical risk minimization. *International Conference on Learning Representations (ICLR)*.

4. **Müller, R., Kornblith, S., & Hinton, G.** (2019). When does label smoothing help? *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

5. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*.

6. **SAM40 Dataset** — EEG recordings for stress/relaxation classification. 40 subjects, 32-channel, 128 Hz.
