# EEG-Based Motor Imagery Classification using EEGNet

**Paradigm:** Active BCI — Motor Imagery Decoding  
**Dataset:** BCI Competition IV Dataset 2a  
**Task:** 4-class Classification — *Left Hand · Right Hand · Foot · Tongue*  
**Architecture:** EEGNet-8,2 (Lawhern et al., 2018)  
**Application:** Assistive Device Control (Motorised Wheelchair Navigation)  
**Framework:** PyTorch

---

## Table of Contents

1. [Overview](#overview)
2. [Neuroscientific Background](#neuroscientific-background)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Model Architecture — EEGNet](#model-architecture--eegnet)
7. [Training Configuration](#training-configuration)
8. [Evaluation & Results](#evaluation--results)
9. [Application: EEG-Controlled Wheelchair](#application-eeg-controlled-wheelchair)
10. [Usage](#usage)
11. [Dependencies](#dependencies)
12. [References](#references)

---

## Overview

This module implements a Motor Imagery (MI) Brain-Computer Interface (BCI) system capable of decoding four distinct imagined movement classes from scalp EEG recordings. The decoded neural intent is mapped to directional commands suitable for controlling a motorised wheelchair or similar assistive device, providing a non-muscular communication channel for individuals with severe motor disabilities.

The system employs **EEGNet** — a compact, parameter-efficient convolutional neural network purpose-built for EEG signal decoding — trained on the widely benchmarked **BCI Competition IV Dataset 2a**. The model's factored convolution design (Depthwise + Separable) makes it inherently generalisable across BCI paradigms, subjects, and recording configurations.

---

## Neuroscientific Background

### Motor Imagery and the Sensorimotor Cortex

Motor Imagery (MI) refers to the mental simulation of a physical movement without any overt muscular activation. During MI, the primary motor cortex (M1) and supplementary motor area (SMA) exhibit activation patterns that closely mirror actual movement execution, detectable non-invasively via scalp EEG.

The principal oscillatory signatures of MI are:

**Event-Related Desynchronisation (ERD):**  
A decrease in oscillatory power in the **mu (8–12 Hz)** and **beta (13–30 Hz)** bands contralateral to the imagined limb, beginning ~0.5–1 s after imagery onset.

$$\text{ERD\%} = \frac{P_{reference} - P_{active}}{P_{reference}} \times 100$$

**Event-Related Synchronisation (ERS):**  
A post-movement rebound in beta power (beta ERS / MRCP rebound), ipsilateral to the imagined limb.

**Somatotopic Organisation:**  
The motor homunculus encodes distinct spatial representations of body parts:

| MI Class | Primary EEG Locus | Channels |
|----------|-------------------|----------|
| Left Hand | Right central (C4, CP4) | Contralateral motor cortex |
| Right Hand | Left central (C3, CP3) | Contralateral motor cortex |
| Foot | Medial central (Cz, Pz) | Bilateral superior region |
| Tongue | Bilateral central-frontal | Inferior motor representation |

These spatial differences are the primary discriminating features exploited by EEGNet's depthwise spatial convolution.

---

## Dataset

**BCI Competition IV — Dataset 2a**

| Property | Value |
|----------|-------|
| Subjects | 9 healthy participants |
| Sessions | 2 per subject (training + evaluation) |
| Channels | 22 EEG + 3 EOG (EOG removed in preprocessing) |
| Sampling Rate | 250 Hz |
| Classes | 4 (Left Hand, Right Hand, Foot, Tongue) |
| Trials per subject | 288 (72 per class) |
| Epoch window | 4 seconds post-cue |
| File format | `.gdf` (General Data Format) |

**Trial Structure:**

```
t = 0 s      Fixation cross appears
t = 2 s      MI cue presented (arrow / symbol)
t = 2–6 s    Motor imagery task (4-second epoch extracted here)
t = 6 s      Rest period
```

**Class encoding:**

| Label | Class | Intended Command |
|-------|-------|-----------------|
| 0 | Left Hand | Turn Left |
| 1 | Right Hand | Turn Right |
| 2 | Foot | Move Forward |
| 3 | Tongue | Stop / Auxiliary |

---

## Repository Structure

```
Motor_Imagery_Detection/
│
├── EEGNet.ipynb      # Complete pipeline: download → preprocess → train → evaluate
├── wheelchair.pdf    # Reference document: EEG-based wheelchair navigation system
└── README.md
```

---

## Preprocessing Pipeline

### 1. GDF Loading via MNE-Python

BCI Competition `.gdf` files are loaded using MNE's `read_raw_gdf()` interface, which parses event triggers embedded in the file to locate MI cue onsets.

### 2. EOG Removal

The 3 EOG (electrooculography) channels are dropped prior to any analysis to prevent eye-movement artifacts from contaminating the EEG spatial filters:

```
Input: 25 channels (22 EEG + 3 EOG)  →  Output: 22 EEG channels
```

### 3. Bandpass Filtering (4–40 Hz)

A zero-phase bandpass filter retains only the neurophysiologically relevant oscillations for MI decoding:

| Band preserved | Relevance |
|----------------|-----------|
| Mu (8–12 Hz) | Primary MI marker: ERD/ERS |
| Beta (13–30 Hz) | Movement-related desynchronisation |
| Low-gamma (30–40 Hz) | High-level sensorimotor processing |

Frequencies below 4 Hz (drift, slow cortical potentials) and above 40 Hz (muscle artifacts) are rejected.

### 4. Notch Filtering (50 Hz)

Power-line interference at 50 Hz (European standard) is removed via a notch filter to prevent its harmonics from contaminating the upper beta / low-gamma band.

### 5. Epoching

4-second epochs are extracted starting at each MI cue trigger:

```
Epoch window: [cue_onset, cue_onset + 4.0 s]
Samples per epoch: 4.0 s × 250 Hz = 1000 samples
Epoch tensor shape: (N_trials, 1, 22, 1000)
```

### 6. Train / Test Split

```
Train set: 90%  (≈ 259 trials per subject)
Test set:  10%  (≈ 29 trials per subject)
```

---

## Model Architecture — EEGNet

### Design Philosophy

EEGNet (Lawhern et al., 2018) was motivated by a critical gap in EEG deep learning: most architectures at the time required large datasets (thousands of trials) and failed to generalise across BCI paradigms. EEGNet addresses this through:

1. **Extreme parameter efficiency** — achieved by factoring convolutions into temporal and spatial components.
2. **Neuroscience-informed inductive biases** — the architecture directly learns the spectrotemporal and spatial filter representations used by classical EEG feature extraction methods (FBCSP, ERD maps, MRCP).
3. **Cross-paradigm generalisability** — validated on P300, SSVEP, ERDS, and MI paradigms.

---

### Layer-by-Layer Architecture

```
Input
  Shape: (B, 1, C, T)  →  B=batch, C=22 channels, T=1000 samples (4 s × 250 Hz)
     │
     ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Block 1 — Temporal Convolution + Depthwise Spatial Convolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────────────┐
  │  Conv2d(1, F1=8, kernel=(1, 64), padding='same', bias=False) │
  │  → Learns F1=8 temporal filters at half the sampling rate    │
  │  → Equivalent to a learned FIR bandpass filter bank          │
  │  Output: (B, 8, 22, 1000)                                    │
  │                                                              │
  │  BatchNorm2d(8)                                              │
  └────────────────────────────┬─────────────────────────────────┘
                               │
  ┌────────────────────────────▼─────────────────────────────────┐
  │  DepthwiseConv2d(F1=8, F1*D=16, kernel=(C,1), groups=F1)     │
  │  D=2 depth multiplier → 16 spatial filters                   │
  │  → Learns optimal linear combination across EEG channels     │
  │  → Approximates CSP spatial filters; each filter encodes     │
  │    contralateral vs. ipsilateral cortical activity           │
  │  Output: (B, 16, 1, 1000)                                    │
  │                                                              │
  │  BatchNorm2d(16) → ELU → AvgPool2d(1,4) → Dropout(0.5)      │
  │  Output: (B, 16, 1, 250)                                     │
  └────────────────────────────┬─────────────────────────────────┘
                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Block 2 — Separable Convolution (Temporal Summary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────────────────────▼─────────────────────────────────┐
  │  DepthwiseConv2d(16, 16, kernel=(1,16), padding='same')      │
  │  + PointwiseConv2d(16, F2=16, kernel=(1,1))                  │
  │  → Decouples learning temporal patterns from combining       │
  │    feature maps — dramatically reduces parameter count       │
  │  Output: (B, 16, 1, 250)                                     │
  │                                                              │
  │  BatchNorm2d(16) → ELU → AvgPool2d(1,8) → Dropout(0.5)      │
  │  Output: (B, 16, 1, 31)                                      │
  └────────────────────────────┬─────────────────────────────────┘
                               │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────────────────────▼─────────────────────────────────┐
  │  Flatten → (B, 496)                                          │
  │  Linear(496 → 4)                                             │
  │  → Softmax → Class probabilities                             │
  └──────────────────────────────────────────────────────────────┘
```

### Correspondence to Classical EEG Feature Extraction

| EEGNet Layer | Classical Equivalent |
|---|---|
| Temporal Conv (kernel = F_s/2) | FIR bandpass filter bank (e.g., alpha, beta, gamma) |
| Depthwise Spatial Conv | Common Spatial Patterns (CSP) |
| Separable Conv (Block 2) | Log-band-power feature aggregation |
| AvgPool | Temporal smoothing / Hilbert envelope |

### Parameter Efficiency

| Model | Parameters | Paradigms Validated |
|-------|-----------|---------------------|
| EEGNet-8,2 | **~2,500** | MI, P300, SSVEP, ERN, MRCP |
| ShallowConvNet | ~18,000 | MI, oscillatory BCIs |
| DeepConvNet | ~102,000 | MI, oscillatory BCIs |
| EEGNet-16,1 | ~4,600 | Cross-paradigm |

EEGNet's compact size is critical for clinical BCIs where only tens of trials per class can be collected before patient fatigue.

---

## Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) | Adaptive gradient; well-suited for sparse EEG gradients |
| Loss function | Cross-Entropy | Standard for multiclass classification |
| Learning rate | 1 × 10⁻³ | Default Adam; effective for small EEGNet |
| Epochs | 500 | Allows full convergence given small dataset size |
| Batch size | 32 | Empirically effective for EEG trial counts (~259 train) |
| Train/Test split | 90/10 | Maximises training data for low-trial paradigm |

**Training objective:**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{4} y_{ik} \log \hat{p}_{ik}$$

where $y_{ik}$ is the one-hot ground truth and $\hat{p}_{ik} = \text{softmax}(f_\theta(x_i))_k$.

---

## Evaluation & Results

### Metrics

- **Test Accuracy** — overall proportion of correctly classified MI trials.
- **Confusion Matrix** — 4×4 matrix revealing per-class and inter-class confusion patterns (e.g., left/right hand confusions expected due to overlapping motor regions).

### Confusion Matrix Interpretation

```
                 Predicted
              LH    RH   Foot  Tongue
Actual LH   [  ·    ·     ·     ·  ]
       RH   [  ·    ·     ·     ·  ]
      Foot  [  ·    ·     ·     ·  ]
    Tongue  [  ·    ·     ·     ·  ]
```

Expected confusion pattern: LH↔RH confusion is common (bilateral motor activity overlap); Foot and Tongue are generally more separable due to distinct scalp topographies.

### BCI Competition IV 2a — Literature Benchmark

| Method | Mean Accuracy |
|--------|--------------|
| FBCSP + LDA (classical) | 68.1% |
| EEGNet-8,2 | ~72% |
| ShallowConvNet | ~73% |
| EEGNet + Transfer Learning | ~78% |
| Riemannian MDM | 66.7% |

*Accuracies across 9 subjects, 4-class, vary significantly by subject (range: ~50–90%).*

---

## Application: EEG-Controlled Wheelchair

The `wheelchair.pdf` reference document describes a real-time BCI system where decoded MI signals drive a motorised wheelchair. This represents one of the most impactful clinical applications of active BCIs.

### System Architecture

```
Scalp EEG Electrodes
        │  (22 channels, 250 Hz)
        ▼
  Amplifier & ADC
        │
        ▼
  Real-Time Preprocessing          ← Bandpass + Notch + Artefact rejection
        │
        ▼
  EEGNet Inference (< 100 ms)      ← Sliding-window classification
        │
        ▼
  Command Decoder
    ├── Left Hand   → Turn Left
    ├── Right Hand  → Turn Right
    ├── Foot        → Move Forward
    └── Tongue      → Stop / Emergency brake
        │
        ▼
  Wheelchair Motor Controller
```

### Key Latency Requirement

For safe real-time control, the full pipeline (EEG acquisition → preprocessing → inference → actuation) must complete within **~200–500 ms**. EEGNet's small parameter count (~2,500) enables inference in **< 5 ms** on CPU, well within this budget.

---

## Usage

The full pipeline is contained in `EEGNet.ipynb`. Open in **Google Colab** for GPU-accelerated training.

### Step-by-Step

```python
# Cell 1: Install dependencies
!pip install mne torch numpy matplotlib scikit-learn

# Cell 2: Download BCI Competition IV 2a dataset
# (URL and download code provided in notebook)

# Cell 3: Preprocessing — runs automatically
# Outputs: X_train (N, 1, 22, 1000), y_train (N,)

# Cell 4: Define EEGNet architecture

# Cell 5: Training loop — 500 epochs, Adam, Cross-Entropy
# Progress printed every N epochs

# Cell 6: Evaluation — test accuracy + confusion matrix plot
```

### Running Locally

```bash
jupyter notebook EEGNet.ipynb
# Or: jupyter lab EEGNet.ipynb
```

---

## Dependencies

| Package | Version (recommended) | Purpose |
|---------|----------------------|---------|
| `mne` | ≥ 1.0 | GDF file loading, epoching, filtering |
| `torch` | ≥ 2.0 | EEGNet model, training |
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Confusion matrix visualisation |
| `scikit-learn` | ≥ 1.3 | Train/test splitting, metrics |

```bash
pip install mne torch numpy matplotlib scikit-learn
```

---

## References

1. **Lawhern, V. J., et al.** (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

2. **Tangermann, M., et al.** (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55. https://doi.org/10.3389/fnins.2012.00055

3. **Pfurtscheller, G., & Lopes da Silva, F. H.** (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842–1857.

4. **Ang, K. K., et al.** (2012). Filter bank common spatial pattern algorithm on BCI competition IV datasets 2a and 2b. *Frontiers in Neuroscience*, 6, 39.

5. **Schirrmeister, R. T., et al.** (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.

6. **BCI Competition IV** — Official dataset repository: https://www.bbci.de/competition/iv/

7. **MNE-Python** — Open-source MEG/EEG analysis package: https://mne.tools
