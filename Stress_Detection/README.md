# SAM40 EEG Stress Classification

This project focuses on processing and classifying EEG signals from the SAM40 dataset for Mental Stress vs Relax state recognition using signal processing and deep learning techniques.

---

## Project Overview

The pipeline includes:

- EEG signal cleaning and artifact removal
- Signal visualization
- EEG quality comparison
- Stress vs Relax classification using Deep Learning

The dataset contains EEG recordings captured under two conditions:

- Relax State
- Stress State (Arithmetic task)

---

# Project Structure

├── eeg_clean.py
├── compare.py
├── graph_plot.py
├── classify.py
└── NEW_PY/

---

# 1. EEG Cleaning (`eeg_clean.py`)

This script performs automated EEG preprocessing and artifact removal.

## Features

### Bandpass Filtering
- 4th-order Butterworth Filter
- Frequency range: **0.5 Hz – 40 Hz**

### Artifact Removal using ICA
- Uses FastICA decomposition
- Detects noisy components using Kurtosis threshold
- Removes eye blink and muscle artifacts

### Output
- Saves cleaned EEG `.mat` files
- Output directory:
  ```bash
  NEW_PY/
  ```

---

# 2. EEG Quality Comparison (`compare.py`)

Compares Python-cleaned EEG data against MATLAB-cleaned EEG data.

## Metrics Used

- Overall Standard Deviation
- Mean Channel Standard Deviation
- Peak Amplitude

## Visualization

- Generates side-by-side bar plots using Matplotlib
- Helps validate cleaning effectiveness

---

# 3. EEG Visualization (`graph_plot.py`)

Visualizes multi-channel EEG signals using MNE-Python.

## Features

- Loads cleaned EEG `.mat` files
- Assigns standard 10-20 EEG channel names
- Interactive EEG plotting

## Sampling Frequency

```python
128 Hz
```

---

# 4. EEG Classification (`classify.py`)

Deep learning pipeline for classifying EEG signals into:

- Relax (0)
- Stress (1)

---

## Preprocessing

### Normalization
- Z-score normalization applied channel-wise

### Sliding Window Segmentation
- 2-second overlapping EEG windows

---

## Model Architecture

### ShallowConvNet (PyTorch)

The model includes:

- Temporal Convolutions
- Spatial Convolutions
- EEG-specific feature extraction

---

## Training Techniques

### Mixup Augmentation
Improves generalization by blending samples and labels.

### Label Smoothing
Reduces overconfidence and overfitting.

### Weighted Sampling
Handles class imbalance.

### Cosine Annealing
Smooth learning rate decay for stable convergence.

---

## Evaluation Metrics

- Accuracy
- Balanced Accuracy
- Macro F1-Score

Uses:

```python
Stratified K-Fold Cross Validation
```

---

# Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib mne scikit-learn torch
```

---

# Usage

## EEG Cleaning

```bash
python eeg_clean.py
```

## Compare Cleaning Quality

```bash
python compare.py
```

## Visualize EEG

```bash
python graph_plot.py
```

## Train Classifier

```bash
python classify.py
```

---

# Technologies Used

- Python
- PyTorch
- MNE-Python
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

---

# Goal

To build a reliable EEG-based stress detection system using advanced signal processing and deep learning methods.
