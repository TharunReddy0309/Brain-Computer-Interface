# Motor Imagery Classification using EEGNet

This project focuses on Motor Imagery (MI) EEG classification using the BCI Competition IV 2a dataset and the EEGNet deep learning architecture.

The system classifies four motor imagery tasks:

- Left Hand
- Right Hand
- Foot
- Tongue

---

# Dataset

Dataset used:

```text
BCI Competition IV - Dataset 2a
```

The dataset contains EEG recordings from multiple subjects performing motor imagery tasks.

---

# Project File

├── EEGNet.ipynb

---

# Project Pipeline

The notebook includes:

1. Dataset Download
2. EEG Preprocessing
3. Epoch Extraction
4. EEGNet Model Implementation
5. Training and Evaluation

---

# 1. Data Preprocessing

## Dataset Loading

- Downloads dataset automatically
- Uses MNE-Python to load `.gdf` EEG files

---

## Signal Cleaning

### Channel Selection
- Removes EOG channels

### Filtering
- Bandpass Filter:
  ```python
  4 – 40 Hz
  ```

### Notch Filtering
- Removes powerline noise:
  ```python
  50 Hz
  ```

---

# 2. Epoch Extraction

Extracts EEG segments immediately after motor imagery cue presentation.

## Epoch Length

```python
4 seconds
```

---

# 3. EEGNet Architecture

Implements:

```text
EEGNet-8,2
```

EEGNet is a compact CNN designed specifically for EEG signal classification.

---

## EEGNet Components

### Depthwise Convolutions
Learns spatial EEG filters efficiently.

### Separable Convolutions
Learns temporal summaries with fewer parameters.

---

# 4. Training

## Train/Test Split

```python
90% Train
10% Test
```

---

## Optimizer

```python
Adam
```

## Loss Function

```python
CrossEntropyLoss
```

## Training Duration

```python
500 Epochs
```

---

# 5. Evaluation

Evaluates model performance on test data.

## Metrics

- Classification Accuracy
- Confusion Matrix

The confusion matrix helps visualize prediction quality across all 4 classes.

---

# Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib mne torch scikit-learn
```

---

# Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
EEGNet.ipynb
```

Run all notebook cells sequentially.

---

# Technologies Used

- Python
- PyTorch
- MNE-Python
- NumPy
- Scikit-learn
- Matplotlib

---

# Goal

To develop an efficient and lightweight EEG-based Motor Imagery classification system using EEGNet.
