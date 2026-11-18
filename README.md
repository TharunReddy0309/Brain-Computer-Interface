# Brain-Computer Interfaces: EEG Signal Decoding

**Research Focus:** Non-invasive EEG-based BCI Systems — Affective State Decoding & Motor Imagery Classification  
**B.Tech Project — Brain-Computer Interfaces**  
**BY** Tharun Reddy & Raghuveer Karanam

---

## Abstract

This repository presents two independent EEG-based Brain-Computer Interface (BCI) systems developed as part of a Bachelor's thesis project. The first system addresses **passive BCI** — the automatic detection of mental stress from scalp EEG using a Shallow Convolutional Neural Network trained on the SAM40 dataset. The second system addresses **active BCI** — four-class motor imagery classification using EEGNet on the BCI Competition IV Dataset 2a, with direct application to assistive wheelchair control for individuals with motor disabilities.

Both systems are implemented in PyTorch and incorporate modern signal processing, deep learning, and evaluation methodologies aligned with current neuroscience and BCI research standards.

---

## Repository Structure

```
BCI/
│
├── Stress_Detection/                     ← Passive BCI: Mental Stress vs. Relax
│   ├── eeg_clean.py                         # Bandpass filtering + FastICA artifact removal
│   ├── compare.py                           # Cleaning quality validation (metrics + plots)
│   ├── graph_plot.py                        # MNE-based interactive EEG visualiser
│   ├── classify.py                          # ShallowConvNet training pipeline + K-Fold CV
│   ├── BTP-Evaluation1.pdf                  # BTP Mid-Semester Evaluation
│   ├── BTP-Evaluvation2.pdf                 # BTP End-Semester Evaluation
│   └── README.md                            ← Module documentation
│
├── Motor_Imagery_Detection/              ← Active BCI: 4-class Motor Imagery
│   ├── EEGNet.ipynb                         # Full EEGNet pipeline (Jupyter Notebook)
│   ├── wheelchair.pdf                       # Reference: MI-based wheelchair control system
│   └── README.md                            ← Module documentation
│
└── README.md                             ← This file (global overview)
```

---

## Projects Overview

### Project 1 — Mental Stress Detection (Passive BCI)

> **Directory:** [`Stress_Detection/`](./Stress_Detection/)  
> **Full Documentation:** [`Stress_Detection/README.md`](./Stress_Detection/README.md)

Passive BCIs monitor the user's mental state without requiring deliberate control actions. This system continuously decodes **cognitive stress** from ongoing EEG activity, with applications in workplace ergonomics, driver monitoring, clinical psychiatry, and adaptive human-computer interaction.

**Core Methodology:**

| Component | Choice | Justification |
|-----------|--------|---------------|
| Dataset | SAM40 (40 subjects, 32 ch, 128 Hz) | Well-validated stress/relax paradigm |
| Preprocessing | Butterworth bandpass (0.5–40 Hz) + FastICA | Remove drift and ocular/EMG artifacts |
| Segmentation | 2 s sliding window, 50% overlap | Sufficient temporal context for oscillatory patterns |
| Normalisation | Z-score per channel (train statistics) | Prevents data leakage, handles inter-session drift |
| Model | ShallowConvNet (PyTorch) | Biologically motivated; learns FBCSP-equivalent features |
| Regularisation | Mixup augmentation + Label smoothing + Cosine annealing | Prevents overfitting on EEG data |
| Evaluation | 5-Fold Stratified Cross-Validation | Robust performance estimate; accounts for class imbalance |

**ShallowConvNet — Architecture Summary:**
```
Input (B, 1, 32, 640)
  → Temporal Conv (1×25)      [FIR filter bank equivalent]
  → Spatial Conv (32×1)       [CSP-equivalent spatial filtering]
  → Square → AvgPool → Log    [Log-band-power features]
  → BatchNorm → Dropout(0.5)
  → Linear → 2-class output   [Relax / Stress]
```

---

### Project 2 — Motor Imagery Classification (Active BCI)

> **Directory:** [`Motor_Imagery_Detection/`](./Intent_Detection/)  
> **Full Documentation:** [`Motor_Imagery_Detection/README.md`](./Intent_Detection/README.md)

Active BCIs translate voluntary cognitive intent — such as imagining moving a limb — into device commands. This system decodes four MI classes from EEG to enable non-muscular control of a motorised wheelchair, providing mobility assistance to patients with ALS, spinal cord injury, or locked-in syndrome.

**Core Methodology:**

| Component | Choice | Justification |
|-----------|--------|---------------|
| Dataset | BCI Competition IV 2a (9 subjects, 22 ch, 250 Hz) | Community benchmark; 4-class MI |
| Preprocessing | Drop EOG → Bandpass (4–40 Hz) → Notch (50 Hz) → 4 s epoching | Standard MI BCI preprocessing chain |
| Model | EEGNet-8,2 (PyTorch) | State-of-art compact CNN for EEG; ~2,500 parameters |
| Training | Adam, Cross-Entropy, 500 epochs | Stable convergence on small trial counts |
| Evaluation | Test accuracy + 4×4 Confusion matrix | Per-class discrimination analysis |

**EEGNet — Architecture Summary:**
```
Input (B, 1, 22, 1000)
  → Temporal Conv (1×64, F1=8)          [learned FIR filter bank @ F_s/2]
  → Depthwise Spatial Conv (22×1, D=2)  [CSP-equivalent spatial filter]
  → BatchNorm → ELU → AvgPool → Dropout
  → Separable Conv (1×16, F2=16)        [temporal feature aggregation]
  → BatchNorm → ELU → AvgPool → Dropout
  → Flatten → Linear → 4-class output   [LH / RH / Foot / Tongue]
```

---

## Technical Comparison

| Dimension | Stress Detection | Motor Imagery Detection |
|-----------|-----------------|------------------------|
| **BCI Type** | Passive (mental state) | Active (voluntary intent) |
| **Cognitive Paradigm** | Mental arithmetic (Stroop) | Kinesthetic imagery |
| **Neural Markers** | Alpha ERD, frontal theta, beta ERS | Mu/beta ERD — somatotopic |
| **Dataset** | SAM40 — 40 subjects | BCI Comp IV 2a — 9 subjects |
| **Channels** | 32 | 22 |
| **Sampling Rate** | 128 Hz | 250 Hz |
| **Classes** | 2 (Binary) | 4 (Multiclass) |
| **Model** | ShallowConvNet (~18K params) | EEGNet-8,2 (~2.5K params) |
| **Training** | 5-Fold CV + Mixup + Label smoothing | Single split, 500 epochs |
| **Evaluation** | Accuracy, Balanced-Acc, Macro-F1 | Accuracy, Confusion matrix |
| **Key Challenge** | Inter-subject variability; class imbalance | Low trial count; spatial overlap between classes |

---

## Background: Brain-Computer Interfaces

A **Brain-Computer Interface** is a system that establishes a direct communication pathway between the central nervous system and an external device, bypassing conventional neuromuscular pathways. BCIs hold transformative potential across:

- **Clinical / Assistive Technology** — restoring communication and mobility for paralysed patients (ALS, spinal cord injury, stroke, locked-in syndrome).
- **Neurological Rehabilitation** — motor recovery using BCI-driven neurofeedback.
- **Cognitive Monitoring** — real-time workload, stress, and fatigue assessment in aviation, medicine, and high-stakes professions.
- **Augmentation** — enhancing healthy human performance and enabling novel human-machine interaction paradigms.

### EEG as a BCI Sensing Modality

Electroencephalography (EEG) is the most widely deployed BCI sensing modality due to its favourable trade-off between spatial resolution, temporal resolution, cost, and non-invasiveness.

| Modality | Temporal Res. | Spatial Res. | Cost | Invasiveness |
|----------|-------------|-------------|------|-------------|
| **EEG** | **~1 ms** | ~1–2 cm | Low | Non-invasive |
| fMRI | ~1–2 s | ~1 mm | Very High | Non-invasive |
| ECoG | ~1 ms | ~1 mm | High | Semi-invasive |
| Spiking Arrays | ~0.1 ms | Single neuron | Very High | Fully invasive |

### BCI Paradigm Classification

```
Brain-Computer Interfaces
├── Passive BCI
│   ├── Mental Workload Monitoring
│   ├── Fatigue Detection
│   └── ← Stress Detection  [Project 1]
│
└── Active BCI
    ├── P300 Speller
    ├── Steady-State Visual Evoked Potential (SSVEP)
    └── Motor Imagery          [Project 2]
        ├── Binary (LH vs. RH)
        └── Multiclass (LH / RH / Foot / Tongue)
```

---

## Deep Learning for EEG: Context

Traditional EEG classification pipelines rely on hand-crafted spectral features (band-power, coherence) and spatial filters (CSP, beamforming) fed into linear discriminant analysis (LDA) or SVM classifiers. This two-stage approach is powerful but brittle — feature engineering must be redone for each paradigm.

Both ShallowConvNet and EEGNet replace this pipeline with **end-to-end learned representations** that are:
- **Task-adaptive** — filters emerge from data, not neuroscientist priors.
- **Jointly optimised** — spatial and spectral filters are co-trained.
- **Interpretable** — learned filters can be visualised via activation maximisation, saliency maps, and topographic projections.

The field has since evolved toward **Foundation Models for EEG** (e.g., LaBraM, BENDR), **Transformer-based architectures** (e.g., EEG-Conformer), and **Riemannian geometry** methods (MDM, TSMNet) — directions that extend naturally from the architectures implemented here.

---

## Getting Started

### Prerequisites

```bash
pip install numpy scipy torch scikit-learn mne matplotlib
```

### Run Stress Detection

```bash
cd Stress_Detection/
python eeg_clean.py         # Preprocess raw EEG
python classify.py          # Train and evaluate ShallowConvNet
```

### Run Motor Imagery Classification

```bash
cd Motor_Imagery_Detection/
jupyter notebook EEGNet.ipynb   # Or open in Google Colab
```

---

## Citation

If you use or reference this repository in your research, please cite:

```bibtex
@misc{karanam2026bci,
  author = {Karanam, Raghuveer},
  title  = {EEG-Based BCI Systems: Stress Detection and Motor Imagery Classification},
  year   = {2026},
  url    = {https://github.com/Rag-2007/Brain-Computer-Interfaces-}
}
```

---

## Key References

1. **Lawhern, V. J., et al.** (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

2. **Schirrmeister, R. T., et al.** (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391–5420.

3. **Pfurtscheller, G., & Lopes da Silva, F. H.** (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842–1857.

4. **Tangermann, M., et al.** (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.

5. **Hyvärinen, A., & Oja, E.** (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4–5), 411–430.

6. **Lotte, F., et al.** (2018). A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. *Journal of Neural Engineering*, 15(3), 031005.

7. **BCI Competition IV** — https://www.bbci.de/competition/iv/

8. **SAM40 Dataset** — EEG stress/relaxation benchmark, 40 subjects, 32-channel.

9. **MNE-Python** — https://mne.tools
