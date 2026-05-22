import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import numpy as np
import scipy.io as sio

def _load(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    key = "Clean_data"
    arr = mat[key].astype(np.float64)
    print(f"Loaded '{key}'  shape={arr.shape}  std={arr.std():.3f} µV")
    return arr

def plot_basic_metrics(a: np.ndarray, b: np.ndarray, label_a="File A", label_b="File B"):
    overall_std = [np.std(a), np.std(b)]
    mean_ch_std = [np.mean(np.std(a, axis=1)), np.mean(np.std(b, axis=1))]
    peak_amp    = [np.max(np.abs(a)), np.max(np.abs(b))]

    metrics = ["Overall STD", "Mean Channel STD", "Peak Amplitude"]

    vals_a = [overall_std[0], mean_ch_std[0], peak_amp[0]]
    vals_b = [overall_std[1], mean_ch_std[1], peak_amp[1]]

    x = np.arange(len(metrics))
    plt.figure(figsize=(6, 4))

    plt.bar(x - 0.2, vals_a, width=0.4, label=label_a)
    plt.bar(x + 0.2, vals_b, width=0.4, label=label_b)

    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Basic EEG Comparison Metrics")

    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    cleaned1 = _load("/Users/karanamraghuveer/Desktop/PROJECTS/BCI/NEW_PY/Arithmetic_sub_2_trial1_cleaned.mat")
    cleaned2 = _load("/Users/karanamraghuveer/Desktop/PROJECTS/BCI/Data/filtered_data/Arithmetic_sub_2_trial1.mat")
    plot_basic_metrics(cleaned1, cleaned2, "Python Cleaned", "MATLAB Cleaned")