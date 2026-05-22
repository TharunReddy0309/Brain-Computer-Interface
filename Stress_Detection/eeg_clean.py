import numpy as np
import scipy.io as sio
from scipy.signal import sosfiltfilt, butter
from scipy.stats import kurtosis
import os


def clean_eeg(input_path: str, sfreq: float = 128.0) -> str:
    name = os.path.splitext(input_path)[0]
    os.makedirs("NEW_PY", exist_ok=True)  
    name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join("NEW_PY", f"{name}_cleaned.mat")
    Data = _load(input_path)
    filtered   = _bandpass(Data, sfreq)
    cleaned    = _ica_kurtosis(filtered)
    sio.savemat(output_path, {"Clean_data": cleaned})
    print(f"Saved → {output_path}  shape={cleaned.shape}")
    return output_path

def _bandpass(data: np.ndarray, sfreq: float, l=0.5, h=40.0) -> np.ndarray:
    nyq = sfreq / 2
    sos = butter(4, [l / nyq, h / nyq], btype="bandpass", output="sos")
    return np.array([sosfiltfilt(sos, ch) for ch in data])

def _ica_kurtosis(data: np.ndarray, n_components: int = 20, kurt_threshold: float = 5.0) -> np.ndarray:
    n_ch = data.shape[0]
    mean     = data.mean(axis=1, keepdims=True)
    centered = data - mean
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered))
    eigvals  = np.maximum(eigvals, 1e-10)
    W_white  = (eigvecs / np.sqrt(eigvals)).T        
    whitened = W_white @ centered                     
    W = _fastica(whitened, n_components)           
    sources = W @ whitened                       
    bad = np.where(np.abs(kurtosis(sources, axis=1)) > kurt_threshold)[0]
    print(f"[ICA] Removed {len(bad)} artifact component(s): {list(bad)}")
    sources[bad] = 0
    mixing = np.linalg.pinv(W @ W_white)              
    return mixing @ sources + mean                    

def _fastica(Z: np.ndarray, n_comp: int, seed: int = 42) -> np.ndarray:
    n_ch = Z.shape[0]
    rng  = np.random.RandomState(seed)
    W    = np.zeros((n_comp, n_ch))

    for i in range(n_comp):
        w = rng.randn(n_ch)
        w /= np.linalg.norm(w) + 1e-10

        for _ in range(300):
            g   = np.tanh(w @ Z)
            gp  = 1 - g ** 2
            w_n = (Z * g).mean(axis=1) - gp.mean() * w
            for j in range(i):
                w_n -= (w_n @ W[j]) * W[j]

            norm = np.linalg.norm(w_n)
            if norm < 1e-10:
                break
            w_n /= norm

            if np.abs(np.abs(w_n @ w) - 1) < 1e-6:
                w = w_n
                break
            w = w_n

        W[i] = w

    return W

def _load(path: str) -> np.ndarray:
    mat = sio.loadmat(path)
    for key in ("Data", "data", "EEG", "eeg"):
        if key in mat:
            arr = mat[key].astype(np.float64)
            print(f"Loaded '{key}'  shape={arr.shape}  std={arr.std():.3f} µV")
            return arr
    raise KeyError(f"No 'Data' variable in {path}. Keys: {[k for k in mat if not k.startswith('_')]}")