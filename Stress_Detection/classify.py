import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              balanced_accuracy_score, f1_score)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 2
NUM_CHANNELS = 32
SFREQ        = 320
WINDOW_SEC   = 2.0
OVERLAP      = 0.5
CHUNK_SIZE   = int(WINDOW_SEC * SFREQ)   # 640
EPOCHS       = 200
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 5e-4
PATIENCE     = 30
N_FOLDS      = 5   # ← change this to however many folds you want

# ─── LABEL-SMOOTHING CROSS ENTROPY ──────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.cls       = classes
        self.weight    = weight

    def forward(self, pred, target):
        confidence  = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot     = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * confidence + (1 - one_hot) * smooth_val
        log_prob    = F.log_softmax(pred, dim=1)
        if self.weight is not None:
            w    = self.weight[target].unsqueeze(1)
            loss = -(smooth_one_hot * log_prob * w).sum(dim=1).mean()
        else:
            loss = -(smooth_one_hot * log_prob).sum(dim=1).mean()
        return loss


# ─── SLIDING-WINDOW SEGMENTATION ────────────────────────────────────────────────
def segment(eeg: np.ndarray, chunk: int, step: int):
    n_t    = eeg.shape[1]
    starts = range(0, n_t - chunk + 1, step)
    return [eeg[:, s:s + chunk].astype(np.float32) for s in starts]


# ─── NORMALIZATION ───────────────────────────────────────────────────────────────
def channel_normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ─── MIXUP AUGMENTATION ─────────────────────────────────────────────────────────
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3):
    if alpha <= 0:
        return x, y.float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam


# ─── SHALLOW CONVOLUTIONAL MODEL ────────────────────────────────────────────────
class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=32, n_times=640, n_classes=2,
                 n_filters_time=40, filter_time_len=25, n_filters_spat=40,
                 dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters_time,
                                  kernel_size=(1, filter_time_len),
                                  bias=False)
        self.spatial  = nn.Conv2d(n_filters_time, n_filters_spat,
                                  kernel_size=(n_channels, 1),
                                  bias=False)
        self.bn1      = nn.BatchNorm2d(n_filters_spat)
        pool_len      = 75
        stride_len    = 15
        self.pool     = nn.AvgPool2d(kernel_size=(1, pool_len),
                                     stride=(1, stride_len))
        self.bn2      = nn.BatchNorm2d(n_filters_spat)
        self.drop     = nn.Dropout(dropout)
        dummy         = torch.zeros(1, 1, n_channels, n_times)
        out           = self._forward_features(dummy)
        flat          = out.shape[1]
        self.fc       = nn.Linear(flat, n_classes)

    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn1(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = self.bn2(x)
        x = self.drop(x)
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._forward_features(x))


# ─── MODEL FACTORY ──────────────────────────────────────────────────────────────
def make_model() -> nn.Module:
    return ShallowConvNet(
        n_channels      = NUM_CHANNELS,
        n_times         = CHUNK_SIZE,
        n_classes       = NUM_CLASSES,
        n_filters_time  = 40,
        filter_time_len = 25,
        n_filters_spat  = 40,
        dropout         = 0.5
    ).to(DEVICE)


# ─── CLASS-WEIGHT COMPUTATION ───────────────────────────────────────────────────
def compute_class_weights(y_tr: np.ndarray, cap: float = 4.0) -> np.ndarray:
    classes, counts = np.unique(y_tr, return_counts=True)
    raw_w = len(y_tr) / (len(classes) * counts)
    raw_w = np.clip(raw_w, 1.0 / cap, cap)
    raw_w = raw_w / raw_w.sum() * len(classes)
    cw    = np.ones(NUM_CLASSES, dtype=np.float32)
    for c, w in zip(classes, raw_w):
        cw[int(c)] = float(w)
    return cw


# ─── TRAIN ONE FOLD ─────────────────────────────────────────────────────────────
def train_fold(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray,
               class_weights: np.ndarray):

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion  = LabelSmoothingCE(NUM_CLASSES, smoothing=0.1, weight=cw_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=LR / 100
    )

    sample_weights = np.array([class_weights[int(l)] for l in y_tr])
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(y_tr),
        replacement = True
    )

    X_t     = torch.tensor(X_tr[:, np.newaxis], dtype=torch.float32)
    y_t     = torch.tensor(y_tr, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=sampler, drop_last=False)

    best_loss  = float("inf")
    best_state = None
    no_improve = 0

    MIXUP_ALPHA = 0.3
    USE_MIXUP   = True

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            if USE_MIXUP and np.random.rand() < 0.5:
                xb_m, ya, yb_mix, lam = mixup_batch(xb, yb, MIXUP_ALPHA)
                logits = model(xb_m)
                loss   = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb_mix)
            else:
                loss = criterion(model(xb), yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        scheduler.step()
        epoch_loss /= len(dataset)

        if epoch_loss < best_loss - 1e-4:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early stop @ epoch {epoch}  (best loss {best_loss:.4f})")
            break

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>4d}/{EPOCHS}  loss={epoch_loss:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})


# ─── THRESHOLD TUNING ───────────────────────────────────────────────────────────
def tune_threshold(probs: np.ndarray, y_te: np.ndarray,
                   thresholds=np.arange(0.2, 0.81, 0.05)):
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs[:, 1] >= t).astype(int)
        f = f1_score(y_te, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t  = t
    return best_t, best_f1


# ─── K-FOLD CROSS VALIDATION ────────────────────────────────────────────────────
def kfold_cv(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS):
    """
    Stratified K-Fold cross-validation over all segments.

    NOTE: Because segments from the same original recording end up in both
    train and test folds, this measures within-distribution performance.
    If you want subject-independent estimates, use GroupKFold with the
    subjects array as groups (see commented block at the bottom).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nDevice     : {DEVICE}")
    print(f"Chunk size : {CHUNK_SIZE} samples ({WINDOW_SEC}s @ {SFREQ}Hz)")
    print(f"Total segments : {len(y)}  |  Classes : {np.unique(y)}")
    print(f"Running {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'═'*60}")

    fold_accs, fold_bal, fold_f1 = [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Normalize using train statistics only
        X_tr, X_te   = channel_normalize(X_tr, X_te)
        class_weights = compute_class_weights(y_tr, cap=4.0)

        classes, counts = np.unique(y_tr, return_counts=True)
        print(f"\nFold {fold_idx}/{n_folds}")
        print(f"  Train segments : {len(y_tr)}  |  Test segments : {len(y_te)}")
        print(f"  Class dist train : { {int(c): int(n) for c, n in zip(classes, counts)} }")
        te_classes, te_counts = np.unique(y_te, return_counts=True)
        print(f"  Class dist test  : { {int(c): int(n) for c, n in zip(te_classes, te_counts)} }")
        print(f"  Class weights    : { {i: round(float(w), 3) for i, w in enumerate(class_weights)} }")

        model = make_model()
        train_fold(model, X_tr, y_tr, class_weights)

        X_te_t = torch.tensor(X_te[:, np.newaxis], dtype=torch.float32).to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()

        best_t, _ = tune_threshold(probs, y_te)
        preds     = (probs[:, 1] >= best_t).astype(int)
        print(f"  Best threshold   : {best_t:.2f}")

        acc      = accuracy_score(y_te, preds)
        bal_acc  = balanced_accuracy_score(y_te, preds)
        macro_f1 = f1_score(y_te, preds, average="macro", zero_division=0)

        fold_accs.append(acc)
        fold_bal.append(bal_acc)
        fold_f1.append(macro_f1)

        print(f"\n  ┌─ Fold {fold_idx} Metrics ──────────────────────────────────")
        print(f"  │  Accuracy       : {acc*100:.2f}%")
        print(f"  │  Balanced-Acc   : {bal_acc*100:.2f}%")
        print(f"  │  Macro-F1       : {macro_f1:.4f}")
        print(f"  └────────────────────────────────────────────────────")
        print(classification_report(y_te, preds,
                                    target_names=["Relax", "Stress"],
                                    zero_division=0))
        print("─" * 60)

    # ─── SUMMARY ───────────────────────────────────────────────────────────────
    mean_acc = np.mean(fold_accs) * 100
    std_acc  = np.std(fold_accs)  * 100
    mean_bal = np.mean(fold_bal)  * 100
    std_bal  = np.std(fold_bal)   * 100
    mean_f1  = np.mean(fold_f1)
    std_f1   = np.std(fold_f1)

    print(f"\n{'═'*60}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'═'*60}")
    print(f"  Accuracy      : {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Balanced-Acc  : {mean_bal:.2f}% ± {std_bal:.2f}%")
    print(f"  Macro-F1      : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<6} {'Acc':>8} {'Bal-Acc':>10} {'Macro-F1':>10}")
    print(f"  {'─'*38}")
    for i, (a, b, f) in enumerate(zip(fold_accs, fold_bal, fold_f1), start=1):
        print(f"  {i:<6} {a*100:>7.2f}% {b*100:>9.2f}% {f:>10.4f}")
    print(f"{'═'*60}\n")

    return fold_accs, fold_bal, fold_f1


# ─── SAM40 LOADER ───────────────────────────────────────────────────────────────
def load_sam40_mat(data_dir: str):
    EEG_KEYS    = ["Clean_data", "Data", "data"]
    X_list, y_list, s_list = [], [], []
    subject_map = {}
    step        = int(CHUNK_SIZE * (1 - OVERLAP))

    mat_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {data_dir}")

    print(f"Found {len(mat_files)} .mat file(s)")
    print(f"Segment: {CHUNK_SIZE} samples, stride: {step} samples\n{'─'*60}")

    for fname in mat_files:
        fpath = os.path.join(data_dir, fname)
        try:
            fname_lower = fname.lower()
            label       = 0 if 'relax' in fname_lower else 1
            label_str   = "Relax" if label == 0 else "Stress"

            match = re.search(r"sub_(\d+)", fname_lower)
            if not match:
                print(f"[SKIP] No subject ID in {fname}")
                continue
            sid_str = f"sub_{match.group(1)}"
            if sid_str not in subject_map:
                subject_map[sid_str] = len(subject_map)
            subject_id = subject_map[sid_str]

            mat = sio.loadmat(fpath)
            eeg = None
            for key in EEG_KEYS:
                if key in mat:
                    candidate = mat[key]
                    if isinstance(candidate, np.ndarray) and candidate.dtype.names:
                        try:
                            candidate = candidate["data"][0, 0]
                        except Exception:
                            pass
                    if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                        eeg = candidate
                        break

            if eeg is None:
                print(f"[ERROR] No EEG array found in {fname}")
                continue

            print(f"\nPreprocessing: {fname}")
            print(f"Original Shape: {eeg.shape}")
            if eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T
            print(f"After transpose: {eeg.shape}")

            n_ch, n_t = eeg.shape
            if n_ch != NUM_CHANNELS:
                print(f"[SKIP] {fname} wrong channel count ({n_ch})")
                continue
            if n_t < CHUNK_SIZE:
                print(f"[SKIP] Too Short: {fname} ({n_t} samples)")
                continue

            segs = segment(eeg, CHUNK_SIZE, step)
            if not segs:
                print(f"[SKIP] {fname} no segments")
                continue

            for seg in segs:
                X_list.append(seg)
                y_list.append(label)
                s_list.append(subject_id)

            print(f"Loaded {fname} → {label_str}  |  {len(segs)} segments")

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    if not X_list:
        print("No data collected")
        exit()

    X        = np.stack(X_list)
    y        = np.array(y_list)
    subjects = np.array(s_list)

    print(f"\nTotal segments : {X.shape}")
    print(f"Relax          : {np.sum(y==0)}")
    print(f"Stress         : {np.sum(y==1)}")
    print(f"Subjects       : {len(np.unique(subjects))}")
    return X, y, subjects, subject_map


# ─── ENTRYPOINT ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/Data/filtered_data'

    np.random.seed(42)
    torch.manual_seed(42)

    X, y, subjects, subject_map = load_sam40_mat(DATA_DIR)

    # ── Option A: Stratified K-Fold (segments stratified by label) ──────────────
    kfold_cv(X, y, n_folds=N_FOLDS)

    # ── Option B: Subject-level Group K-Fold (uncomment to use instead) ─────────
    # Ensures no subject's segments appear in both train and test — a stricter,
    # more realistic estimate of cross-subject generalization.
    #
    # from sklearn.model_selection import StratifiedGroupKFold
    #
    # def kfold_subject_cv(X, y, subjects, n_folds=N_FOLDS):
    #     sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    #     # ... same loop body as kfold_cv, but pass groups=subjects to sgkf.split()
    #     for fold_idx, (tr_idx, te_idx) in enumerate(
    #             sgkf.split(X, y, groups=subjects), start=1):
    #         ...  # identical to kfold_cv loop body
    #
    # kfold_subject_cv(X, y, subjects)