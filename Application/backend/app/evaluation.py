"""SAM40 recording-level EEG inference and metric calculation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .dataset import Recording
from .model import (
    ModelProfile,
    SAM40_STANDARD,
    WESAD_FEATURES,
    class_indices,
    load_model,
    load_sklearn_model,
    load_wesad_model,
)
from .wesad import WesadWindow


CLASSES = ["Relax", "Stress / Arithmetic"]
WINDOW_STEP = 320


def _four_band_signal(signal: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
    """Build the documented broadband/theta/alpha/beta views for SAM40 multiband weights."""
    def bandpass(low: float, high: float) -> np.ndarray:
        sos = butter(4, (low, high), btype="bandpass", fs=sampling_rate_hz, output="sos")
        return sosfiltfilt(sos, signal, axis=1).astype(np.float32)

    return np.stack((
        bandpass(0.5, 45.0),
        bandpass(4.0, 8.0),
        bandpass(8.0, 13.0),
        bandpass(13.0, 30.0),
    ))


def _windows_for_recording(record: Recording, profile: ModelProfile) -> list[np.ndarray]:
    return _windows_for_signal(record.signal, profile)


def _windows_for_signal(signal: np.ndarray, profile: ModelProfile) -> list[np.ndarray]:
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (
        signal.std(axis=1, keepdims=True) + 1e-8
    )
    if profile.bands == 1:
        views = signal[None].astype(np.float32)
    elif profile.bands == 4 and profile.sampling_rate_hz:
        views = _four_band_signal(signal, profile.sampling_rate_hz)
    else:
        raise ValueError(f"Unsupported input-band configuration: {profile.bands}.")
    return [views[:, :, start : start + profile.samples]
            for start in range(0, signal.shape[1] - profile.samples + 1, WINDOW_STEP)]


def evaluate(
    checkpoint: bytes | Path,
    records: list[Recording],
    progress: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if progress:
        progress("model", 10, "Loading compatible EEG model")
    if isinstance(checkpoint, Path) and checkpoint.suffix.lower() == ".pkl":
        model = load_sklearn_model(checkpoint)
        profile = SAM40_STANDARD
        model_format = "scikit-learn pickle"
        profile_id = "sam40_sklearn_eeg_v1"
        device_name = "cpu (scikit-learn)"
    else:
        model, profile = load_model(checkpoint, device)
        model_format = "PyTorch state dictionary"
        profile_id = profile.id
        device_name = str(device)
    windows: list[np.ndarray] = []
    owners: list[int] = []
    for index, record in enumerate(records):
        for window in _windows_for_recording(record, profile):
            windows.append(window)
            owners.append(index)
        if progress:
            progress("preprocessing", 15 + int(30 * (index + 1) / len(records)), f"Prepared recording {index + 1}/{len(records)}")

    if not windows:
        raise ValueError(f"No valid {profile.samples}-sample windows were created.")
    if model_format == "scikit-learn pickle":
        x = np.stack(windows).reshape(len(windows), -1)
        window_probs = _sklearn_probabilities(model, x)
    else:
        predictions: list[np.ndarray] = []
        x = torch.from_numpy(np.stack(windows))
        with torch.inference_mode():
            for offset in range(0, len(x), 128):
                logits = model(x[offset : offset + 128].to(device))
                predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
        window_probs = np.concatenate(predictions)

    rows = []
    for index, record in enumerate(records):
        selected = window_probs[np.asarray(owners) == index]
        probabilities = selected.mean(axis=0)
        predicted = int(probabilities.argmax())
        rows.append({
            "filename": record.filename,
            "true_label": record.label,
            "true_class": CLASSES[record.label],
            "predicted_label": predicted,
            "predicted_class": CLASSES[predicted],
            "probability_relax": float(probabilities[0]),
            "probability_stress": float(probabilities[1]),
            "windows_used": int(len(selected)),
        })
    y_true = np.asarray([row["true_label"] for row in rows], dtype=int)
    y_pred = np.asarray([row["predicted_label"] for row in rows], dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0)
    result = {
        "profile": profile_id,
        "evaluation_unit": "recording",
        "device": device_name,
        "recordings": len(rows),
        "windows": len(windows),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": {"labels": CLASSES, "values": matrix.tolist()},
            "per_class": {name: {key: float(value) if key != "support" else int(value)
                                    for key, value in class_report[name].items()}
                          for name in CLASSES},
        },
        "configuration": {
            "model_format": model_format,
            "channels": profile.channels,
            "input_bands": profile.bands,
            "sampling_rate_hz": profile.sampling_rate_hz,
            "window_samples": profile.samples,
            "step_samples": WINDOW_STEP,
            "normalization": "per_recording_channel_zscore",
            "band_preprocessing": (
                "broadband_0.5_45Hz_theta_4_8Hz_alpha_8_13Hz_beta_13_30Hz"
                if profile.bands == 4 else "none"
            ),
            "aggregation": "mean_window_probability_per_recording",
        },
        "predictions": rows,
    }
    if progress:
        progress("complete", 100, "Evaluation complete")
    return result


def predict_stress(
    checkpoint: Path,
    signal: np.ndarray,
    filename: str,
) -> dict[str, Any]:
    """Run single-recording stress inference without presenting it as validation accuracy."""
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, profile = load_model(checkpoint, device)
    if profile.bands != 1:
        raise ValueError("Single-file stress inference currently supports the standard one-band SAM40 model.")

    windows = _windows_for_signal(signal, profile)
    if not windows:
        raise ValueError(f"Recording must contain at least {profile.samples} samples.")
    inputs = torch.from_numpy(np.stack(windows))
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(inputs), 128):
            logits = model(inputs[offset : offset + 128].to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    window_probabilities = np.concatenate(probabilities)
    aggregate = window_probabilities.mean(axis=0)
    predicted = int(aggregate.argmax())
    window_results = []
    for index, values in enumerate(window_probabilities):
        window_label = int(values.argmax())
        window_results.append({
            "window": index + 1,
            "start_sample": index * WINDOW_STEP,
            "predicted_label": window_label,
            "predicted_class": CLASSES[window_label],
            "probability_relax": float(values[0]),
            "probability_stress": float(values[1]),
        })

    return {
        "task": "stress",
        "profile": profile.id,
        "evaluation_unit": "single_recording_inference",
        "filename": filename,
        "predicted_label": predicted,
        "predicted_class": CLASSES[predicted],
        "confidence": float(aggregate[predicted]),
        "probability_relax": float(aggregate[0]),
        "probability_stress": float(aggregate[1]),
        "windows": len(window_results),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        "configuration": {
            "channels": profile.channels,
            "sampling_rate_hz": profile.sampling_rate_hz,
            "window_samples": profile.samples,
            "step_samples": WINDOW_STEP,
            "normalization": "per_recording_channel_zscore",
            "aggregation": "mean_window_probability",
        },
        "window_predictions": window_results,
        "interpretation_note": "Prediction confidence is the model's output score; it is not measured accuracy or clinical confidence.",
    }


def _sklearn_probabilities(model: Any, windows: np.ndarray) -> np.ndarray:
    class_to_index = class_indices(model.classes_)
    if callable(getattr(model, "predict_proba", None)):
        raw_probabilities = np.asarray(model.predict_proba(windows), dtype=np.float64)
        if raw_probabilities.shape != (len(windows), len(model.classes_)):
            raise ValueError("The .pkl model returned probabilities with an unexpected shape.")
        if not np.isfinite(raw_probabilities).all():
            raise ValueError("The .pkl model returned non-finite probabilities.")
        probabilities = np.empty((len(windows), 2), dtype=np.float64)
        for column, label in enumerate(model.classes_):
            probabilities[:, class_to_index[label]] = raw_probabilities[:, column]
        return probabilities

    labels = np.asarray(model.predict(windows))
    if labels.shape != (len(windows),):
        raise ValueError("The .pkl model returned predictions with an unexpected shape.")
    probabilities = np.zeros((len(windows), 2), dtype=np.float64)
    for row, label in enumerate(labels):
        try:
            probabilities[row, class_to_index[label]] = 1.0
        except KeyError as exc:
            raise ValueError(f"The .pkl model predicted an unknown class: {label!r}.") from exc
    return probabilities


def evaluate_wesad(
    checkpoint: Path,
    windows: list[WesadWindow],
    progress: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate WESAD baseline/stress windows with the matching packaged model."""
    started = time.perf_counter()
    if progress:
        progress("model", 10, "Loading WESAD wrist-sensor classifier")
    model, feature_columns = load_wesad_model(checkpoint)
    if tuple(feature_columns) != WESAD_FEATURES:
        raise ValueError("The WESAD classifier feature order is not supported.")

    x = np.asarray([
        [window.features[name] for name in feature_columns]
        for window in windows
    ], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("WESAD feature extraction produced non-finite values.")
    probabilities = _sklearn_probabilities(model, x)
    y_true = np.asarray([window.label for window in windows], dtype=int)
    y_pred = probabilities.argmax(axis=1)
    class_names = ["Baseline", "Stress"]
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    predictions = []
    for window, predicted, probability in zip(windows, y_pred, probabilities):
        predictions.append({
            "filename": f"{window.subject}_{window.window_start_seconds}s",
            "subject": window.subject,
            "window_start_seconds": window.window_start_seconds,
            "true_label": window.label,
            "true_class": class_names[window.label],
            "predicted_label": int(predicted),
            "predicted_class": class_names[int(predicted)],
            "probability_baseline": float(probability[0]),
            "probability_stress": float(probability[1]),
            "windows_used": 1,
        })

    subjects = sorted({window.subject for window in windows})
    result = {
        "profile": "wesad_wrist_svm_v1",
        "evaluation_unit": "window",
        "device": "cpu (scikit-learn)",
        "recordings": len(subjects),
        "subjects": len(subjects),
        "windows": len(windows),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": {
                "labels": class_names,
                "values": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            },
            "per_class": {
                name: {
                    key: float(report[name][key]) if key != "support" else int(report[name][key])
                    for key in ("precision", "recall", "f1-score", "support")
                }
                for name in class_names
            },
        },
        "configuration": {
            "model_format": "scikit-learn pickle",
            "dataset": "WESAD wrist sensors",
            "window_seconds": 60,
            "step_seconds": 30,
            "sampling_rates_hz": {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32, "labels": 700},
            "features": feature_columns,
            "label_mapping": {"0": "Baseline (WESAD label 1)", "1": "Stress (WESAD label 2)"},
        },
        "predictions": predictions,
    }
    if progress:
        progress("complete", 100, "WESAD evaluation complete")
    return result
