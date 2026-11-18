"""Restricted loading and feature extraction for WESAD wrist-sensor archives."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import pickle
import re
import zipfile
from typing import Any

import numpy as np

from .model import WESAD_FEATURES


SOURCE_LABEL_RATE = 700
WINDOW_SECONDS = 60
STEP_SECONDS = 30
WRIST_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
MAX_SUBJECTS = 15
MAX_SUBJECT_BYTES = 500_000_000
MAX_UNCOMPRESSED_BYTES = 2_000_000_000


@dataclass
class WesadWindow:
    subject: str
    window_start_seconds: int
    label: int
    features: dict[str, float]


class _WesadUnpickler(pickle.Unpickler):
    _allowed = {
        ("builtins", "set"): set,
        ("builtins", "frozenset"): frozenset,
        ("builtins", "slice"): slice,
        ("__builtin__", "set"): set,
        ("__builtin__", "frozenset"): frozenset,
        ("__builtin__", "slice"): slice,
        ("_codecs", "encode"): __import__("_codecs").encode,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    }

    def find_class(self, module: str, name: str) -> Any:
        value = self._allowed.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"Unsupported object in WESAD data pickle: {module}.{name}")
        return value


def _read_subject_pickle(contents: bytes, filename: str) -> dict[str, Any]:
    try:
        data = _WesadUnpickler(BytesIO(contents), encoding="latin1").load()
    except Exception as exc:
        raise ValueError(f"{filename}: could not safely read the WESAD subject pickle: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("signal"), dict):
        raise ValueError(f"{filename}: expected a WESAD subject dictionary with signal and label fields.")
    if not isinstance(data.get("label"), np.ndarray):
        raise ValueError(f"{filename}: missing the WESAD label array.")
    return data


def _wrist_signal(data: dict[str, Any], name: str) -> np.ndarray:
    try:
        values = np.asarray(data["signal"]["wrist"][name])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing WESAD wrist signal {name}.") from exc
    if name == "ACC":
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("WESAD wrist ACC must have three axes.")
        values = np.linalg.norm(values, axis=1)
    else:
        values = values.reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"WESAD wrist signal {name} is empty or contains non-finite values.")
    return values.astype(np.float64, copy=False)


def _resample_labels(labels: np.ndarray, rate: int) -> np.ndarray:
    labels = labels.reshape(-1)
    sample_count = int(len(labels) * rate / SOURCE_LABEL_RATE)
    source_indices = np.ceil(np.arange(sample_count) * SOURCE_LABEL_RATE / rate).astype(np.int64)
    np.clip(source_indices, 0, len(labels) - 1, out=source_indices)
    return labels[source_indices]


def _window_label(labels: np.ndarray) -> int | None:
    relevant = np.unique(labels)
    relevant = relevant[relevant != 0]
    if len(relevant) != 1 or relevant[0] not in (1, 2):
        return None
    return int(relevant[0]) - 1


def _feature_window(signal: np.ndarray, rate: int, kind: str) -> dict[str, float]:
    if kind == "BVP":
        import neurokit2 as nk

        cleaned = nk.ppg_clean(signal, sampling_rate=rate)
        signals, info = nk.ppg_peaks(cleaned, sampling_rate=rate)
        peaks = np.asarray(info["PPG_Peaks"])
        if len(peaks) < 4:
            return {name: float("nan") for name in ("mean_hr", "rmssd", "pnn50", "lf_hf")}
        hrv_time = nk.hrv_time(signals, sampling_rate=rate, show=False).iloc[0]
        hrv_frequency = nk.hrv_frequency(signals, sampling_rate=rate, show=False).iloc[0]
        return {
            "mean_hr": float(60 / (np.diff(peaks).mean() / rate)),
            "rmssd": float(hrv_time["HRV_RMSSD"]),
            "pnn50": float(hrv_time["HRV_pNN50"]),
            "lf_hf": float(hrv_frequency["HRV_LFHF"]),
        }

    if kind == "EDA":
        import neurokit2 as nk

        cleaned = nk.eda_clean(signal, sampling_rate=rate)
        decomposed = nk.eda_phasic(cleaned, sampling_rate=rate)
        _, info = nk.eda_peaks(decomposed["EDA_Phasic"], sampling_rate=rate, method="neurokit")
        onsets = info.get("SCR_Onsets")
        amplitudes = info.get("SCR_Amplitude")
        count = len(onsets) if onsets is not None else 0
        x = np.arange(len(signal), dtype=np.float64)
        return {
            "eda_mean": float(np.mean(signal)),
            "eda_std": float(np.std(signal)),
            "eda_slope": float(np.polyfit(x, signal, 1)[0]),
            "scr_count": float(count),
            "scr_amplitude_mean": float(np.mean(amplitudes)) if count else 0.0,
        }

    x = np.arange(len(signal), dtype=np.float64)
    if kind == "TEMP":
        return {
            "temp_mean": float(np.mean(signal)),
            "temp_slope": float(np.polyfit(x, signal, 1)[0]),
        }

    magnitude = np.abs(signal)
    return {"acc_mean": float(np.mean(magnitude)), "acc_std": float(np.std(magnitude))}


def _process_subject(subject: str, data: dict[str, Any]) -> list[WesadWindow]:
    labels = np.asarray(data["label"]).reshape(-1)
    if not labels.size or not np.isfinite(labels).all():
        raise ValueError(f"{subject}: label array is empty or contains non-finite values.")
    windows_by_start: dict[int, WesadWindow] = {}

    for kind, rate in WRIST_RATES.items():
        signal = _wrist_signal(data, kind)
        aligned_labels = _resample_labels(labels, rate)
        usable_length = min(len(signal), len(aligned_labels))
        signal = signal[:usable_length]
        aligned_labels = aligned_labels[:usable_length]
        window_samples = WINDOW_SECONDS * rate
        step_samples = STEP_SECONDS * rate

        for start in range(0, usable_length - window_samples + 1, step_samples):
            end = start + window_samples
            label = _window_label(aligned_labels[start:end])
            if label is None:
                continue
            start_seconds = round(start / rate)
            if kind == "BVP":
                features = _feature_window(signal[start:end], rate, kind)
                windows_by_start[start_seconds] = WesadWindow(
                    subject=subject,
                    window_start_seconds=start_seconds,
                    label=label,
                    features=features,
                )
            elif start_seconds in windows_by_start:
                if windows_by_start[start_seconds].label == label:
                    windows_by_start[start_seconds].features.update(
                        _feature_window(signal[start:end], rate, kind)
                    )

    records = sorted(windows_by_start.values(), key=lambda item: item.window_start_seconds)
    valid = []
    for record in records:
        missing = [name for name in WESAD_FEATURES if not np.isfinite(record.features.get(name, np.nan))]
        missing = [name for name in missing if name != "scr_amplitude_mean"]
        if missing:
            continue
        record.features.setdefault("scr_amplitude_mean", 0.0)
        valid.append(record)
    return valid


def identify_wesad_subjects(contents: bytes) -> bool:
    """Return whether a ZIP contains WESAD-style subject pickle entries."""
    try:
        with zipfile.ZipFile(BytesIO(contents)) as archive:
            return any(
                not item.is_dir()
                and re.fullmatch(r"(?:.*/)?S\d+/S\d+\.pkl", item.filename, re.IGNORECASE)
                for item in archive.infolist()
            )
    except zipfile.BadZipFile as exc:
        raise ValueError("Dataset upload must be a valid ZIP archive.") from exc


def read_wesad_zip(contents: bytes) -> list[WesadWindow]:
    """Extract model-compatible wrist features from WESAD subject pickle files."""
    try:
        archive = zipfile.ZipFile(BytesIO(contents))
    except zipfile.BadZipFile as exc:
        raise ValueError("Dataset upload must be a valid ZIP archive.") from exc

    with archive:
        entries = [
            item for item in archive.infolist()
            if not item.is_dir()
            and re.fullmatch(r"(?:.*/)?(S\d+)/(S\d+)\.pkl", item.filename, re.IGNORECASE)
        ]
        if not entries:
            raise ValueError(
                "For a WESAD .pkl model, ZIP the raw WESAD subject folders (for example S2/S2.pkl). "
                "A SAM40 .mat EEG ZIP is a different dataset and cannot be used with this wearable model."
            )
        if len(entries) > MAX_SUBJECTS:
            raise ValueError(f"The WESAD archive contains more than {MAX_SUBJECTS} subject files.")
        if sum(item.file_size for item in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("The WESAD ZIP expands beyond the 2 GB safety limit.")

        paths: set[str] = set()
        subjects: set[str] = set()
        records: list[WesadWindow] = []
        for item in entries:
            match = re.fullmatch(r"(?:.*/)?(S\d+)/(S\d+)\.pkl", item.filename, re.IGNORECASE)
            assert match is not None
            subject, file_subject = match.group(1).upper(), match.group(2).upper()
            if subject != file_subject:
                raise ValueError(f"{item.filename}: folder and filename subject IDs do not match.")
            if item.filename in paths or subject in subjects:
                raise ValueError(f"The WESAD archive contains a duplicate subject: {subject}.")
            if item.file_size > MAX_SUBJECT_BYTES:
                raise ValueError(f"{item.filename} exceeds the 500 MB per-subject safety limit.")
            paths.add(item.filename)
            subjects.add(subject)
            data = _read_subject_pickle(archive.read(item), item.filename)
            records.extend(_process_subject(subject, data))

        if not records:
            raise ValueError(
                "No valid WESAD baseline/stress windows were found. Include raw wrist signals "
                "and labels for at least one subject."
            )
    return records
