"""Secure ZIP and labelled SAM40 MATLAB recording reader."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import zipfile

import numpy as np
from scipy.io import loadmat


MAX_ARCHIVE_FILES = 1000
MAX_UNCOMPRESSED_BYTES = 1_000_000_000
MAX_RECORDING_BYTES = 100_000_000


@dataclass
class Recording:
    filename: str
    label: int
    signal: np.ndarray


def _label(filename: str) -> int | None:
    name = filename.lower().rsplit("/", 1)[-1]
    if "relax" in name or re.search(r"(?:^|[_-])lo(?:[_\-.]|$)", name):
        return 0
    if "arithmetic" in name or "stress" in name or re.search(r"(?:^|[_-])hi(?:[_\-.]|$)", name):
        return 1
    return None


def _signal_from_mat(contents: bytes, filename: str) -> np.ndarray:
    try:
        variables = loadmat(BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Could not read {filename} as a MATLAB file: {exc}") from exc
    signal = next(
        (variables[key] for key in ("Clean_data", "Data", "data")
         if key in variables and isinstance(variables[key], np.ndarray) and variables[key].ndim == 2),
        None,
    )
    if signal is None:
        raise ValueError(f"{filename}: expected a 2-D Clean_data, Data, or data array.")
    try:
        signal = np.asarray(signal, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{filename}: EEG array must contain numeric values.") from exc
    if signal.shape[0] != 32 and signal.shape[1] == 32:
        signal = signal.T
    if signal.shape[0] != 32:
        raise ValueError(f"{filename}: expected 32 channels, found array shape {signal.shape}.")
    if signal.shape[1] < 640:
        raise ValueError(f"{filename}: needs at least 640 samples, found {signal.shape[1]}.")
    if not np.isfinite(signal).all():
        raise ValueError(f"{filename}: EEG array contains NaN or infinite values.")
    return signal


def read_sam40_mat(contents: bytes, filename: str) -> np.ndarray:
    """Read one SAM40-style MATLAB recording for model inference."""
    return _signal_from_mat(contents, filename)


def read_sam40_zip(contents: bytes) -> list[Recording]:
    try:
        archive = zipfile.ZipFile(BytesIO(contents))
    except zipfile.BadZipFile as exc:
        raise ValueError("Dataset upload must be a valid ZIP archive.") from exc

    with archive:
        entries = [item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith(".mat")]
        if not entries:
            raise ValueError("ZIP must contain SAM40 .mat EEG recordings.")
        if len(entries) > MAX_ARCHIVE_FILES:
            raise ValueError(f"ZIP contains more than {MAX_ARCHIVE_FILES} MAT files.")
        if sum(item.file_size for item in archive.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP expands beyond the 1 GB safety limit.")

        records: list[Recording] = []
        paths: set[str] = set()
        for item in entries:
            if item.file_size > MAX_RECORDING_BYTES:
                raise ValueError(f"{item.filename}: MAT file exceeds the 100 MB limit.")
            relative_path = item.filename.replace("\\", "/")
            filename = relative_path.rsplit("/", 1)[-1]
            if not filename or relative_path in paths:
                raise ValueError(f"ZIP contains an empty or duplicate path: {relative_path!r}.")
            paths.add(relative_path)
            label = _label(filename)
            if label is None:
                # Data.zip contains utility MAT files in artifact_removal. They are
                # not labelled evaluation recordings, so they are excluded.
                continue
            signal = _signal_from_mat(archive.read(item), filename)
            records.append(Recording(filename=relative_path, label=label, signal=signal))
        if not records:
            raise ValueError(
                "No labelled recordings were found. File names must include Relax, Arithmetic, Stress, _lo, or _hi."
            )
    return records
