"""BCI Competition IV 2a EEGNet inference for one GDF recording."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Any

import mne
import numpy as np
import torch
from torch import nn


INTENT_CLASSES = ("Left hand", "Right hand", "Feet", "Tongue")
INTENT_CHANNELS = 22
INTENT_SAMPLING_RATE = 250
INTENT_SAMPLES = 1001


class EEGNetModel(nn.Module):
    """EEGNet architecture used to create the supplied intent checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 32), padding="same", bias=False),
            nn.BatchNorm2d(16),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (INTENT_CHANNELS, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), groups=32, padding="same", bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(0.5),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((INTENT_SAMPLES // (8 * 16)) * 32, len(INTENT_CLASSES))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.block1(value)
        value = self.block2(value)
        value = self.block3(value)
        return self.fc(self.flatten(value))


def load_intent_model(checkpoint: Path, device: torch.device) -> nn.Module:
    """Load only a tensor state dictionary matching the known EEGNet checkpoint."""
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not safely read the EEGNet checkpoint: {exc}") from exc
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Intent checkpoint must be a tensor-only EEGNet state dictionary.")

    model = EEGNetModel()
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "Intent checkpoint weights do not match the BCI IV 2a EEGNet architecture."
        ) from exc
    return model.to(device).eval()


def _read_trials(
    contents: bytes,
    filename: str,
) -> tuple[np.ndarray, list[int]]:
    with tempfile.TemporaryDirectory(prefix="bci-intent-") as temporary_directory:
        gdf_path = Path(temporary_directory) / "recording.gdf"
        gdf_path.write_bytes(contents)
        try:
            raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Could not read {filename} as a BCI IV 2a GDF file: {exc}") from exc

        if raw.info["sfreq"] != INTENT_SAMPLING_RATE:
            raise ValueError(
                f"{filename}: EEGNet requires {INTENT_SAMPLING_RATE} Hz data; "
                f"this file uses {raw.info['sfreq']:g} Hz."
            )

        eog_names = {
            "eog-left",
            "eog-central",
            "eog-right",
        }
        eog_channels = [
            channel for channel in raw.ch_names if channel.strip().lower() in eog_names
        ]
        if eog_channels:
            raw.drop_channels(eog_channels)
        if len(raw.ch_names) != INTENT_CHANNELS:
            raise ValueError(
                f"{filename}: expected 22 EEG channels after dropping EOG channels, "
                f"found {len(raw.ch_names)}."
            )

        raw.filter(l_freq=4, h_freq=40, method="iir", verbose="ERROR")
        raw.notch_filter(freqs=50, verbose="ERROR")
        events, annotation_ids = mne.events_from_annotations(raw, verbose="ERROR")

        description_to_label = {
            "769": 0,
            "7": 0,
            "770": 1,
            "8": 1,
            "771": 2,
            "9": 2,
            "772": 3,
            "10": 3,
        }
        selected_ids: dict[str, int] = {}
        for description, event_id in annotation_ids.items():
            description = str(description).strip()
            label = description_to_label.get(description)
            if label is not None:
                selected_ids[INTENT_CLASSES[label]] = event_id
            elif description == "783":
                selected_ids["Unlabelled cue"] = event_id

        if not selected_ids:
            raise ValueError(
                f"{filename}: no BCI IV 2a motor-imagery cue events were found. "
                "Expected cues 769-772 (labelled) or 783 (unlabelled)."
            )

        event_ids = set(selected_ids.values())
        selected_events = events[np.isin(events[:, 2], list(event_ids))]
        epochs = mne.Epochs(
            raw,
            selected_events,
            event_id=selected_ids,
            tmin=0,
            tmax=4,
            reject=None,
            baseline=None,
            preload=True,
            verbose="ERROR",
        )
        trials = np.asarray(epochs.get_data(), dtype=np.float32)
        if trials.size == 0:
            raise ValueError(f"{filename}: no complete four-second motor-imagery trials were found.")
        if trials.shape[1:] != (INTENT_CHANNELS, INTENT_SAMPLES):
            raise ValueError(
                f"{filename}: expected trials shaped 22 x {INTENT_SAMPLES}, "
                f"found {trials.shape[1]} x {trials.shape[2]}."
            )
        start_samples = [int(sample) for sample in epochs.events[:, 0]]

    standard_deviation = float(trials.std())
    if not np.isfinite(trials).all() or standard_deviation < 1e-8:
        raise ValueError(f"{filename}: EEG trials contain invalid or constant values.")
    trials = (trials - trials.mean()) / standard_deviation
    return trials[:, np.newaxis, :, :], start_samples


def predict_intent(
    checkpoint: Path,
    contents: bytes,
    filename: str,
) -> dict[str, Any]:
    """Predict the motor-imagery class independently for each GDF cue epoch."""
    started = time.perf_counter()
    trials, start_samples = _read_trials(contents, filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_intent_model(checkpoint, device)
    inputs = torch.from_numpy(trials)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()

    trial_predictions = []
    for index, values in enumerate(probabilities):
        predicted = int(values.argmax())
        trial_predictions.append({
            "trial": index + 1,
            "start_sample": start_samples[index],
            "predicted_label": predicted,
            "predicted_class": INTENT_CLASSES[predicted],
            "confidence": float(values[predicted]),
            "class_probabilities": {
                name: float(values[class_index])
                for class_index, name in enumerate(INTENT_CLASSES)
            },
        })

    mean_probabilities = probabilities.mean(axis=0)
    predicted = int(mean_probabilities.argmax())
    return {
        "task": "intent",
        "profile": "bci2a_eegnet_v1",
        "evaluation_unit": "single_recording_inference",
        "filename": filename,
        "predicted_label": predicted,
        "predicted_class": INTENT_CLASSES[predicted],
        "confidence": float(mean_probabilities[predicted]),
        "class_probabilities": {
            name: float(mean_probabilities[index])
            for index, name in enumerate(INTENT_CLASSES)
        },
        "trials": len(trial_predictions),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "device": str(device),
        "configuration": {
            "channels": INTENT_CHANNELS,
            "sampling_rate_hz": INTENT_SAMPLING_RATE,
            "window_samples": INTENT_SAMPLES,
            "epoch_seconds": 4,
            "bandpass_hz": [4, 40],
            "notch_hz": 50,
            "normalization": "global_zscore_over_trials_in_uploaded_recording",
            "aggregation": "mean_trial_probability",
        },
        "trial_predictions": trial_predictions,
        "interpretation_note": (
            "Predictions are per-trial model outputs. The displayed score is not "
            "measured accuracy or clinical confidence."
        ),
    }
