"""Supported SAM40 model profiles and model loaders."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import torch
from torch import nn


@dataclass(frozen=True)
class ModelProfile:
    """The input contract needed to reproduce a known checkpoint."""

    id: str
    channels: int
    samples: int
    bands: int
    sampling_rate_hz: int | None


SAM40_STANDARD = ModelProfile(
    id="sam40_shallowconvnet_v1", channels=32, samples=640, bands=1, sampling_rate_hz=320
)
SAM40_MULTIBAND = ModelProfile(
    id="sam40_multiband_shallowconvnet_v1", channels=32, samples=640, bands=4, sampling_rate_hz=320
)
WESAD_FEATURES = (
    "mean_hr",
    "rmssd",
    "pnn50",
    "lf_hf",
    "eda_mean",
    "eda_std",
    "eda_slope",
    "scr_count",
    "scr_amplitude_mean",
    "temp_mean",
    "temp_slope",
    "acc_mean",
    "acc_std",
)


class ShallowConvNet(nn.Module):
    """The shallow ConvNet used by the supported SAM40 checkpoints."""

    def __init__(self, channels: int = 32, samples: int = 640, bands: int = 1, classes: int = 2):
        super().__init__()
        self.temporal = nn.Conv2d(bands, 40, kernel_size=(1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.bn2 = nn.BatchNorm2d(40)
        self.drop = nn.Dropout(0.5)
        with torch.no_grad():
            flattened = self._features(torch.zeros(1, bands, channels, samples)).shape[1]
        self.fc = nn.Linear(flattened, classes)

    def _features(self, value: torch.Tensor) -> torch.Tensor:
        value = self.temporal(value)
        value = self.spatial(value)
        value = self.bn1(value).square()
        value = self.pool(value)
        value = torch.log(torch.clamp(value, min=1e-7))
        value = self.bn2(value)
        return self.drop(value).flatten(1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc(self._features(value))


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for name in ("model_state", "state_dict"):
            if isinstance(checkpoint.get(name), dict):
                checkpoint = checkpoint[name]
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        raise ValueError("Only tensor state dictionaries are supported by this model profile.")
    return checkpoint


def _profile_from_state(state: dict[str, torch.Tensor]) -> ModelProfile:
    """Identify only the input contracts this evaluator can reproduce safely."""
    try:
        temporal = state["temporal.weight"]
        spatial = state["spatial.weight"]
        classifier = state["fc.weight"]
    except KeyError as exc:
        raise ValueError("Checkpoint is missing a required ShallowConvNet weight tensor.") from exc

    if temporal.ndim != 4 or spatial.ndim != 4 or classifier.ndim != 2:
        raise ValueError("Checkpoint tensors do not describe a supported ShallowConvNet.")
    bands = temporal.shape[1]
    channels = spatial.shape[2]
    feature_count = classifier.shape[1]

    if (bands, channels, feature_count, classifier.shape[0]) == (1, 32, 1480, 2):
        return SAM40_STANDARD
    if (bands, channels, feature_count, classifier.shape[0]) == (4, 32, 1480, 2):
        return SAM40_MULTIBAND
    if bands == 4 and channels == 12:
        raise ValueError(
            "This is a 12-channel STEW-transfer checkpoint. It needs the STEW 12-electrode "
            "dataset adapter and its saved preprocessing metadata; it cannot be evaluated "
            "against the current 32-channel SAM40 ZIP profile."
        )
    raise ValueError(
        "Checkpoint is not a supported profile. Supported checkpoints are the 32-channel, "
        "640-sample SAM40 ShallowConvNet with one or four input bands."
    )


def load_model(contents: bytes | Path, device: torch.device) -> tuple[nn.Module, ModelProfile]:
    """Load a supported tensor-only checkpoint and return its verified input profile."""
    try:
        source = BytesIO(contents) if isinstance(contents, bytes) else contents
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not safely read the .pth checkpoint: {exc}") from exc
    state = _state_dict(checkpoint)
    profile = _profile_from_state(state)
    model = ShallowConvNet(
        channels=profile.channels, samples=profile.samples, bands=profile.bands
    )
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint weights do not match the identified SAM40 ShallowConvNet profile."
        ) from exc
    model.to(device).eval()
    return model, profile


def _load_sklearn_artifact(path: Path) -> tuple[Any, list[str] | None]:
    """Load a trusted local scikit-learn estimator and its optional feature metadata."""
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Could not load the .pkl model: {exc}") from exc

    model = artifact
    feature_columns = None
    if isinstance(artifact, dict):
        model = artifact.get("model")
        candidate = artifact.get("feature_cols")
        if isinstance(candidate, (list, tuple)) and all(isinstance(item, str) for item in candidate):
            feature_columns = list(candidate)

    if not callable(getattr(model, "predict", None)):
        raise ValueError(
            "The .pkl file does not contain a supported scikit-learn estimator. "
            "A packaged artifact may store the estimator under its 'model' key."
        )
    feature_count = getattr(model, "n_features_in_", len(feature_columns) if feature_columns else None)
    if feature_columns is not None and len(feature_columns) != feature_count:
        raise ValueError("The .pkl feature_cols metadata does not match the estimator input size.")

    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        raise ValueError("The .pkl model must expose exactly two classes.")

    if feature_count == len(WESAD_FEATURES):
        if feature_columns is not None and tuple(feature_columns) != WESAD_FEATURES:
            raise ValueError("The WESAD .pkl model feature_cols do not match the supported WESAD feature schema.")
        if set(class_indices(classes).values()) != {0, 1}:
            raise ValueError("The WESAD .pkl model must identify baseline and stress as classes 0 and 1.")
        return model, list(WESAD_FEATURES)

    expected_features = SAM40_STANDARD.channels * SAM40_STANDARD.samples
    if feature_count != expected_features:
        model_name = artifact.get("model_name") if isinstance(artifact, dict) else None
        prefix = f"{model_name} " if isinstance(model_name, str) else "This model "
        raise ValueError(
            f"{prefix}is not compatible with this EEG dataset: it expects "
            f"{feature_count!r} engineered wearable-sensor features, but SAM40 EEG windows "
            f"have {expected_features} values ({SAM40_STANDARD.channels} channels x "
            f"{SAM40_STANDARD.samples} samples). The WESAD model requires WESAD wrist-sensor "
            "features (heart rate/HRV, EDA, temperature, and acceleration); it cannot predict "
            "from this app's EEG .mat files. Use a SAM40 EEG-trained .pth or .pkl model."
        )

    class_indices(classes)
    return model, feature_columns


def load_sklearn_model(path: Path) -> Any:
    """Load a trusted scikit-learn estimator that accepts flattened SAM40 EEG windows."""
    model, feature_columns = _load_sklearn_artifact(path)
    expected_features = SAM40_STANDARD.channels * SAM40_STANDARD.samples
    feature_count = getattr(model, "n_features_in_", None)
    if feature_columns == list(WESAD_FEATURES) or feature_count != expected_features:
        raise ValueError(
            "This .pkl model expects WESAD wrist-sensor features, not flattened SAM40 EEG windows."
        )
    return model


def load_wesad_model(path: Path) -> tuple[Any, list[str]]:
    """Load the packaged WESAD estimator and its ordered physiological features."""
    model, feature_columns = _load_sklearn_artifact(path)
    if feature_columns != list(WESAD_FEATURES):
        raise ValueError(
            "The .pkl model does not match the supported WESAD physiological feature schema."
        )
    return model, feature_columns


def class_indices(classes: Any) -> dict[Any, int]:
    indices: dict[Any, int] = {}
    for label in classes:
        if label in (0, "0", "Relax", "relax"):
            indices[label] = 0
        elif label in (1, "1", "Stress", "stress", "Stress / Arithmetic", "stress / arithmetic"):
            indices[label] = 1
        else:
            raise ValueError(
                "The .pkl model classes must identify class 0 as Relax and class 1 as Stress."
            )
    if set(indices.values()) != {0, 1}:
        raise ValueError(
            "The .pkl model classes must identify class 0 as Relax and class 1 as Stress."
        )
    return indices
