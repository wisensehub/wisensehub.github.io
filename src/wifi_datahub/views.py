from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


TIME_AXIS_NAMES = {"time", "packet", "time_bin"}
SIGNAL_ARRAYS = {
    "amplitude", "csi_real", "csi_imag", "phase_rad", "power_db_rel",
    "official_normalized_amplitude", "official_normalized_bvp", "bvp",
}
LABEL_ARRAYS = {
    "source_label", "activity_label", "task_label", "subject", "environment", "experiment",
}


@dataclass(frozen=True)
class ViewOptions:
    target_rate_hz: Optional[float] = None
    duration_s: Optional[float] = None
    target_length: Optional[int] = None
    interpolation: str = "linear"
    layout: str = "canonical"
    links: Optional[int] = None
    subcarriers: Optional[int] = None
    tx_links: Optional[int] = None
    rx_links: Optional[int] = None
    profile: Optional[str] = None
    window_mode: str = "segment"

    def requested(self) -> bool:
        return any((
            self.target_rate_hz is not None,
            self.duration_s is not None,
            self.target_length is not None,
            self.interpolation != "linear",
            self.layout != "canonical",
            self.links is not None,
            self.subcarriers is not None,
            self.tx_links is not None,
            self.rx_links is not None,
            self.profile is not None,
            self.window_mode != "segment",
        ))


# Task-family profiles: same rate/length policy within a family; keep native L/S.
TASK_PROFILES: Dict[str, ViewOptions] = {
    "task-auto": ViewOptions(profile="task-auto"),
    "general-sensing": ViewOptions(
        target_rate_hz=100.0,
        duration_s=3.0,
        interpolation="linear",
        layout="canonical",
        profile="general-sensing",
    ),
    "vital-sign": ViewOptions(
        target_rate_hz=10.0,
        duration_s=30.0,
        interpolation="linear",
        layout="canonical",
        profile="vital-sign",
    ),
    "semantic-bvp": ViewOptions(
        target_length=22,
        interpolation="none",
        layout="canonical",
        profile="semantic-bvp",
    ),
}

# Legacy catalog profile names → task family.
PROFILE_ALIASES: Dict[str, str] = {
    "task-auto": "task-auto",
    "general-sensing": "general-sensing",
    "vital-sign": "vital-sign",
    "semantic-bvp": "semantic-bvp",
    "clip-4s-100hz": "general-sensing",
    "clip-3s-100hz": "general-sensing",
    "clip-3s-native": "general-sensing",
    "clip-1.8s-100hz": "general-sensing",
    "continuous-100hz": "general-sensing",
    "continuous-native": "general-sensing",
    "native-10hz": "vital-sign",
    "vital-native-plus-20hz": "vital-sign",
    "clip-250-packets": "general-sensing",
    "clip-500-packets": "general-sensing",
    "clip-native": "general-sensing",
}

DEFAULT_PROFILE = "general-sensing"


def resolve_task_profile(name: Optional[str]) -> ViewOptions:
    """Map a catalog/CLI profile name to concrete view options."""
    if not name:
        return TASK_PROFILES[DEFAULT_PROFILE]
    key = PROFILE_ALIASES.get(name, name)
    if key not in TASK_PROFILES:
        known = ", ".join(sorted(TASK_PROFILES))
        raise ValueError(f"unknown task profile {name!r}; choose one of: {known}")
    return TASK_PROFILES[key]


# Backward-compatible alias used by older call sites.
CANONICAL_VIEW = TASK_PROFILES[DEFAULT_PROFILE]


def _load_sidecar(path: Path) -> Dict[str, object]:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def _primary_array(files: Iterable[str], sidecar: Dict[str, object]) -> str:
    preferred = sidecar.get("standard_representation")
    if isinstance(preferred, str) and preferred in files:
        return preferred
    for name in ("amplitude", "csi_real", "bvp"):
        if name in files:
            return name
    return next(iter(files))


def _time_axis(axes: Sequence[str], shape: Sequence[int]) -> Optional[int]:
    for index, axis in enumerate(axes):
        if axis in TIME_AXIS_NAMES:
            return index
    if axes and axes[0] == "sample" and len(shape) >= 2:
        return 1
    if len(shape) >= 3:
        return 0
    return None


def _source_rate(sidecar: Dict[str, object], arrays: Dict[str, np.ndarray], time_axis: int) -> Optional[float]:
    value = sidecar.get("sample_rate_hz")
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    timestamp = arrays.get("timestamp_s")
    if timestamp is not None and timestamp.ndim == 1 and timestamp.size >= 2:
        diffs = np.diff(timestamp.astype(np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return 1.0 / float(np.median(diffs))
    return None


def _target_count(length: int, source_rate_hz: Optional[float], options: ViewOptions) -> int:
    if options.target_length is not None:
        if options.target_length <= 0:
            raise ValueError("target_length must be positive")
        return int(options.target_length)
    rate = options.target_rate_hz if options.target_rate_hz is not None else source_rate_hz
    if options.duration_s is not None:
        if options.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if rate is None:
            raise ValueError("--duration requires --target-rate, a sidecar sample_rate_hz, or timestamp_s")
        return max(1, int(round(float(options.duration_s) * float(rate))))
    if options.target_rate_hz is not None:
        if source_rate_hz is None:
            # Assume packets already sit on the target grid; windowing may still change shape.
            return length
        return max(1, int(round(length * float(options.target_rate_hz) / float(source_rate_hz))))
    return length


def _rate_converted_length(length: int, source_rate_hz: Optional[float], target_rate_hz: Optional[float]) -> int:
    if target_rate_hz is None:
        return length
    if source_rate_hz is None or abs(float(source_rate_hz) - float(target_rate_hz)) < 1e-9:
        return length
    return max(1, int(round(length * float(target_rate_hz) / float(source_rate_hz))))


def view_source_info(input_path: Path, options: ViewOptions) -> Dict[str, object]:
    """Inspect the source time axis and its length after rate conversion."""
    sidecar = _load_sidecar(input_path)
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    primary = _primary_array(arrays.keys(), sidecar)
    axes = list(sidecar.get("axis_order") or [])
    if not axes:
        axes = ["sample", "time", "link", "subcarrier"] if arrays[primary].ndim == 4 else ["time", "link", "subcarrier"]
    time_axis = _time_axis(axes, arrays[primary].shape)
    if time_axis is None:
        raise ValueError(f"cannot identify a time axis for primary array {primary} with axes {axes}")
    source_length = int(arrays[primary].shape[time_axis])
    source_rate = _source_rate(sidecar, arrays, time_axis)
    rate_length = _rate_converted_length(source_length, source_rate, options.target_rate_hz)
    return {
        "primary": primary,
        "axes": axes,
        "source_length": source_length,
        "source_rate_hz": source_rate,
        "rate_length": rate_length,
    }


def _nearest_indices(source_count: int, target_count: int) -> np.ndarray:
    if target_count == 1:
        return np.asarray([0], dtype=np.int64)
    positions = np.linspace(0, source_count - 1, target_count)
    return np.rint(positions).astype(np.int64)


def _linear_resample(value: np.ndarray, axis: int, target_count: int) -> np.ndarray:
    source_count = value.shape[axis]
    if source_count == target_count:
        return value.copy()
    if source_count == 1:
        return np.repeat(value, target_count, axis=axis)
    moved = np.moveaxis(value, axis, 0)
    flat = moved.reshape(source_count, -1)
    source_x = np.linspace(0.0, 1.0, source_count)
    target_x = np.linspace(0.0, 1.0, target_count)
    result = np.empty((target_count, flat.shape[1]), dtype=np.float32 if np.issubdtype(value.dtype, np.floating) else value.dtype)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_x, source_x, flat[:, column])
    result = result.reshape((target_count,) + moved.shape[1:])
    return np.moveaxis(result, 0, axis).astype(value.dtype, copy=False)


def _resize_time(value: np.ndarray, axis: int, target_count: int, interpolation: str) -> np.ndarray:
    source_count = value.shape[axis]
    if source_count == target_count:
        return value.copy()
    if interpolation == "none":
        slices = [slice(None)] * value.ndim
        keep = min(source_count, target_count)
        slices[axis] = slice(0, keep)
        cropped = value[tuple(slices)]
        if keep == target_count:
            return cropped.copy()
        pad_width = [(0, 0)] * value.ndim
        pad_width[axis] = (0, target_count - keep)
        return np.pad(cropped, pad_width, mode="constant")
    if interpolation == "nearest" or not np.issubdtype(value.dtype, np.floating):
        return np.take(value, _nearest_indices(source_count, target_count), axis=axis)
    if interpolation == "linear":
        return _linear_resample(value, axis, target_count)
    raise ValueError(f"unsupported interpolation {interpolation!r}; choose none, nearest, or linear")


def _resize_mask(mask: np.ndarray, axis: int, target_count: int, interpolation: str) -> np.ndarray:
    if mask.shape[axis] == target_count:
        return mask.copy()
    if interpolation == "none":
        resized = _resize_time(mask.astype(np.uint8), axis, target_count, "none").astype(bool)
        source_count = mask.shape[axis]
        if target_count > source_count:
            slices = [slice(None)] * resized.ndim
            slices[axis] = slice(source_count, None)
            resized[tuple(slices)] = False
        return resized
    return np.take(mask, _nearest_indices(mask.shape[axis], target_count), axis=axis).astype(bool)


def _segment_time(
    value: np.ndarray, axes: Sequence[str], window_length: int, *, pad_value: object = 0,
) -> tuple[np.ndarray, list[str], int, int, np.ndarray]:
    """Split the complete time axis into non-overlapping windows without dropping data."""
    axes = list(axes)
    time_axis = _time_axis(axes, value.shape)
    if time_axis is None:
        raise ValueError(f"cannot identify time axis for shape {value.shape} and axes {axes}")
    sample_axis = axes.index("sample") if "sample" in axes and len(axes) == value.ndim else None
    leading = [time_axis] if sample_axis is None else [sample_axis, time_axis]
    order = leading + [index for index in range(value.ndim) if index not in leading]
    moved = np.transpose(value, order)
    if sample_axis is None:
        moved = moved[np.newaxis, ...]
    sample_count, source_length = int(moved.shape[0]), int(moved.shape[1])
    windows_per_sample = max(1, int(np.ceil(source_length / float(window_length))))
    padded_length = windows_per_sample * window_length
    if padded_length != source_length:
        pad_width = [(0, 0)] * moved.ndim
        pad_width[1] = (0, padded_length - source_length)
        moved = np.pad(moved, pad_width, mode="constant", constant_values=pad_value)
    segmented = moved.reshape((sample_count, windows_per_sample, window_length) + moved.shape[2:])
    segmented = segmented.reshape((sample_count * windows_per_sample, window_length) + moved.shape[2:])
    keep_sample_axis = sample_axis is not None or windows_per_sample > 1
    remaining_axes = [axes[index] for index in order if index not in {sample_axis, time_axis}]
    output_time_axis = "time_bin" if axes[time_axis] == "time_bin" else "time"
    output_axes = (
        ["sample", output_time_axis, *remaining_axes]
        if keep_sample_axis else [output_time_axis, *remaining_axes]
    )
    if not keep_sample_axis:
        segmented = segmented[0]
    valid_lengths = np.minimum(
        window_length,
        np.maximum(0, source_length - np.arange(windows_per_sample, dtype=np.int64) * window_length),
    )
    return segmented, output_axes, sample_count, windows_per_sample, valid_lengths


def _segments_from_labels(labels: np.ndarray, sample_rate_hz: Optional[float]) -> list[dict[str, object]]:
    values = np.asarray(labels).reshape(-1)
    if values.size <= 1:
        return []
    rate = float(sample_rate_hz) if sample_rate_hz else 1.0
    segments: list[dict[str, object]] = []
    start = 0
    current = str(values[0])
    for index in range(1, values.size):
        label = str(values[index])
        if label == current:
            continue
        segments.append({
            "start_seconds": start / rate,
            "end_seconds": index / rate,
            "label": current,
            "source_label": current,
        })
        start = index
        current = label
    segments.append({
        "start_seconds": start / rate,
        "end_seconds": values.size / rate,
        "label": current,
        "source_label": current,
    })
    return segments


def _normalize_channels(
    value: np.ndarray, axes: Sequence[str], target_links: Optional[int], target_subcarriers: Optional[int],
    target_tx_links: Optional[int] = None, target_rx_links: Optional[int] = None,
) -> np.ndarray:
    result = np.asarray(value)
    axes = list(axes)
    if target_links is not None and "link" in axes:
        link_axis = axes.index("link")
        observed_links = result.shape[link_axis]
        if observed_links > target_links:
            if target_links == 1:
                result = np.mean(result, axis=link_axis, keepdims=True)
            else:
                slices = [slice(None)] * result.ndim
                slices[link_axis] = slice(0, target_links)
                result = result[tuple(slices)]
        elif observed_links < target_links:
            pad_width = [(0, 0)] * result.ndim
            pad_width[link_axis] = (0, target_links - observed_links)
            result = np.pad(result, pad_width, mode="constant")
    if target_subcarriers is not None and "subcarrier" in axes:
        sub_axis = axes.index("subcarrier")
        observed = result.shape[sub_axis]
        if observed > target_subcarriers:
            start = (observed - target_subcarriers) // 2
            slices = [slice(None)] * result.ndim
            slices[sub_axis] = slice(start, start + target_subcarriers)
            result = result[tuple(slices)]
        elif observed < target_subcarriers:
            pad = target_subcarriers - observed
            before = pad // 2
            pad_width = [(0, 0)] * result.ndim
            pad_width[sub_axis] = (before, pad - before)
            result = np.pad(result, pad_width, mode="constant")
    for axis_name, target in (("tx_link", target_tx_links), ("rx_link", target_rx_links)):
        if target is None or axis_name not in axes:
            continue
        axis = axes.index(axis_name)
        observed = result.shape[axis]
        if observed > target:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(0, target)
            result = result[tuple(slices)]
        elif observed < target:
            pad_width = [(0, 0)] * result.ndim
            pad_width[axis] = (0, target - observed)
            result = np.pad(result, pad_width, mode="constant")
    return result


def _dimension_mask(observed: int, target: Optional[int], *, centered: bool = False) -> np.ndarray:
    """Mark preserved source positions after deterministic crop/pad."""
    size = observed if target is None else target
    mask = np.zeros(size, dtype=bool)
    keep = min(observed, size)
    start = (size - keep) // 2 if centered and size > observed else 0
    mask[start:start + keep] = True
    return mask


def _resize_coordinate(value: np.ndarray, observed: int, target: Optional[int], *, centered: bool = False) -> np.ndarray:
    flat = np.asarray(value).reshape(-1)
    if flat.size != observed or target is None or target == observed:
        return flat.copy()
    if observed > target:
        start = (observed - target) // 2 if centered else 0
        return flat[start:start + target].copy()
    pad = target - observed
    before = pad // 2 if centered else 0
    return np.pad(flat, (before, pad - before), mode="constant", constant_values=-1)


def _reshape_link_subcarrier(value: np.ndarray, axes: Sequence[str], options: ViewOptions) -> tuple[np.ndarray, list[str]]:
    axes = list(axes)
    if options.layout == "canonical":
        return value, axes
    time_axis = _time_axis(axes, value.shape)
    sample_axis = axes.index("sample") if "sample" in axes and len(axes) == value.ndim else None
    keep = [axis for axis in (sample_axis, time_axis) if axis is not None]
    feature_axes = [index for index in range(value.ndim) if index not in keep]
    order = keep + feature_axes
    moved = np.transpose(value, order)
    leading = moved.shape[:len(keep)]
    feature_size = int(np.prod(moved.shape[len(keep):], dtype=np.int64))
    flat = moved.reshape(leading + (feature_size,))
    return flat, [axes[index] for index in keep] + ["feature"]


def create_standard_view(input_path: Path, output_path: Path, options: ViewOptions) -> Path:
    if options.target_rate_hz is not None and options.target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    for name, count in (
        ("links", options.links), ("subcarriers", options.subcarriers),
        ("tx_links", options.tx_links), ("rx_links", options.rx_links),
    ):
        if count is not None and count <= 0:
            raise ValueError(f"{name} must be positive")
    if options.interpolation not in {"none", "nearest", "linear"}:
        raise ValueError("--interpolation must be none, nearest, or linear")
    if options.layout not in {"canonical", "flat", "link-subcarrier"}:
        raise ValueError("--layout must be canonical, flat, or link-subcarrier")
    if options.window_mode != "segment":
        raise ValueError("window_mode must be segment")
    if not options.requested():
        raise ValueError("no view options were requested")

    sidecar = _load_sidecar(input_path)
    source = np.load(input_path, allow_pickle=False)
    arrays = {name: source[name] for name in source.files}
    primary = _primary_array(arrays.keys(), sidecar)
    axes = list(sidecar.get("axis_order") or [])
    if not axes:
        axes = ["sample", "time", "link", "subcarrier"] if arrays[primary].ndim == 4 else ["time", "link", "subcarrier"]
    time_axis = _time_axis(axes, arrays[primary].shape)
    if time_axis is None:
        raise ValueError(f"cannot identify a time axis for primary array {primary} with axes {axes}")
    source_length = arrays[primary].shape[time_axis]
    source_rate = _source_rate(sidecar, arrays, time_axis)
    # Preserve the complete recording: resample the full time axis first, then
    # split it into consecutive windows. Only the final remainder is padded.
    rate_length = _rate_converted_length(source_length, source_rate, options.target_rate_hz)
    target_length = _target_count(source_length, source_rate, options)
    semantic_time_bins = bool(
        sidecar.get("canonical_tensor_exception") and "time_bin" in axes and source_rate is None
    )
    # BVP time bins are not packet samples. Do not turn an unknown bin cadence
    # into a fictitious 100 Hz sampling rate merely because the task profile
    # has a CSI default.
    target_rate = None if semantic_time_bins else (
        options.target_rate_hz if options.target_rate_hz is not None else source_rate
    )
    source_has_sample = "sample" in axes and len(axes) == arrays[primary].ndim
    source_sample_count = int(arrays[primary].shape[axes.index("sample")]) if source_has_sample else 1
    primary_output_axes: list[str] | None = None
    windows_per_sample = 1
    window_valid_lengths = np.asarray([min(rate_length, target_length)], dtype=np.int64)

    output_arrays: Dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        value_axes = axes if value.shape == arrays[primary].shape else None
        if name in SIGNAL_ARRAYS and value_axes:
            resized = _resize_time(value, time_axis, rate_length, options.interpolation)
            segmented, segmented_axes, _, segment_count, valid_lengths = _segment_time(
                resized, value_axes, target_length,
            )
            segmented = _normalize_channels(
                segmented, segmented_axes, options.links, options.subcarriers,
                options.tx_links, options.rx_links,
            )
            reshaped, new_axes = _reshape_link_subcarrier(segmented, segmented_axes, options)
            output_arrays[name] = reshaped
            if name == primary:
                primary_output_axes = new_axes
                windows_per_sample = segment_count
                window_valid_lengths = valid_lengths
        elif name == "valid_mask":
            mask_axes = ["sample", "time"] if source_has_sample and value.ndim >= 2 and value.shape[0] == source_sample_count else ["time"]
            mask_axis = mask_axes.index("time")
            resized = _resize_mask(value.astype(bool), mask_axis, rate_length, options.interpolation)
            output_arrays[name] = _segment_time(
                resized, mask_axes, target_length, pad_value=False,
            )[0].astype(bool)
        elif name in {"packet_index", "timestamp_s"} and value.ndim == 1 and value.size == source_length:
            if name == "timestamp_s" and target_rate:
                timeline = np.arange(rate_length, dtype=np.float64) / float(target_rate)
            elif name == "timestamp_s":
                timeline = np.linspace(float(value[0]), float(value[-1]), rate_length, dtype=np.float64)
            else:
                timeline = np.arange(rate_length, dtype=np.int32)
            pad_value = np.nan if name == "timestamp_s" else -1
            segmented = _segment_time(timeline, ["time"], target_length, pad_value=pad_value)[0]
            if source_has_sample and windows_per_sample > 0:
                segmented = np.tile(segmented, (source_sample_count, 1))
            output_arrays[name] = segmented
        elif name in LABEL_ARRAYS and value.ndim == 1 and value.size == source_length:
            resampled = np.take(value, _nearest_indices(source_length, rate_length))
            segmented = _segment_time(resampled, ["time"], target_length, pad_value="")[0]
            if source_has_sample and windows_per_sample > 0:
                segmented = np.tile(segmented, (source_sample_count, 1))
            output_arrays[name] = segmented
        elif name == "subcarrier_index" and "subcarrier" in axes:
            observed = int(arrays[primary].shape[axes.index("subcarrier")])
            output_arrays[name] = _resize_coordinate(value, observed, options.subcarriers, centered=True)
        elif name == "tx_link_index" and "tx_link" in axes:
            observed = int(arrays[primary].shape[axes.index("tx_link")])
            output_arrays[name] = _resize_coordinate(value, observed, options.tx_links)
        elif name == "rx_link_index" and "rx_link" in axes:
            observed = int(arrays[primary].shape[axes.index("rx_link")])
            output_arrays[name] = _resize_coordinate(value, observed, options.rx_links)
        elif source_has_sample and value.ndim >= 1 and value.shape[0] == source_sample_count:
            output_arrays[name] = np.repeat(value, windows_per_sample, axis=0)
        else:
            output_arrays[name] = value

    # Wrapped phase angles cannot be interpolated as ordinary scalars. When
    # complex components are available, derive the view phase from the
    # resampled real/imaginary CSI instead of preserving a false jump at ±π.
    if {"phase_rad", "csi_real", "csi_imag"}.issubset(output_arrays):
        real = np.asarray(output_arrays["csi_real"])
        imag = np.asarray(output_arrays["csi_imag"])
        if real.shape == imag.shape == np.asarray(output_arrays["phase_rad"]).shape:
            output_arrays["phase_rad"] = np.arctan2(imag, real).astype(np.float32)

    if "valid_mask" not in output_arrays:
        native_mask = np.ones((source_sample_count, rate_length), dtype=bool) if source_has_sample else np.ones(rate_length, dtype=bool)
        native_mask_axes = ["sample", "time"] if source_has_sample else ["time"]
        output_arrays["valid_mask"] = _segment_time(
            native_mask, native_mask_axes, target_length, pad_value=False,
        )[0].astype(bool)

    output_arrays["source_sample_index"] = np.repeat(
        np.arange(source_sample_count, dtype=np.int32), windows_per_sample,
    )
    output_arrays["window_index"] = np.tile(
        np.arange(windows_per_sample, dtype=np.int32), source_sample_count,
    )
    output_arrays["window_start_index"] = output_arrays["window_index"].astype(np.int64) * target_length
    output_arrays["window_valid_length"] = np.tile(window_valid_lengths, source_sample_count)
    output_arrays["window_end_index"] = output_arrays["window_start_index"] + output_arrays["window_valid_length"]

    if "subcarrier" in axes:
        observed = int(arrays[primary].shape[axes.index("subcarrier")])
        output_arrays["subcarrier_mask"] = _dimension_mask(observed, options.subcarriers, centered=True)
    if "tx_link" in axes:
        observed = int(arrays[primary].shape[axes.index("tx_link")])
        output_arrays["tx_link_mask"] = _dimension_mask(observed, options.tx_links)
    if "rx_link" in axes:
        observed = int(arrays[primary].shape[axes.index("rx_link")])
        output_arrays["rx_link_mask"] = _dimension_mask(observed, options.rx_links)

    primary_value = output_arrays[primary]
    output_axes = primary_output_axes or axes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output_arrays)
    metadata = dict(sidecar)
    if "source_label" in output_arrays:
        resampled = np.asarray(output_arrays["source_label"]).reshape(-1)
        resampled = resampled[np.asarray([str(item) != "" for item in resampled], dtype=bool)]
        if output_arrays["source_label"].ndim == 1 and resampled.size > 1 and len({str(item) for item in resampled}) > 1:
            metadata["segments"] = _segments_from_labels(resampled, target_rate)
        else:
            labels = dict(metadata.get("labels") or {})
            if resampled.size and len({str(item) for item in resampled}) == 1:
                labels.setdefault("activity", str(resampled[0]))
            if labels:
                metadata["labels"] = labels
    transformations = list(metadata.get("transformations") or [])
    if {"phase_rad", "csi_real", "csi_imag"}.issubset(output_arrays):
        transformations.append("recompute wrapped phase from resampled real/imaginary CSI")
    transformations.append(
        f"derived segmented view: profile={options.profile}, window_length={target_length}, "
        f"target_rate_hz={target_rate}, duration_s={options.duration_s}, "
        f"interpolation={options.interpolation}, layout={options.layout}, "
        f"links={options.links}, subcarriers={options.subcarriers}, "
        f"tx_links={options.tx_links}, rx_links={options.rx_links}, dropped_time_steps=0"
    )
    metadata.update({
        "schema_version": metadata.get("schema_version", "1.0"),
        "derived_from": str(input_path),
        "standard_representation": primary,
        "shape": list(primary_value.shape),
        "axis_order": output_axes,
        "sample_rate_hz": target_rate,
        "source_sample_rate_hz": source_rate,
        "target_rate_assumed": bool(
            options.target_rate_hz is not None and source_rate is None and not semantic_time_bins
        ),
        "target_rate_ignored_for_semantic_bins": semantic_time_bins,
        "duration_s": (float(target_length) / float(target_rate)) if target_rate else metadata.get("duration_s"),
        "profile": options.profile,
        "window_mode": "segment",
        "window_length": target_length,
        "source_time_steps": source_length,
        "resampled_time_steps": rate_length,
        "windows_per_source_sample": windows_per_sample,
        "output_window_count": int(source_sample_count * windows_per_sample),
        "dropped_time_steps": 0,
        "full_source_preserved": True,
        "view_options": {
            "profile": options.profile,
            "target_rate_hz": options.target_rate_hz,
            "duration_s": options.duration_s,
            "target_length": options.target_length,
            "interpolation": options.interpolation,
            "layout": options.layout,
            "links": options.links,
            "subcarriers": options.subcarriers,
            "tx_links": options.tx_links,
            "rx_links": options.rx_links,
            "window_mode": options.window_mode,
        },
        "transformations": transformations,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
