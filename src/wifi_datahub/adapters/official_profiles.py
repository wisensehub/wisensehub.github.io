from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .generic import convert_generic, load_generic_source
from .xrf55_schema import (
    XRF55_ACTION_NAMES,
    XRF55_RX_LINKS,
    XRF55_SAMPLE_RATE_HZ,
    XRF55_SUBCARRIERS,
    XRF55_TIME_SAMPLES,
    parse_xrf55_clip_id,
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mat(path: Path) -> Dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat  # type: ignore
        return {key: np.asarray(value) for key, value in loadmat(path).items() if not key.startswith("__")}
    except (ImportError, NotImplementedError, ValueError):
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MAT v7.3 input requires the optional h5py dependency") from exc
        result: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    result[name.split("/")[-1]] = np.asarray(obj)
            handle.visititems(visit)
        return result


def _find(mapping: Dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray | None:
    lowered = {key.lower(): key for key in mapping}
    for name in names:
        if name.lower() in lowered:
            return np.asarray(mapping[lowered[name.lower()]])
    return None


def _largest_numeric(mapping: Dict[str, np.ndarray], minimum_ndim: int = 2) -> np.ndarray:
    candidates = [
        np.asarray(value) for value in mapping.values()
        if np.asarray(value).ndim >= minimum_ndim and np.issubdtype(np.asarray(value).dtype, np.number)
    ]
    if not candidates:
        raise ValueError(f"no numeric CSI array found; available keys: {sorted(mapping)}")
    return max(candidates, key=lambda value: value.size)


def _path_metadata_labels(input_path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    csi_bench_tasks = {
        "BreathingDetection", "FallDetection", "Localization",
        "MotionSourceRecognition", "HumanActivityRecognition",
        "HumanIdentification", "ProximityRecognition", "Multitask",
    }
    for part in input_path.parts:
        if part in csi_bench_tasks:
            labels["task"] = part
        elif part.startswith("act_"):
            labels["activity"] = part[len("act_") :]
        elif part.startswith("user_"):
            labels["subject"] = part[len("user_") :]
        elif part.startswith("env_"):
            labels["environment"] = part[len("env_") :]
        elif part.startswith("sub_"):
            labels["subject_type"] = part[len("sub_") :]
        elif part.startswith("Diff_"):
            labels["difficulty"] = part[len("Diff_") :]
        elif part.startswith("motionsrc_"):
            raw = part[len("motionsrc_") :]
            labels["location"] = raw.lstrip("0") or "0"
    path_text = "/".join(input_path.parts)
    # CSI-Bench tasks use different official primary labels.
    if "HumanIdentification" in path_text and "subject" in labels:
        labels["class"] = labels["subject"]
    elif "MotionSourceRecognition" in path_text and "subject_type" in labels:
        labels["class"] = labels["subject_type"]
    elif "Localization" in path_text and "location" in labels:
        labels["class"] = labels["location"]
    elif "HumanActivityRecognition" in path_text and "activity" in labels:
        activity = labels["activity"]
        labels["class"] = "walking" if activity.startswith("walking") else activity
    elif "ProximityRecognition" in path_text and "activity" in labels:
        # Distance is not encoded in the path; keep activity as a fallback class.
        labels["class"] = labels["activity"]
    elif "activity" in labels:
        activity = labels["activity"]
        if activity.startswith("walking"):
            labels["activity"] = "walking"
            labels["class"] = "walking"
        else:
            labels.setdefault("class", activity)
    return labels


def _segments_from_packet_labels(labels: np.ndarray, sample_rate_hz: float | None) -> list[dict[str, object]]:
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


def _clip_labels_from_arrays(
    arrays: Dict[str, np.ndarray], input_path: Path, dataset_id: str, sample_rate_hz: float | None,
) -> tuple[Dict[str, object], list[dict[str, object]]]:
    labels: Dict[str, object] = {}
    if dataset_id == "csi-bench":
        labels.update(_path_metadata_labels(input_path))
    for key, value in arrays.items():
        if key.startswith("config_") and np.asarray(value).ndim == 0:
            labels[key[len("config_") :]] = str(np.asarray(value))
    for name in ("activity_label", "subject", "environment", "experiment"):
        if name in arrays:
            item = np.asarray(arrays[name])
            if item.ndim == 0 or item.size == 1:
                labels[name.replace("_label", "")] = str(item.reshape(-1)[0])
    if "source_label" in arrays:
        source = np.asarray(arrays["source_label"]).reshape(-1)
        if source.size == 1:
            labels.setdefault("activity", str(source[0]))
        elif source.size > 1 and len({str(item) for item in source}) == 1:
            labels.setdefault("activity", str(source[0]))
    segments: list[dict[str, object]] = []
    if "source_label" in arrays:
        source = np.asarray(arrays["source_label"]).reshape(-1)
        if source.size > 1 and len({str(item) for item in source}) > 1:
            segments = _segments_from_packet_labels(source, sample_rate_hz)
    return labels, segments


def _save(
    input_path: Path, output_path: Path, dataset_id: str, arrays: Dict[str, np.ndarray],
    primary: str, axes: list[str], source_representation: str, transformations: list[str],
    sample_rate_hz: float | None = None, power_unit: str = "source_amplitude_arbitrary_unit",
    metadata_extra: Dict[str, object] | None = None,
) -> Path:
    value = np.asarray(arrays[primary])
    if value.ndim < 3:
        raise ValueError(f"standardized primary array must be at least 3-D, got {value.shape}")
    if axes[0] == "sample":
        time_length = value.shape[1]
        valid_shape = value.shape[:2]
    else:
        time_length = value.shape[0]
        valid_shape = (time_length,)
    arrays.setdefault("packet_index", np.arange(time_length, dtype=np.int32))
    arrays.setdefault("valid_mask", np.ones(valid_shape, dtype=bool))
    arrays = {key: np.asarray(item) for key, item in arrays.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    clip_labels, segments = _clip_labels_from_arrays(arrays, input_path, dataset_id, sample_rate_hz)
    sidecar = {
        "schema_version": "1.0", "dataset_id": dataset_id,
        "source_file": input_path.name, "source_sha256": _sha256(input_path),
        "source_representation": source_representation,
        "standard_representation": primary, "shape": list(value.shape), "axis_order": axes,
        "sample_rate_hz": sample_rate_hz, "time_axis": "packet_index", "power_unit": power_unit,
        "transformations": transformations, "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "wisensehub-0.6.0",
    }
    if metadata_extra:
        sidecar.update(metadata_extra)
    if sample_rate_hz:
        sidecar["duration_s"] = time_length / float(sample_rate_hz)
    if clip_labels:
        sidecar["labels"] = clip_labels
    if segments:
        sidecar["segments"] = segments
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def _complex_arrays(value: np.ndarray) -> Dict[str, np.ndarray]:
    value = np.asarray(value)
    if np.iscomplexobj(value):
        return {
            "csi_real": value.real.astype(np.float32),
            "csi_imag": value.imag.astype(np.float32),
            "amplitude": np.abs(value).astype(np.float32),
            "phase_rad": np.angle(value).astype(np.float32),
        }
    return {"amplitude": value.astype(np.float32)}


def _canonical_csi_bench_tensor(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return CSI-Bench CSI as [N,] time, subcarrier, tx_link, rx_link.

    CSI-Bench source files expose either a batch of [time, subcarrier]
    samples or [subcarrier, time, flattened_link]. The release does not
    provide a reliable Tx/Rx factorization for the flattened link axis, so
    we preserve every link and represent it explicitly as 1 Tx x L Rx.
    """
    data = np.asarray(data)
    if data.ndim == 4:
        if data.shape[-1] in (56, 64, 114, 168, 208, 232, 30, 52, 100) or data.shape[1] > data.shape[0]:
            # Existing supported layout: [sample, time, flattened_link, subcarrier].
            return np.transpose(data, (0, 1, 3, 2))[:, :, :, None, :], [
                "sample", "time", "subcarrier", "tx_link", "rx_link",
            ]
        raise ValueError(f"CSI-Bench 4-D tensor has unsupported shape {data.shape}")
    if data.ndim != 3:
        raise ValueError(f"CSI-Bench expects a 3-D or 4-D CSI tensor, got {data.shape}")
    if data.shape[0] <= 32 and data.shape[1] > data.shape[0]:
        return data[:, :, :, None, None], ["sample", "time", "subcarrier", "tx_link", "rx_link"]
    # Common CSI-Bench window layout: [subcarrier, time, link]
    if data.shape[2] <= 16 and data.shape[1] >= 32 and data.shape[0] >= 16 and data.shape[0] != data.shape[1]:
        return np.transpose(data, (1, 0, 2))[:, :, None, :], ["time", "subcarrier", "tx_link", "rx_link"]
    if data.shape[0] in (56, 64, 114, 168, 208, 232) and data.shape[1] >= data.shape[0]:
        return np.transpose(data, (1, 0, 2))[:, :, None, :], ["time", "subcarrier", "tx_link", "rx_link"]
    if data.shape[-1] in (56, 64, 114, 168, 208, 232, 30, 52, 100) and data.shape[1] >= data.shape[0]:
        return np.transpose(data, (0, 2, 1))[:, :, None, :], ["time", "subcarrier", "tx_link", "rx_link"]
    raise ValueError(f"CSI-Bench tensor layout is not recognized: {data.shape}")


def convert_csi_bench_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    data = _find(mapping, ("CSI_amps", "X", "csi", "CSI"))
    data = data if data is not None else _largest_numeric(mapping, 2)
    data, axes = _canonical_csi_bench_tensor(data)
    arrays = _complex_arrays(data) if np.iscomplexobj(data) else {"amplitude": data.astype(np.float32)}
    arrays["subcarrier_index"] = np.arange(data.shape[axes.index("subcarrier")], dtype=np.int32)
    arrays["tx_link_index"] = np.arange(data.shape[axes.index("tx_link")], dtype=np.int32)
    arrays["rx_link_index"] = np.arange(data.shape[axes.index("rx_link")], dtype=np.int32)
    path_labels = _path_metadata_labels(input_path)
    if path_labels.get("class"):
        # Use the task's official primary label as activity for sidecars / by_label.
        path_labels["activity"] = path_labels["class"]
    if path_labels.get("activity"):
        arrays["activity_label"] = np.asarray(path_labels["activity"], dtype="U64")
    if path_labels.get("task"):
        arrays["task_label"] = np.asarray(path_labels["task"], dtype="U64")
    return _save(
        input_path, output_path, "csi-bench", arrays, "amplitude", axes,
        "complex_csi" if np.iscomplexobj(data) else "processed_amplitude",
        [
            "load official CSI_amps/X tensor",
            "transpose to [time, subcarrier, tx_link, rx_link] canonical axes",
            "preserve flattened source links as one Tx by L Rx",
            "attach path-derived task and activity labels",
            "cast float32",
        ],
        metadata_extra={
            "antenna_mapping": "single_tx_flattened_rx",
            "antenna_mapping_assumption": (
                "The source exposes a flattened link axis without a reliable Tx/Rx factorization; "
                "WiSenseHub preserves all links as tx_link=1 and rx_link=source_link_count."
            ),
        },
    )


def convert_mmfi_directory(input_path: Path, output_path: Path) -> Path:
    files = sorted(input_path.glob("frame*.mat"))
    if not files:
        raise ValueError(f"MM-Fi wifi-csi directory contains no frame*.mat files: {input_path}")
    frames = []
    for path in files:
        mapping = _load_mat(path)
        frame = _find(mapping, ("CSIamp",))
        if frame is None:
            raise ValueError(f"MM-Fi frame lacks CSIamp: {path.name}")
        frame = np.asarray(frame, dtype=np.float64).squeeze()
        if frame.ndim != 3 or frame.shape[:2] != (3, 114):
            raise ValueError(
                f"MM-Fi CSIamp must be [3 Rx,114 subcarriers,packet captures], got {frame.shape}"
            )
        # Match the official reader's invalid-value repair, then unfold every
        # packet capture into time so none of the released CSI is discarded.
        for packet_index in range(frame.shape[2]):
            packet = frame[:, :, packet_index]
            finite = np.isfinite(packet)
            fill = float(packet[finite].mean()) if finite.any() else 0.0
            packet[~finite] = fill
        frames.append(frame.transpose(2, 1, 0)[:, :, None, :])  # [10,S,Tx=1,Rx=3]
    amplitude = np.concatenate(frames, axis=0).astype(np.float32)  # [T,S,Tx,Rx]
    activity = next((part for part in reversed(input_path.parts) if re.fullmatch(r"A\d+", part)), None)
    arrays: Dict[str, np.ndarray] = {"amplitude": amplitude}
    if activity:
        arrays["activity_label"] = np.asarray(activity)
    activity_names = {
        "A01": "stretching and relaxing", "A02": "chest expansion (horizontal)",
        "A03": "chest expansion (vertical)", "A04": "twist left", "A05": "twist right",
        "A06": "mark time", "A07": "limb extension left", "A08": "limb extension right",
        "A09": "lunge left-front", "A10": "lunge right-front", "A11": "limb extension both",
        "A12": "squat", "A13": "raising hand left", "A14": "raising hand right",
        "A15": "lunge left side", "A16": "lunge right side", "A17": "waving hand left",
        "A18": "waving hand right", "A19": "picking up things", "A20": "throwing left",
        "A21": "throwing right", "A22": "kicking left", "A23": "kicking right",
        "A24": "body extension left", "A25": "body extension right", "A26": "jumping up",
        "A27": "bowing",
    }
    if activity:
        arrays["activity_label"] = np.asarray(activity_names.get(activity, activity))
    label_sets = {"Activity": [activity_names.get(activity, activity)]} if activity else {}
    return _save(input_path, output_path, "mm-fi", arrays, "amplitude",
                 ["time", "subcarrier", "tx_link", "rx_link"], "processed_amplitude",
                 ["load sorted frame*.mat CSIamp [3,114,10]",
                  "replace non-finite packet values with the finite packet mean",
                  "unfold all 10 packet captures per synchronized frame into time",
                  "transpose to [time,114 subcarriers,1 Tx,3 Rx]"],
                 sample_rate_hz=100.0, metadata_extra={
                     "label_sets": label_sets,
                     "source_axis_order": ["frame", "rx_link", "subcarrier", "packet_within_frame"],
                     "source_shape": [len(files), 3, 114, 10],
                     "source_frame_rate_hz": 10.0,
                     "packet_captures_per_frame": 10,
                     "sample_rate_evidence": "10 synchronized frames/s × 10 released CSI packet captures/frame",
                     "antenna_layout": {"tx_links": 1, "rx_links": 3},
                     "antenna_mapping_evidence": "official MM-Fi CSIamp layout",
                 })


def convert_ntu_fi_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    data = _find(mapping, ("CSIamp",))
    if data is None:
        raise ValueError("NTU-Fi MAT file must contain CSIamp")
    data = np.asarray(data).squeeze()
    if data.ndim == 2 and data.shape[0] == 342:
        source = data.reshape(3, 114, data.shape[1])
    elif data.ndim == 3 and data.shape[0:2] == (3, 114):
        source = data
    elif data.ndim == 3 and data.shape[-2:] == (3, 114):
        source = data.transpose(1, 2, 0)
    else:
        raise ValueError(f"NTU-Fi expected [3,114,time], got {data.shape}")
    # Match the released SenseFi loader exactly: retain packets 0,4,8,... and
    # reshape the three antenna streams to canonical 1 Tx x 3 Rx. Preserve the
    # released amplitude as the primary signal and expose SenseFi's published
    # normalization as a separate, selectable array.
    sampled = source[:, :, ::4]
    amplitude = sampled.transpose(2, 1, 0)[:, :, None, :].astype(np.float32)
    official_normalized = ((amplitude - 42.3199) / 4.9802).astype(np.float32)
    label = input_path.parent.name
    identity_clip = "ntu-fi-humanid" in str(input_path).lower()
    arrays = {
        "amplitude": amplitude,
        "official_normalized_amplitude": official_normalized,
        "subcarrier_index": np.arange(114, dtype=np.int32),
        "tx_link_index": np.arange(1, dtype=np.int32),
        "rx_link_index": np.arange(3, dtype=np.int32),
    }
    if identity_clip:
        arrays["subject"] = np.asarray(label)
    else:
        arrays["activity_label"] = np.asarray(label)
    return _save(input_path, output_path, "ntu-fi", arrays, "amplitude",
                 ["time", "subcarrier", "tx_link", "rx_link"], "processed_amplitude",
                 [
                     "load official CSIamp [3 antennas,114 subcarriers,2000 packets]",
                     "retain every fourth packet following the official SenseFi loader",
                     "reshape to canonical [T,S,Tx,Rx] = [500,114,1,3]",
                     "preserve source amplitude and add official normalization (x - 42.3199) / 4.9802",
                 ],
                 metadata_extra={
                     "label_sets": {"Identity" if identity_clip else "Activity": [label]},
                     "source_axis_order": ["antenna_stream", "subcarrier", "packet"],
                     "source_shape": list(source.shape),
                     "antenna_layout": {"tx_links": 1, "rx_links": 3},
                     "antenna_mapping_evidence": "SenseFi exposes three antenna streams; retained as 1 Tx x 3 Rx",
                     "official_packet_downsample_factor": 4,
                     "official_normalization": {
                         "array": "official_normalized_amplitude",
                         "formula": "(amplitude - 42.3199) / 4.9802",
                         "mean": 42.3199,
                         "standard_deviation": 4.9802,
                     },
                     "sample_rate_evidence": "Physical packet cadence is not reported in the processed MAT files",
                 })


def convert_widar_csv(input_path: Path, output_path: Path) -> Path:
    value = np.genfromtxt(input_path, delimiter=",")
    if value.size != 22 * 20 * 20:
        raise ValueError(f"SenseFi Widar BVP CSV must contain 8800 values, got {value.size}")
    bvp = value.reshape(22, 20, 20).astype(np.float32)
    official_normalized_bvp = ((bvp - 0.0025) / 0.0119).astype(np.float32)
    label = re.sub(r"^\d+-", "", input_path.parent.name).replace("_", " ")
    subject_match = re.match(r"^(user\d+)-", input_path.name, re.IGNORECASE)
    split = next((part for part in input_path.parts if part.lower() in {"train", "test"}), None)
    arrays = {
        "bvp": bvp,
        "official_normalized_bvp": official_normalized_bvp,
        "activity_label": np.asarray(label),
    }
    if subject_match:
        arrays["subject"] = np.asarray(subject_match.group(1).lower())
    if split:
        arrays["source_split"] = np.asarray(split.lower())
    return _save(input_path, output_path, "widar3", arrays, "bvp",
                 ["time_bin", "velocity_x_bin", "velocity_y_bin"], "processed_bvp",
                 [
                     "load authentic SenseFi Widar BVP CSV",
                     "reshape source [22 time bins, 400 flattened velocity bins] to [22,20,20]",
                     "retain the released BVP and add SenseFi's fixed normalized companion",
                 ], power_unit="bvp_power_distribution",
                 metadata_extra={
                     "label_sets": {"Gesture": [label]},
                     "time_axis": "time_bin",
                     "source_axis_order": ["time_bin", "flattened_velocity_bin"],
                     "source_shape": [22, 400],
                     "labels": {
                         "gesture": label,
                         **({"subject": subject_match.group(1).lower()} if subject_match else {}),
                         **({"split": split.lower()} if split else {}),
                     },
                     "official_normalization": {
                         "array": "official_normalized_bvp",
                         "formula": "(bvp - 0.0025) / 0.0119",
                         "mean": 0.0025,
                         "standard_deviation": 0.0119,
                         "source": "SenseFi Widar_Dataset loader",
                     },
                     "sample_rate_evidence": "The processed SenseFi CSV exposes 22 semantic BVP snapshots but no physical cadence",
                     "canonical_tensor_exception": {
                         "reason": "Widar3 stores processed body-velocity profiles, not raw CSI.",
                         "axis_order": ["time_bin", "velocity_x_bin", "velocity_y_bin"],
                     },
                 })


def convert_three_rooms_directory(input_path: Path, output_path: Path) -> Path:
    data_path = input_path / "data.csv" if input_path.is_dir() else input_path
    matrix = np.genfromtxt(data_path, delimiter=",")
    matrix = np.atleast_2d(matrix)
    for subcarriers in (114, 56):
        for links in (4, 3, 1):
            if matrix.shape[1] >= subcarriers * (1 + 2 * links):
                amp = matrix[:, subcarriers:subcarriers * (1 + links)].reshape(matrix.shape[0], links, subcarriers)
                phase = matrix[:, subcarriers * (1 + links):subcarriers * (1 + 2 * links)].reshape(matrix.shape[0], links, subcarriers)
                arrays: Dict[str, np.ndarray] = {"amplitude": amp.astype(np.float32), "phase_rad": phase.astype(np.float32)}
                label_path = data_path.with_name("label.csv")
                if label_path.exists():
                    labels = np.genfromtxt(label_path, delimiter=",", dtype="U64")
                    labels = np.atleast_2d(labels)
                    arrays["source_label"] = labels[:matrix.shape[0], 1]
                return _save(data_path, output_path, "figshare-csi-har", arrays, "amplitude",
                             ["time", "link", "subcarrier"], "amplitude_phase",
                             ["apply official CSV column offsets", "reshape antenna pairs", "attach sibling label.csv"])
    raise ValueError(f"Three Rooms data.csv has unsupported width {matrix.shape[1]}")


def convert_signfi_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("csid_lab", "csid_home", "csi", "CSI"))
    if csi is None:
        complex_values = [value for value in mapping.values() if np.asarray(value).ndim == 4]
        if not complex_values:
            raise ValueError(f"SignFi CSI tensor not found; keys: {sorted(mapping)}")
        csi = max(complex_values, key=lambda value: np.asarray(value).size)
    csi = np.asarray(csi)
    if csi.ndim != 4:
        raise ValueError(f"SignFi expects [time,subcarrier,link,sample], got {csi.shape}")
    source_shape = list(csi.shape)
    if csi.shape[0] != 200 or csi.shape[1] != 30 or csi.shape[2] != 3:
        raise ValueError(
            "SignFi expects the official [200 time,30 subcarrier,3 Rx,N sample] "
            f"tensor, got {csi.shape}"
        )
    # The official release stores [T,S,Rx,N].  Add the documented singleton
    # Tx axis and write the hub contract directly as [N,T,S,Tx,Rx].
    canonical = csi.transpose(3, 0, 1, 2)[:, :, :, None, :]
    arrays = _complex_arrays(canonical)
    label = _find(mapping, ("label_lab", "label_home", "label", "labels"))
    if label is not None:
        arrays["source_label"] = np.asarray(label).reshape(-1)
    return _save(input_path, output_path, "signfi", arrays, "amplitude",
                 ["sample", "time", "subcarrier", "tx_link", "rx_link"], "complex_csi",
                 ["load official csid tensor", "transpose [T,S,Rx,N] to [N,T,S,Tx,Rx]",
                  "insert the documented singleton Tx axis", "derive amplitude and phase"],
                 metadata_extra={
                     "source_axis_order": ["time", "subcarrier", "rx_link", "sample"],
                     "source_shape": source_shape,
                     "antenna_layout": {"tx_links": 1, "rx_links": 3},
                     "antenna_mapping_evidence": "official SignFi tensor schema",
                     "source_rate_status": "not reported by the official release page",
                 })


def _canonical_sequence(value: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    value = np.asarray(value).squeeze()
    if value.ndim == 1:
        return value[:, None, None], ["time", "link", "subcarrier"]
    if value.ndim == 2:
        return value[:, None, :], ["time", "link", "subcarrier"]
    if value.ndim == 3:
        return value, ["time", "link", "subcarrier"]
    if value.ndim == 4 and value.shape[-1] in (10, 30, 56, 114):
        return value.reshape(value.shape[0], -1, value.shape[-1]), ["time", "link", "subcarrier"]
    if value.ndim == 4:
        return value, ["sample", "time", "link", "subcarrier"]
    raise ValueError(f"unsupported official processed CSI shape {value.shape}")


def convert_wimans(input_path: Path, output_path: Path) -> Path:
    if input_path.suffix.lower() == ".npy":
        value = np.load(input_path, allow_pickle=False)
    else:
        mapping = _load_mat(input_path)
        value = _find(mapping, ("CSI", "csi", "CSIamp", "amp", "amplitude"))
        value = value if value is not None else _largest_numeric(mapping, 2)
    value = np.asarray(value).squeeze()
    if value.ndim != 4 or value.shape[1:3] != (3, 3) or value.shape[-1] != 30:
        raise ValueError(
            "WiMANS CSI amplitude must use the official "
            f"[time,3 Tx,3 Rx,30 subcarriers] layout, got {value.shape}"
        )
    # The release stores [T,Tx,Rx,S].  Only transpose named axes; do not
    # flatten the antenna grid and later guess how to factor it again.
    canonical = value.transpose(0, 3, 1, 2)
    # Every recording lasts exactly three seconds.  Packet loss makes T lower
    # than the 3000 transmitted packets, so T/3 is the truthful received-packet
    # cadence for resampling the complete clip back to a 3 s task grid.
    effective_sample_rate_hz = float(canonical.shape[0]) / 3.0
    axes = ["time", "subcarrier", "tx_link", "rx_link"]
    arrays = _complex_arrays(canonical)
    arrays["timestamp_s"] = (
        np.arange(canonical.shape[0], dtype=np.float64) / effective_sample_rate_hz
    )
    arrays["subcarrier_index"] = np.arange(30, dtype=np.int32)
    arrays["tx_link_index"] = np.arange(3, dtype=np.int32)
    arrays["rx_link_index"] = np.arange(3, dtype=np.int32)
    metadata: Dict[str, object] = {
        "source_axis_order": ["time", "tx_link", "rx_link", "subcarrier"],
        "source_shape": list(value.shape),
        "antenna_layout": {"tx_links": 3, "rx_links": 3},
        "antenna_mapping": "released Tx and Rx axes",
        "nominal_sample_rate_hz": 1000.0,
        "fixed_clip_duration_s": 3.0,
        "sample_rate_evidence": (
            "official paper: 3000 packets in 3 seconds at 1000 packets/s; "
            "effective received-packet rate is T/3 because shorter clips reflect packet loss"
        ),
    }
    annotation_path = next(
        (parent / "annotation.csv" for parent in input_path.parents if (parent / "annotation.csv").exists()),
        None,
    )
    if annotation_path:
        with annotation_path.open(newline="", encoding="utf-8-sig") as handle:
            row = next((item for item in csv.DictReader(handle) if item.get("label") == input_path.stem), None)
        if row:
            activities = list(dict.fromkeys(
                value for key, value in row.items() if key.startswith("user_") and key.endswith("_activity") and value
            ))
            locations = list(dict.fromkeys(
                value for key, value in row.items() if key.startswith("user_") and key.endswith("_location") and value
            ))
            if activities:
                arrays["activity_label"] = np.asarray(" + ".join(activities))
            occupancy = int(row["number_of_users"])
            label_sets: Dict[str, list[str]] = {
                "Occupancy": [f"{occupancy} {'person' if occupancy == 1 else 'people'}"],
                "Environment": [row["environment"]],
                "WiFi band": [f"{row['wifi_band']} GHz"],
            }
            if activities:
                label_sets["Activity"] = activities
            if locations:
                label_sets["Location"] = locations
            metadata.update({
                "annotation": row,
                "label_sets": label_sets,
            })
    if "activity_label" not in arrays:
        arrays["activity_label"] = np.asarray(input_path.stem)
    return _save(input_path, output_path, "wimans", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 ["load official WiMANS amplitude [T,3 Tx,3 Rx,30 subcarriers]",
                  "join official annotation.csv by sample label",
                  "transpose to canonical [T,S,Tx,Rx]", "cast float32"],
                 sample_rate_hz=effective_sample_rate_hz,
                 metadata_extra=metadata)


def convert_xrf55_npy(input_path: Path, output_path: Path) -> Path:
    value = np.load(input_path, allow_pickle=False)
    expected_shape = (XRF55_RX_LINKS * XRF55_SUBCARRIERS, XRF55_TIME_SAMPLES)
    if value.shape != expected_shape:
        raise ValueError(
            "XRF55 WiFi NPY must use the official [270 flattened channels, "
            f"1000 time samples] layout, got {value.shape}"
        )

    clip = parse_xrf55_clip_id(input_path)
    scene_match = next(
        (
            match for part in reversed(input_path.parts)
            if (match := re.fullmatch(r"Scene(\d+)", part, re.IGNORECASE))
        ),
        None,
    )
    scene_id = int(scene_match.group(1)) if scene_match else None
    # The official Q&A defines each consecutive group of 30 rows as the
    # subcarriers for one receiving stream.  Preserve those nine streams
    # without inventing a Tx/Rx factorization: Tx=1 and Rx=9.
    canonical = value.reshape(
        XRF55_RX_LINKS, XRF55_SUBCARRIERS, XRF55_TIME_SAMPLES,
    ).transpose(2, 1, 0)[:, :, None, :]
    axes = ["time", "subcarrier", "tx_link", "rx_link"]
    arrays = _complex_arrays(canonical)
    arrays.update({
        "subcarrier_index": np.arange(XRF55_SUBCARRIERS, dtype=np.int16),
        "tx_link_index": np.arange(1, dtype=np.int8),
        "rx_link_index": np.arange(XRF55_RX_LINKS, dtype=np.int8),
        "activity_id": np.asarray(clip.action_id, dtype=np.int16),
        "activity_label": np.asarray(clip.action_name),
        "subject": np.asarray(str(clip.subject_id)),
        "repetition_id": np.asarray(clip.repetition_id, dtype=np.int16),
        "config_activity_id": np.asarray(str(clip.action_id)),
        "config_repetition": np.asarray(str(clip.repetition_id)),
    })
    return _save(input_path, output_path, "xrf55", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 [
                     "load official XRF55 WiFi NPY stored as [270 flattened channels, 1000 time samples]",
                     "split channels into nine receiving streams with 30 subcarriers following the official Q&A",
                     "transpose to canonical [time, subcarrier, tx_link=1, rx_link=9] axes",
                     "cast signal values to float32",
                 ],
                 sample_rate_hz=XRF55_SAMPLE_RATE_HZ,
                 metadata_extra={
                     "source_shape": list(expected_shape),
                     "source_axis_order": ["flattened_rx_link_subcarrier", "time"],
                     "source_flattened_order": "receiver_device_then_receiving_antenna_then_subcarrier",
                     "antenna_layout": {"tx_links": 1, "rx_links": XRF55_RX_LINKS},
                     "receiver_grouping": {"receiver_devices": 3, "receiving_antennas_per_device": 3},
                     "antenna_mapping": "rx_link_major_then_subcarrier",
                     "antenna_mapping_evidence": (
                         "XRF55 Q&A: each consecutive block of 30 values is one receiving "
                         "antenna's subcarriers; the first 90 values belong to receiver device 1."
                     ),
                     "filename_schema": "subject_id_action_id_repetition_id.npy",
                     "subject_id": clip.subject_id,
                     "scene_id": scene_id,
                     "activity_id": clip.action_id,
                     "activity_name": clip.action_name,
                     "repetition_id": clip.repetition_id,
                     "label_vocabulary": XRF55_ACTION_NAMES,
                     "label_sets": {"Activity": [clip.action_name]},
                 })


def _scalar(mapping: Dict[str, np.ndarray], names: Iterable[str], default: Any = None) -> Any:
    value = _find(mapping, names)
    if value is None or np.asarray(value).size == 0:
        return default
    item = np.asarray(value).reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def convert_ehunam_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("CSI",))
    if csi is None:
        raise ValueError("EHUNAM MAT must contain CSI")
    csi = np.asarray(csi).squeeze()
    if csi.ndim != 2:
        raise ValueError(f"EHUNAM CSI must be [time,subcarrier], got {csi.shape}")
    reported_subcarriers = int(_scalar(mapping, ("Subcarriers",), csi.shape[-1]))
    if csi.shape[0] == reported_subcarriers and csi.shape[1] != reported_subcarriers:
        csi = csi.T
    bandwidth = int(_scalar(mapping, ("BW",), 20))
    environment = str(_scalar(mapping, ("Environment", "Enviroment"), ""))
    if reported_subcarriers == 56:
        remove = []
    elif bandwidth == 20:
        remove = [0, 1, 2, 3, 32, 61, 62, 63]
    elif bandwidth == 80 and "industrial" in environment.lower():
        remove = list(range(0, 6)) + [32] + list(range(59, 70)) + [96] + list(range(123, 134)) + [160] + list(range(187, 198)) + [224] + list(range(251, 256))
    elif bandwidth == 80:
        remove = list(range(0, 6)) + list(range(127, 131)) + list(range(251, 256))
    else:
        raise ValueError(f"unsupported EHUNAM bandwidth/subcarrier combination: {bandwidth}/{reported_subcarriers}")
    keep = np.asarray([index for index in range(csi.shape[1]) if index not in set(remove)], dtype=np.int32)
    csi = csi[:, keep][:, None, :]
    arrays = _complex_arrays(csi)
    arrays["subcarrier_index"] = keep
    source_rate_hz = None
    source_timestamps = _find(mapping, ("TimeStamp", "Timestamp"))
    if source_timestamps is not None and np.asarray(source_timestamps).size == csi.shape[0]:
        timestamp_s = np.asarray(source_timestamps).reshape(-1).astype(np.float64)
        timestamp_s = timestamp_s - timestamp_s[0]
        if timestamp_s.size > 1 and np.all(np.diff(timestamp_s) >= 0) and timestamp_s[-1] > 0:
            arrays["timestamp_s"] = timestamp_s
            source_rate_hz = float((timestamp_s.size - 1) / timestamp_s[-1])
    rssi = _find(mapping, ("RSSI",))
    if rssi is not None:
        arrays["rssi_dbm"] = np.asarray(rssi).reshape(-1).astype(np.float32)

    # EHUNAM's nine filename fields are part of the released annotation
    # schema.  Preserve both the multi-label values (for example ``WG``) and
    # their atomic labels so the website can offer one real clip for every
    # published target instead of reducing the dataset to one task example.
    parts = input_path.stem.split("_")
    label_sets: Dict[str, list[str]] = {}
    metadata: Dict[str, object] = {}
    if len(parts) >= 9:
        campaign, collection_set, receiver, application, people, activity, machines, status, sequence = parts[:9]
        activity_names = {
            "J": "jumping", "W": "walking", "S": "standing",
            "T": "sitting", "G": "sit down / get up", "F": "falling",
        }
        label_sets["Occupancy"] = [f"{0 if people == '#' else len(people)} people"]
        label_sets["Application"] = [application]
        if people != "#":
            label_sets["Identity"] = [f"person {person}" for person in people]
        if activity != "#":
            atomic_activities = [activity_names[code] for code in activity if code in activity_names]
            if atomic_activities:
                label_sets["Activity"] = list(dict.fromkeys(atomic_activities))
                label_sets["Activity combination"] = [" + ".join(atomic_activities)]
                arrays["activity_label"] = np.asarray(" + ".join(atomic_activities))
        if machines != "#":
            label_sets["Machine"] = [f"machine {machine}" for machine in machines if machine.isdigit()]
            label_sets["Machine combination"] = [" + ".join(label_sets["Machine"])]
        if status in {"O", "R"}:
            label_sets["Machine state"] = ["powered on" if status == "O" else "running"]
        metadata.update({
            "campaign": campaign,
            "collection_set": collection_set,
            "receiver": receiver,
            "application": application,
            "sequence": sequence,
            "label_sets": label_sets,
        })
    if source_rate_hz is not None:
        metadata["sample_rate_provenance"] = "inferred from the released TimeStamp span"
    return _save(input_path, output_path, "ehunam", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["load official CSI/BW/Subcarriers/TimeStamp metadata", "remove null and pilot carriers using official Subcarrier.m", "infer effective source rate from the released timestamp span", "derive amplitude and phase"],
                 sample_rate_hz=source_rate_hz,
                 metadata_extra=metadata)


def convert_wifi_presence_json(input_path: Path, output_path: Path) -> Path:
    opener = gzip.open if input_path.name.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count == 0:
        raise ValueError("CSI JSON file contains no records")
    with opener(input_path, "rt", encoding="utf-8") as handle:
        first = json.loads(next(line for line in handle if line.strip()))
    if "t" not in first or "csi" not in first:
        raise ValueError("presence/movement JSON records require t and csi attributes")
    subcarriers = len(first["csi"])
    links = len(first["csi"][0])
    timestamps = np.empty(count, dtype=np.float64)
    real = np.empty((count, links, subcarriers), dtype=np.float32)
    imag = np.empty_like(real)
    with opener(input_path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(item for item in handle if item.strip()):
            record = json.loads(line)
            packet = record["csi"]
            if len(packet) != subcarriers or any(len(row) != links for row in packet):
                raise ValueError(f"inconsistent CSI shape at JSON record {index}")
            timestamps[index] = float(record["t"])
            for subcarrier, values in enumerate(packet):
                for link, value in enumerate(values):
                    real[index, link, subcarrier] = float(value["r"])
                    imag[index, link, subcarrier] = float(value["i"])
    timestamps -= timestamps[0]
    amplitude = np.hypot(real, imag).astype(np.float32)
    positive_deltas = np.diff(timestamps)
    positive_deltas = positive_deltas[positive_deltas > 0]
    sample_rate_hz = float(1.0 / np.median(positive_deltas)) if positive_deltas.size else None
    state = input_path.parent.name.replace("_", " ").title() if input_path.parent.name != "original" else None
    arrays: Dict[str, np.ndarray] = {
        "csi_real": real, "csi_imag": imag, "amplitude": amplitude, "timestamp_s": timestamps,
    }
    if state:
        arrays["activity_label"] = np.asarray(state)
    return _save(input_path, output_path, "wifi-presence-movement",
                 arrays,
                 "amplitude", ["time", "link", "subcarrier"], "complex_csi",
                 ["stream gzip JSON Lines", "parse official r/i complex fields", "transpose subcarrier/link axes", "normalize epoch timestamps to elapsed seconds"],
                 sample_rate_hz=sample_rate_hz,
                 metadata_extra={"label_sets": {"Activity state": [state]} if state else {}})


def _find_ancestor_child(path: Path, child: str) -> Path | None:
    for parent in (path.parent, *path.parents):
        candidate = parent / child
        if candidate.exists():
            return candidate
    return None


WIFI_TAD_CLASS_NAMES = {
    1: "run",
    2: "walk",
    3: "jump",
    4: "wave",
    5: "bend",
    6: "stand",
    7: "sit",
}


def convert_wifi_tad_npy(input_path: Path, output_path: Path) -> Path:
    data = np.load(input_path, allow_pickle=False).squeeze()
    if data.ndim != 2:
        raise ValueError(f"WiFiTAD NPY must be 2-D, got {data.shape}")
    # The public release stores 8,500 time samples by 30 processed feature
    # channels. Its loader transposes that array for model input; the hub keeps
    # time first so annotations remain direct sample indices.
    if data.shape[0] == 30 and data.shape[1] != 30:
        time_first = data.T
    elif data.shape[1] == 30:
        time_first = data
    else:
        raise ValueError(f"WiFiTAD expects one 30-feature axis, got {data.shape}")
    amplitude = time_first[:, None, :].astype(np.float32)
    arrays: Dict[str, np.ndarray] = {
        "amplitude": amplitude,
        "official_normalized_amplitude": amplitude / 40.0,
    }
    annotation_dir = _find_ancestor_child(input_path, "annotations")
    if annotation_dir is not None:
        video_name = input_path.stem
        info: Tuple[float, float] | None = None
        for info_path in sorted(annotation_dir.glob("*_video_info.csv")):
            with info_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.reader(handle):
                    if row and row[0] == video_name and len(row) >= 5:
                        info = (float(row[3]), float(row[4]))
                        break
            if info:
                break
        segments = []
        for anno_path in sorted(annotation_dir.glob("*Annotation*.csv")):
            with anno_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.reader(handle):
                    if row and row[0] == video_name and len(row) >= 5:
                        ratio = info[1] / info[0] if info and info[0] else 1.0
                        try:
                            label_id = int(float(row[2]))
                            if label_id not in WIFI_TAD_CLASS_NAMES:
                                raise ValueError(f"unknown WiFiTAD class id {label_id}")
                            segments.append((
                                float(row[-2]) * ratio,
                                float(row[-1]) * ratio,
                                label_id,
                            ))
                        except (IndexError, ValueError):
                            continue
        if segments:
            arrays["segment_start_index"] = np.asarray([item[0] for item in segments], dtype=np.float32)
            arrays["segment_end_index"] = np.asarray([item[1] for item in segments], dtype=np.float32)
            arrays["segment_label"] = np.asarray([item[2] for item in segments], dtype=np.int16)
            arrays["segment_label_name"] = np.asarray(
                [WIFI_TAD_CLASS_NAMES[item[2]] for item in segments], dtype="U16"
            )
    metadata_extra: Dict[str, object] = {
        "canonical_tensor_exception": {
            "reason": "WiFiTAD stores processed temporal feature channels, not raw CSI subcarriers.",
            "axis_order": ["time", "link", "feature"],
        },
        "label_vocabulary": {
            "activity": [WIFI_TAD_CLASS_NAMES[index] for index in sorted(WIFI_TAD_CLASS_NAMES)],
            "source_class_index": {
                str(index): WIFI_TAD_CLASS_NAMES[index] for index in sorted(WIFI_TAD_CLASS_NAMES)
            },
        },
    }
    if "segment_label" in arrays:
        metadata_extra["segments"] = [
            {
                "start_seconds": float(start) / 100.0,
                "end_seconds": float(end) / 100.0,
                "label": WIFI_TAD_CLASS_NAMES[int(label)],
                "source_label": str(int(label)),
            }
            for start, end, label in zip(
                arrays["segment_start_index"], arrays["segment_end_index"], arrays["segment_label"]
            )
        ]
    return _save(input_path, output_path, "wifi-tad", arrays, "amplitude",
                 ["time", "link", "feature"], "processed_amplitude",
                 ["load official smartwifi NPY", "orient 30 feature channels with time first", "store official amplitude/40 normalization", "attach 100 Hz temporal annotation indices and class names"],
                 sample_rate_hz=100.0, metadata_extra=metadata_extra)


def _column_mapping(value: Any) -> Dict[str, np.ndarray] | None:
    if isinstance(value, dict):
        if any(re.fullmatch(r"tx\d+rx\d+_sub\d+", str(key)) for key in value):
            return {str(key): np.asarray(item).reshape(-1) for key, item in value.items()}
        for item in value.values():
            found = _column_mapping(item)
            if found:
                return found
    if isinstance(value, np.ndarray) and value.dtype.names:
        return {name: np.asarray(value[name]).reshape(-1) for name in value.dtype.names}
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        keys = value[0].keys()
        if any(re.fullmatch(r"tx\d+rx\d+_sub\d+", str(key)) for key in keys):
            return {str(key): np.asarray([row[key] for row in value]).reshape(-1) for key in keys}
    return None


def convert_operanet_mat(input_path: Path, output_path: Path) -> Path:
    try:
        from scipy.io import loadmat  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OPERAnet MAT conversion requires scipy") from exc
    payload = loadmat(input_path, simplify_cells=True)
    columns = _column_mapping(payload)
    if not columns:
        raise ValueError("OPERAnet MAT table with tx#rx#_sub# columns was not found")
    csi_names = [name for name in columns if re.fullmatch(r"tx\d+rx\d+_sub\d+", name)]
    csi_names.sort(key=lambda name: tuple(int(value) for value in re.findall(r"\d+", name)))
    if len(csi_names) != 270:
        raise ValueError(f"OPERAnet WiFi table must contain 270 CSI columns, found {len(csi_names)}")
    flat = np.column_stack([columns[name] for name in csi_names])
    source_csi = flat.reshape(flat.shape[0], 3, 3, 30)
    csi = source_csi.transpose(0, 3, 1, 2)
    arrays = _complex_arrays(csi)
    lowered = {name.lower(): name for name in columns}
    for source_name, target_name in {
        "activity": "source_label", "person_id": "subject", "room_no": "environment", "exp_no": "experiment"
    }.items():
        if source_name in lowered:
            arrays[target_name] = np.char.strip(
                np.asarray(columns[lowered[source_name]]).astype("U64")
            )
    nominal_sample_rate_hz = 1600.0
    sample_rate_hz = None
    sample_rate_provenance = ""
    if "timestamp" in lowered:
        timestamp = np.asarray(columns[lowered["timestamp"]], dtype=np.float64).reshape(-1)
        arrays["timestamp_s"] = (timestamp - timestamp[0]) / 1000.0
        # Timestamps are quantized to whole milliseconds while packets arrive
        # at about 1.6 kHz, so adjacent packets often share a timestamp. The
        # median positive difference would therefore produce a false 1 kHz
        # rate. Estimate effective cadence over the complete excerpt instead.
        span_s = float(arrays["timestamp_s"][-1]) if timestamp.size > 1 else 0.0
        if span_s > 0:
            sample_rate_hz = float((timestamp.size - 1) / span_s)
            sample_rate_provenance = "effective rate inferred from packet count over released timestamp span"
    if sample_rate_hz is None:
        sample_rate_hz = nominal_sample_rate_hz
        arrays["timestamp_s"] = np.arange(csi.shape[0], dtype=np.float64) / sample_rate_hz
        sample_rate_provenance = "official nominal 1600 Hz; excerpt does not contain timestamp metadata"
    label_sets = {}
    if "source_label" in arrays:
        label_sets["Activity"] = sorted({str(item) for item in np.asarray(arrays["source_label"]).reshape(-1)})
    setting_values = {}
    for field, setting in {
        "subject": "Person", "environment": "Room", "experiment": "Experiment",
    }.items():
        if field in arrays:
            setting_values[setting] = sorted({str(item) for item in np.asarray(arrays[field]).reshape(-1)})
    return _save(input_path, output_path, "operanet", arrays, "amplitude",
                 ["time", "subcarrier", "tx_link", "rx_link"], "complex_csi",
                 ["load official MATLAB table", "order tx1rx1_sub1 through tx3rx3_sub30", "reshape and transpose 270 complex fields to [T,S,Tx,Rx]", "attach activity/person/room/experiment metadata"],
                 sample_rate_hz=sample_rate_hz,
                 metadata_extra={
                     "label_sets": label_sets,
                     "setting_values": setting_values,
                     "antenna_layout": {"tx_links": 3, "rx_links": 3},
                     "source_axis_order": ["time", "tx_link", "rx_link", "subcarrier"],
                     "source_shape": list(source_csi.shape),
                     "nominal_sample_rate_hz": nominal_sample_rate_hz,
                     "sample_rate_provenance": sample_rate_provenance,
                 })


def _read_numeric_csv(path: Path) -> np.ndarray:
    value = np.genfromtxt(path, delimiter=",")
    value = np.atleast_2d(value)
    if np.all(np.isnan(value[0])):
        value = value[1:]
    value = value[:, ~np.all(np.isnan(value), axis=0)]
    return value


def convert_nist_breathesmart(input_path: Path, output_path: Path) -> Path:
    name = input_path.name
    if "_csi_real_log.csv" not in name:
        raise ValueError("BreatheSmart discovery expects *_csi_real_log.csv")
    imag_path = input_path.with_name(name.replace("_csi_real_log.csv", "_csi_imag_log.csv"))
    if not imag_path.exists():
        raise ValueError(f"matching imaginary CSI file not found: {imag_path.name}")
    real, imag = _read_numeric_csv(input_path), _read_numeric_csv(imag_path)
    if real.shape != imag.shape:
        raise ValueError(f"BreatheSmart real/imag shapes differ: {real.shape} vs {imag.shape}")
    if real.shape[1] == 1026:
        stored_links, stored_subcarriers = 9, 114
    elif real.shape[1] == 504:
        stored_links, stored_subcarriers = 9, 56
    elif real.shape[1] % 114 == 0:
        stored_links, stored_subcarriers = real.shape[1] // 114, 114
    elif real.shape[1] % 56 == 0:
        stored_links, stored_subcarriers = real.shape[1] // 56, 56
    else:
        raise ValueError(f"BreatheSmart CSI width is not compatible with 56/114 subcarriers: {real.shape[1]}")

    # The NIST CSV envelope reserves 3 x 3 link slots and, in some files, 114
    # carrier slots per link.  The experiment documented by NIST measured only
    # 2 Tx x 3 Rx x 56 subcarriers.  In the official 1026-column files the
    # unmeasured positions are literal zero columns: carrier slots 56..113 and
    # every third flattened link.  Do not present those reserved slots as
    # measured CSI.
    source_real = real.reshape(real.shape[0], stored_links, stored_subcarriers).astype(np.float32)
    source_imag = imag.reshape(imag.shape[0], stored_links, stored_subcarriers).astype(np.float32)
    nonzero = np.logical_or(source_real != 0, source_imag != 0)
    source_subcarrier_mask = np.any(nonzero, axis=(0, 1))
    if not np.any(source_subcarrier_mask):
        raise ValueError("BreatheSmart recording contains no non-zero CSI subcarriers")
    source_real = source_real[:, :, source_subcarrier_mask]
    source_imag = source_imag[:, :, source_subcarrier_mask]
    source_link_mask = np.any(
        np.logical_or(source_real != 0, source_imag != 0), axis=(0, 2),
    )
    active_link_indices = np.flatnonzero(source_link_mask)

    if stored_links == 9 and active_link_indices.tolist() == [0, 1, 3, 4, 6, 7]:
        # Stored order is three receiver groups, each reserving three Tx slots.
        # Select the two measured Tx slots and transpose to [T,S,Tx,Rx].
        real = source_real.reshape(source_real.shape[0], 3, 3, -1)[:, :, :2, :]
        imag = source_imag.reshape(source_imag.shape[0], 3, 3, -1)[:, :, :2, :]
        real = np.transpose(real, (0, 3, 2, 1))
        imag = np.transpose(imag, (0, 3, 2, 1))
    elif stored_links == 6 and active_link_indices.tolist() == list(range(6)):
        # Compact releases may omit the unused third Tx slot altogether while
        # retaining the same Rx-major, then Tx ordering.
        real = np.transpose(source_real.reshape(source_real.shape[0], 3, 2, -1), (0, 3, 2, 1))
        imag = np.transpose(source_imag.reshape(source_imag.shape[0], 3, 2, -1), (0, 3, 2, 1))
    else:
        raise ValueError(
            "BreatheSmart active link slots do not match the documented 2 Tx x 3 Rx layout: "
            f"stored_links={stored_links}, active_slots={active_link_indices.tolist()}"
        )

    measured_subcarriers = int(np.count_nonzero(source_subcarrier_mask))
    if measured_subcarriers != 56:
        raise ValueError(
            "BreatheSmart active subcarriers do not match the documented 56-carrier layout: "
            f"found {measured_subcarriers} of {stored_subcarriers} stored slots"
        )
    arrays: Dict[str, np.ndarray] = {
        "csi_real": real, "csi_imag": imag, "amplitude": np.hypot(real, imag).astype(np.float32),
        "subcarrier_index": np.flatnonzero(source_subcarrier_mask).astype(np.int32),
        "tx_link_index": np.arange(2, dtype=np.int32),
        "rx_link_index": np.arange(3, dtype=np.int32),
        "source_subcarrier_mask": source_subcarrier_mask.astype(bool),
        "source_link_mask": source_link_mask.astype(bool),
    }
    source_rate_hz: float | None = None
    pattern_label: str | None = None
    config_candidates = sorted(input_path.parent.glob("config*.csv")) + sorted(input_path.parent.glob("config*.cvs"))
    if config_candidates:
        with config_candidates[0].open(newline="", encoding="utf-8-sig", errors="ignore") as handle:
            rows = list(csv.reader(handle))
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                key = re.sub(r"[^a-z0-9]+", "_", row[0].strip().lower()).strip("_")
                if key:
                    value = str(row[1]).strip()
                    arrays[f"config_{key}"] = np.asarray(value)
                    if key == "msgfreq":
                        try:
                            source_rate_hz = float(value)
                        except ValueError:
                            pass
                    elif key == "pattern":
                        pattern_label = value
    if not source_rate_hz:
        rate_match = next((re.search(r"FrameRate(\d+(?:\.\d+)?)", part, re.IGNORECASE)
                           for part in input_path.parts if "framerate" in part.lower()), None)
        source_rate_hz = float(rate_match.group(1)) if rate_match else 10.0
    arrays["timestamp_s"] = np.arange(real.shape[0], dtype=np.float64) / source_rate_hz
    if not pattern_label:
        pattern_match = next((re.search(r"BreathingPattern\d+", part, re.IGNORECASE)
                              for part in input_path.parts if "breathingpattern" in part.lower()), None)
        pattern_label = pattern_match.group(0) if pattern_match else None
    return _save(input_path, output_path, "nist-breathesmart", arrays, "amplitude",
                 ["time", "subcarrier", "tx_link", "rx_link"], "complex_csi",
                 [
                     "pair official real and imaginary CSV logs",
                     "discard carrier and link slots that are zero for the complete recording",
                     "map the six measured streams to canonical [T,S,Tx,Rx] = [T,56,2,3]",
                     "attach config CSV values",
                 ],
                 sample_rate_hz=source_rate_hz,
                 metadata_extra={
                     "label_sets": {"Breathing pattern": [pattern_label]} if pattern_label else {},
                     "source_axis_order": ["time", "stored_link_slot", "stored_subcarrier_slot"],
                     "source_shape": [int(real.shape[0]), stored_links, stored_subcarriers],
                     "antenna_layout": {"tx_links": 2, "rx_links": 3},
                     "antenna_mapping": "source_rx_major_then_tx_slot_to_canonical_tx_rx",
                     "discarded_all_zero_subcarrier_slots": int(stored_subcarriers - measured_subcarriers),
                     "discarded_all_zero_link_slots": int(stored_links - active_link_indices.size),
                 })


def convert_csida_zarr(input_path: Path, output_path: Path) -> Path:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CSIDA conversion requires the optional zarr dependency") from exc
    root = input_path.parent if input_path.name == "csi_data_amp" else input_path
    amp_path = root / "csi_data_amp"
    phase_path = root / "csi_data_pha"
    if not amp_path.exists():
        raise ValueError(f"CSIDA csi_data_amp Zarr array not found under {root}")
    source_amplitude = np.asarray(zarr.open(str(amp_path), mode="r"), dtype=np.float32)
    if source_amplitude.ndim != 4 or source_amplitude.shape[2:] != (3, 114):
        raise ValueError(
            "CSIDA amplitude must use the official [sample,time,3 Rx,114 subcarrier] "
            f"layout, got {source_amplitude.shape}"
        )

    # Official CSIDA stores [sample, time, Rx, subcarrier]. WiSenseHub keeps
    # antenna directions explicit and places subcarriers before the links:
    # [sample, time, subcarrier, Tx=1, Rx=3].
    amplitude = np.transpose(source_amplitude, (0, 1, 3, 2))[:, :, :, None, :]
    arrays: Dict[str, np.ndarray] = {
        "amplitude": amplitude,
        "subcarrier_index": np.arange(amplitude.shape[2], dtype=np.int16),
        "tx_link_index": np.asarray([0], dtype=np.int16),
        "rx_link_index": np.arange(amplitude.shape[4], dtype=np.int16),
    }
    if phase_path.exists():
        source_phase = np.asarray(zarr.open(str(phase_path), mode="r"), dtype=np.float32)
        if source_phase.shape != source_amplitude.shape:
            raise ValueError("CSIDA amplitude and phase shapes differ")
        phase = np.transpose(source_phase, (0, 1, 3, 2))[:, :, :, None, :]
        arrays["phase_rad"] = phase
        arrays["csi_real"] = (amplitude * np.cos(phase)).astype(np.float32)
        arrays["csi_imag"] = (amplitude * np.sin(phase)).astype(np.float32)
    for source_name, target_name in {
        "csi_label_act": "activity_label", "csi_label_env": "environment",
        "csi_label_loc": "location", "csi_label_user": "subject",
    }.items():
        path = root / source_name
        if path.exists():
            values = np.asarray(zarr.open(str(path), mode="r")).reshape(-1)
            if values.size != source_amplitude.shape[0]:
                raise ValueError(f"CSIDA {source_name} count does not match samples")
            arrays[target_name] = values
    source_representation = "amplitude_phase" if phase_path.exists() else "amplitude"
    return _save(input_path, output_path, "csida", arrays, "amplitude",
                 ["sample", "time", "subcarrier", "tx_link", "rx_link"], source_representation,
                 [
                     "open official Zarr arrays",
                     "align amplitude/phase and gesture, room, position, and user labels",
                     "transpose [time,Rx,subcarrier] to canonical [time,subcarrier,Tx,Rx]",
                     "preserve all 114 subcarriers and the native 1 Tx by 3 Rx antenna layout",
                     "reconstruct real and imaginary CSI when phase is available",
                 ],
                 sample_rate_hz=1000.0,
                 metadata_extra={
                     "labels": {
                         "activity": ["pull left", "pull right", "lift up", "press down", "circle", "zigzag"],
                         "environment": ["room 0", "room 1"],
                         "location": ["position 0", "position 1", "position 2"],
                         "subject": ["user 0", "user 1", "user 2", "user 3", "user 4"],
                     },
                     "source_axis_order": ["sample", "time", "rx_link", "subcarrier"],
                     "antenna_layout": {"tx_links": 1, "rx_links": 3},
                 })


def convert_exposing_csi_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("csi", "csi_buff"))
    if csi is None:
        raise ValueError("Exposing the CSI MAT file must contain csi or csi_buff")
    csi = np.asarray(csi)
    if csi.ndim not in {2, 3} or csi.shape[1] < 2048:
        raise ValueError(f"expected AX-CSI [packet,2048+,stream?], got {csi.shape}")
    csi = np.fft.fftshift(csi[:, :2048], axes=1)
    csi = csi[np.sum(np.abs(csi), axis=tuple(range(1, csi.ndim))) != 0]
    remove = np.asarray([
        *range(0, 12), 509, 510, 511, 512, 513, 514, 1013, 1014, 1015, 1016, 1017,
        1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
        1030, 1031, 1032, 1033, 1034, 1035, 1535, 1536, 1537, 1538, 1539,
        2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047,
    ], dtype=np.int32)
    if csi.ndim == 3:
        if csi.shape[2] != 4:
            raise ValueError(f"official Exposing CSI stream axis must have length 4, got {csi.shape}")
        canonical = np.delete(csi, remove, axis=1).transpose(0, 2, 1)
        scale = np.mean(np.abs(canonical), axis=2, keepdims=True)
        canonical = canonical / np.maximum(scale, np.finfo(np.float32).eps)
        source_layout = "load official csi [packet,2048,4 streams]"
    else:
        streams = []
        for stream in range(4):
            value = csi[stream::4]
            value = np.delete(value, remove, axis=1)
            scale = np.mean(np.abs(value), axis=1, keepdims=True)
            streams.append(value / np.maximum(scale, np.finfo(np.float32).eps))
        length = min(value.shape[0] for value in streams)
        canonical = np.stack([value[:length] for value in streams], axis=1)
        source_layout = "deinterleave legacy csi_buff into four monitor streams"
    arrays = _complex_arrays(canonical)
    activity = re.search(r"(?:^|_)([A-L])(?:_|$)", input_path.stem.upper())
    if activity:
        activity_code = activity.group(1)
        activity_names = {
            "A": "walk", "B": "run", "C": "jump", "D": "sitting",
            "E": "empty_room", "F": "standing", "G": "wave_hands",
            "H": "clapping", "I": "lay_down", "J": "wiping",
            "K": "squat", "L": "stretching",
        }
        arrays["source_label"] = np.asarray(activity_code)
        arrays["activity_label"] = np.asarray(activity_names[activity_code])
    arrays["subcarrier_index"] = np.delete(np.arange(2048, dtype=np.int32), remove)
    return _save(input_path, output_path, "exposing-csi", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 [source_layout, "FFT-shift 160 MHz AX-CSI", "remove official null carriers", "normalize each packet by mean amplitude"],
                 sample_rate_hz=1.0 / 0.006)


def convert_wifi_80mhz_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("csi_buff", "CFR", "cfr", "CSI", "csi"))
    csi = csi if csi is not None else _largest_numeric(mapping, 2)
    csi = np.asarray(csi)
    if csi.ndim != 2:
        raise ValueError(f"80 MHz CFR trace must be a 2-D complex matrix, got {csi.shape}")
    source_shape = list(csi.shape)
    transformations = ["load the official complex CFR matrix"]
    shifted = False
    if csi.shape[1] == 1024:
        csi = np.fft.fftshift(csi, axes=1)[:, :1024:4]
        shifted = True
        transformations.append("FFT-shift the 1024-bin capture and select every fourth bin")
    if csi.shape[1] == 256:
        if not shifted:
            csi = np.fft.fftshift(csi, axes=1)
            transformations.append("FFT-shift the 256-bin capture")
        remove = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255])
        csi = np.delete(csi, remove, axis=1)
        transformations.append("remove the 14 non-data bins to retain 242 subcarriers")
    elif csi.shape[1] != 242:
        raise ValueError(f"80 MHz trace expects 242, 256, or 1024 frequency bins, got {csi.shape[1]}")
    else:
        transformations.append("preserve all 242 Nexmon data subcarriers")
    if csi.shape[0] < 4:
        raise ValueError("80 MHz trace contains fewer than four monitor-antenna packets")
    length = csi.shape[0] // 4
    trailing_rows = csi.shape[0] - length * 4
    if trailing_rows:
        transformations.append(f"discard {trailing_rows} incomplete trailing monitor row(s)")
    # The release interleaves four monitor antennas for every transmitted
    # packet. Reshape first, then remove fully empty packet groups; deleting
    # individual zero rows would shift every later antenna assignment.
    canonical = csi[:length * 4].reshape(length, 4, 242)
    valid_packets = np.any(np.abs(canonical) > 0, axis=(1, 2))
    removed_packets = int(np.count_nonzero(~valid_packets))
    canonical = canonical[valid_packets]
    transformations.append("group each four consecutive rows as the monitor-antenna axis")
    if removed_packets:
        transformations.append(f"discard {removed_packets} all-zero packet group(s)")
    if canonical.shape[0] == 0:
        raise ValueError("80 MHz trace contains no non-zero packet groups")
    arrays = _complex_arrays(canonical)
    arrays["subcarrier_index"] = np.arange(242, dtype=np.int32)
    arrays["monitor_antenna_index"] = np.arange(4, dtype=np.int16)
    activity = re.search(r"(?:^|_)([WRJLSCGE])(?:_|\d|$)", input_path.stem.upper())
    if activity:
        activity_code = activity.group(1)
        activity_names = {
            "W": "walking", "R": "running", "J": "jumping", "L": "sitting still",
            "S": "standing", "C": "sit down / stand up", "G": "arm exercises",
            "E": "empty room",
        }
        arrays["source_label"] = np.asarray(activity_code)
        arrays["activity_label"] = np.asarray(activity_names[activity_code])
    label_sets: Dict[str, list[str]] = {}
    if activity:
        label_sets["Activity"] = [activity_names[activity_code]]
    identity = re.search(r"PI\d+[a-z]?_p(\d+)", input_path.stem, re.IGNORECASE)
    if identity:
        person = int(identity.group(1))
        identity_label = "empty room" if person == 0 else f"p{person:02d}"
        arrays["subject"] = np.asarray(identity_label)
        label_sets["Identity"] = [identity_label]
    occupancy = re.search(r"PC\d+[a-z]?_n(\d+)", input_path.stem, re.IGNORECASE)
    if occupancy:
        people = int(occupancy.group(1))
        occupancy_label = "empty room" if people == 0 else f"{people} people"
        arrays["occupancy_label"] = np.asarray(occupancy_label)
        label_sets["Occupancy"] = [occupancy_label]
    subset = re.search(r"\b(AR|PI|PC)(\d+)([a-z])", input_path.stem, re.IGNORECASE)
    subset_code = subset.group(0).upper() if subset else None
    return _save(input_path, output_path, "wifi-80mhz", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 transformations, sample_rate_hz=173.0, metadata_extra={
                     "label_sets": label_sets,
                     "source_shape": source_shape,
                     "source_axis_order": ["interleaved_packet_monitor_antenna", "subcarrier"],
                     "official_source_shape": source_shape,
                     "official_source_axis_order": ["interleaved_packet_monitor_antenna", "subcarrier"],
                     "source_matrix_contract": "[(T × 4 monitor antennas), 242 subcarriers]",
                     "official_nominal_sample_rate_hz": 173.0,
                     "antenna_layout": {"tx_links": 1, "rx_links": 4},
                     "antenna_mapping_evidence": (
                         "The release records one transmitted spatial stream and four interleaved "
                         "monitor antennas; WiSenseHub represents the monitor streams as Rx links."
                     ),
                     **({"official_subset": subset_code} if subset_code else {}),
                 })


def _numeric_csv_matrix(input_path: Path) -> np.ndarray:
    """Read an amplitude CSV while tolerating a single header/index column."""
    value = np.genfromtxt(input_path, delimiter=",", dtype=np.float64, invalid_raise=False)
    value = np.atleast_2d(value)
    value = value[~np.all(np.isnan(value), axis=1)]
    value = value[:, ~np.all(np.isnan(value), axis=0)]
    if value.size == 0 or np.isnan(value).any():
        raise ValueError(f"{input_path.name} is not a rectangular numeric amplitude CSV")
    return value


def convert_usrp_amplitude_csv(dataset_id: str, input_path: Path, output_path: Path) -> Path:
    """Convert Glasgow USRP releases whose official files contain CSI amplitude."""
    value = _numeric_csv_matrix(input_path)
    transformations = ["load official numeric amplitude CSV"]
    # The Glasgow releases use 51/52 OFDM carriers. Some exports store one
    # carrier per row; orient those matrices to time-first without guessing for
    # arbitrary feature counts.
    if value.shape[0] in {51, 52} and value.shape[1] > value.shape[0]:
        value = value.T
        transformations.append("transpose documented 51/52-carrier rows to time-first")
    if dataset_id == "wipe-fall":
        if value.shape[1] != 51:
            raise ValueError(
                "WiPE-FaLl CSV must contain the documented 51 OFDM subcarriers, "
                f"got {value.shape}"
            )
        release_folder = next(
            (part.lower() for part in input_path.parts
             if part.lower() in {"low", "low_unseen", "med", "med_unseen", "high", "high_unseen"}),
            None,
        )
        if not release_folder:
            raise ValueError(
                "WiPE-FaLl label is encoded by its official folder; keep the CSV inside "
                "low, low_unseen, med, med_unseen, high, or high_unseen"
            )
        risk = {"low": "low", "med": "medium", "high": "high"}[release_folder.split("_")[0]]
        release_split = "unseen test" if release_folder.endswith("_unseen") else "training"
        arrays = {
            "amplitude": value[:, :, None, None].astype(np.float32),
            "source_label": np.asarray(risk),
            "activity_label": np.asarray(risk),
        }
        return _save(
            input_path, output_path, dataset_id, arrays, "amplitude",
            ["time", "subcarrier", "tx_link", "rx_link"], "processed_amplitude",
            transformations + ["insert singleton Tx and Rx axes", "cast float32"],
            metadata_extra={
                "label_sets": {"Fall risk": [risk], "Release split": [release_split]},
                "release_folder": release_folder,
                "source_axis_order": ["time", "subcarrier"],
                "source_shape": list(value.shape),
                "antenna_layout": {"tx_links": 1, "rx_links": 1},
                "sample_rate_evidence": "The public README does not report the CSI sample rate.",
            },
        )
    amplitude = value[:, None, :].astype(np.float32)
    label = input_path.parent.name
    if dataset_id == "wireless-har-wifi-uwb" and re.fullmatch(r"Room_\d+", label, re.IGNORECASE):
        label = re.sub(r"_\d+$", "", input_path.stem)
    metadata: Dict[str, object] = {}
    if dataset_id == "wireless-har-wifi-uwb":
        room = next((part for part in input_path.parts if re.fullmatch(r"Room_\d+", part, re.IGNORECASE)), None)
        metadata["label_sets"] = {
            "Activity": [label.lower()],
            **({"Room": [room.replace("_", " ").lower()]} if room else {}),
        }
    elif dataset_id == "glasgow-multiuser":
        release_class = input_path.parent.name
        occupancy_match = re.match(r"(\d+)_Subjects?_(.+)$", release_class)
        occupancy = int(occupancy_match.group(1)) if occupancy_match else 0
        raw_scenario = occupancy_match.group(2) if occupancy_match else release_class
        activities = []
        for token in re.findall(r"Sitting|Standing|Walking|Walk|Empty(?: Room)?", raw_scenario, re.IGNORECASE):
            normalized = {
                "sitting": "sitting", "standing": "standing",
                "walking": "walking", "walk": "walking",
                "empty": "empty room", "empty room": "empty room",
            }[token.lower()]
            if normalized not in activities:
                activities.append(normalized)
        scenario = re.sub(r"_+", " ", raw_scenario).strip().lower()
        scenario = re.sub(r"\bwalk\b", "walking", scenario)
        scenario = f"{occupancy} people: {scenario}" if occupancy else "0 people: empty room"
        # Keep the legacy scalar activity field for downstream code that
        # expects one simple label, while ``label_sets`` retains the full
        # released joint class without collapsing it.
        label = activities[0] if len(activities) == 1 else scenario
        metadata["release_class"] = release_class
        metadata["label_sets"] = {
            "Scenario": [scenario],
            "Occupancy": [f"{occupancy} people"],
            "Activity": activities or ["empty room"],
        }
    elif dataset_id == "glasgow-activity-localization":
        stem = input_path.stem
        activity_match = re.match(r"(EmptyRoom|Leaning|NoActivity|Sitting|Standing|WalkingRxTx|WalkingTxRx)", stem, re.IGNORECASE)
        raw_activity = activity_match.group(1) if activity_match else stem
        activity = {
            "emptyroom": "empty room", "leaning": "leaning",
            "noactivity": "no activity", "sitting": "sitting", "standing": "standing",
            "walkingrxtx": "walking Rx to Tx", "walkingtxrx": "walking Tx to Rx",
        }.get(raw_activity.lower(), raw_activity.replace("_", " "))
        location_match = re.search(r"L(\d+)Z(\d+)", stem, re.IGNORECASE)
        if not location_match:
            parent_match = re.search(r"Location(\d+)", input_path.parent.name, re.IGNORECASE)
            zone_match = re.search(r"Z(\d+)", stem, re.IGNORECASE)
            location_match = (
                re.match(r"(\d+) (\d+)", f"{parent_match.group(1)} {zone_match.group(1)}")
                if parent_match and zone_match else None
            )
        label = activity
        label_sets = {"Activity": [activity]}
        if location_match:
            location, zone = location_match.groups()
            label_sets["Location"] = [f"location {location}, zone {zone}"]
        metadata["label_sets"] = label_sets
    arrays: Dict[str, np.ndarray] = {
        "amplitude": amplitude,
        "source_label": np.asarray(label.lower()),
        "activity_label": np.asarray(label.lower()),
    }
    return _save(input_path, output_path, dataset_id, arrays, "amplitude",
                 ["time", "link", "subcarrier"], "processed_amplitude",
                 transformations + ["insert singleton USRP link axis", "cast float32"],
                 metadata_extra=metadata)


def convert_wireless_har_wifi(input_path: Path, output_path: Path) -> Path:
    """Convert only the WiFi_CSI branch of the paired WiFi/UWB release."""
    if "wifi_csi" not in {part.lower() for part in input_path.parts}:
        raise ValueError("wireless HAR adapter only accepts files inside the official WiFi_CSI directory")
    if input_path.suffix.lower() == ".csv":
        return convert_usrp_amplitude_csv("wireless-har-wifi-uwb", input_path, output_path)
    data = None
    source_shape = None
    source_channel_names: list[str] = []
    if input_path.suffix.lower() == ".mat":
        try:
            source, _ = load_generic_source(input_path)
            data = _find(source, ("csi", "CSI", "amp", "amplitude", "data", "x", "input"))
        except (NotImplementedError, ValueError):
            # Room 2 is a MATLAB 7.3 table. Its channel names explicitly run
            # from tx1rx1_sub1 through tx1rx3_sub30. Keep that released
            # factorization rather than treating the 90 columns as carriers.
            import h5py

            with h5py.File(input_path, "r") as handle:
                value_cells = []
                name_cells = []
                def collect(_name, obj):
                    if (
                        isinstance(obj, h5py.Dataset)
                        and obj.attrs.get("MATLAB_class") == np.bytes_(b"cell")
                        and obj.size == 90
                    ):
                        targets = [handle[reference] for reference in np.asarray(obj).reshape(-1)]
                        if all(
                            isinstance(target, h5py.Dataset)
                            and target.dtype.names == ("real", "imag")
                            for target in targets
                        ):
                            value_cells.append(obj)
                        elif all(
                            isinstance(target, h5py.Dataset)
                            and target.attrs.get("MATLAB_class") == np.bytes_(b"char")
                            for target in targets
                        ):
                            name_cells.append(obj)
                handle.visititems(collect)
                if not value_cells or not name_cells:
                    raise ValueError("Wireless HAR MATLAB table does not contain its 90 named complex CSI columns")
                names = []
                for reference in np.asarray(name_cells[0]).reshape(-1):
                    chars = np.asarray(handle[reference]).reshape(-1)
                    names.append("".join(chr(int(value)) for value in chars))
                columns = []
                for reference in np.asarray(value_cells[0]).reshape(-1):
                    raw = np.asarray(handle[reference]).reshape(-1)
                    columns.append(raw["real"] + 1j * raw["imag"])
                if len({column.size for column in columns}) != 1:
                    raise ValueError("Wireless HAR CSI columns do not share one time length")
                by_name = dict(zip(names, columns))
                expected_names = [
                    f"tx1rx{rx}_sub{subcarrier}"
                    for rx in range(1, 4)
                    for subcarrier in range(1, 31)
                ]
                missing = [name for name in expected_names if name not in by_name]
                if missing:
                    raise ValueError(f"Wireless HAR MATLAB table is missing CSI columns: {missing[:5]}")
                source_channel_names = expected_names
                source_flat = np.column_stack([by_name[name] for name in expected_names])
                source_shape = list(source_flat.shape)
                # Released storage is Rx-major, then subcarrier. Canonical CSI
                # is [T,S,Tx,Rx], with one transmit and three receive links.
                data = source_flat.reshape(source_flat.shape[0], 3, 30).transpose(0, 2, 1)[:, :, None, :]
    else:
        source, _ = load_generic_source(input_path)
        data = _find(source, ("csi", "CSI", "amp", "amplitude", "data", "x", "input"))
    if data is None:
        data = _largest_numeric(source, 1)
    if source_shape is None:
        canonical, axes = _canonical_sequence(np.asarray(data))
    else:
        canonical = np.asarray(data)
        axes = ["time", "subcarrier", "tx_link", "rx_link"]
    arrays = _complex_arrays(canonical)
    source_label = input_path.parent.name.lower()
    display_labels = {
        "kneel": "kneel",
        "liedown": "lie down",
        "pickup": "pick up",
        "sit": "sit",
        "sitrotate": "sit and rotate",
        "stand": "stand",
        "standrotate": "stand and rotate",
        "walk": "walk",
    }
    label = display_labels.get(source_label, source_label.replace("_", " "))
    arrays["source_label"] = np.asarray(label)
    arrays["activity_label"] = np.asarray(label)
    room = next((part for part in input_path.parts if re.fullmatch(r"Room_\d+", part, re.IGNORECASE)), None)
    metadata_extra: Dict[str, object] = {
        "label_sets": {"Activity": [label]},
        "source_activity_code": source_label,
        "sample_rate_evidence": (
            "The released Room 2 MAT table contains packet order but no timestamp or documented packet rate."
        ),
    }
    if room:
        metadata_extra["settings"] = {"room": room.replace("_", " ").lower()}
    if source_shape is not None:
        metadata_extra.update({
            "source_axis_order": ["time", "flattened_csi_channel"],
            "source_shape": source_shape,
            "antenna_layout": {"tx_links": 1, "rx_links": 3},
            "antenna_mapping": "released names tx1rx{1..3}_sub{1..30}",
            "source_channel_names": source_channel_names,
        })
    return _save(input_path, output_path, "wireless-har-wifi-uwb", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 [
                     "select official WiFi_CSI branch",
                     "load released 90-column complex CSI table",
                     "reshape named channels from [T,90] to [T,30,1,3]",
                     "preserve all 30 native subcarriers and all 3 receive links",
                 ],
                 metadata_extra=metadata_extra)


def convert_profile(dataset_id: str, input_path: Path, output_path: Path) -> Path:
    if dataset_id == "csi-bench" and input_path.suffix.lower() in {".mat", ".h5", ".hdf5"}:
        return convert_csi_bench_mat(input_path, output_path)
    if dataset_id == "mm-fi" and input_path.is_dir():
        return convert_mmfi_directory(input_path, output_path)
    if dataset_id == "ntu-fi" and input_path.suffix.lower() == ".mat":
        return convert_ntu_fi_mat(input_path, output_path)
    if dataset_id == "widar3" and input_path.suffix.lower() == ".csv":
        return convert_widar_csv(input_path, output_path)
    if dataset_id == "figshare-csi-har" and input_path.name == "data.csv":
        return convert_three_rooms_directory(input_path, output_path)
    if dataset_id == "signfi" and input_path.suffix.lower() == ".mat":
        return convert_signfi_mat(input_path, output_path)
    if dataset_id == "wimans" and input_path.suffix.lower() in {".mat", ".npy"}:
        return convert_wimans(input_path, output_path)
    if dataset_id == "xrf55" and input_path.suffix.lower() == ".npy":
        return convert_xrf55_npy(input_path, output_path)
    if dataset_id == "ehunam" and input_path.suffix.lower() == ".mat":
        return convert_ehunam_mat(input_path, output_path)
    if dataset_id == "wifi-presence-movement" and (input_path.name.endswith(".json.gz") or input_path.suffix.lower() == ".json"):
        return convert_wifi_presence_json(input_path, output_path)
    if dataset_id == "wifi-tad" and input_path.suffix.lower() == ".npy":
        return convert_wifi_tad_npy(input_path, output_path)
    if dataset_id == "operanet" and input_path.suffix.lower() == ".mat":
        return convert_operanet_mat(input_path, output_path)
    if dataset_id == "nist-breathesmart" and input_path.name.endswith("_csi_real_log.csv"):
        return convert_nist_breathesmart(input_path, output_path)
    if dataset_id == "csida" and input_path.is_dir():
        return convert_csida_zarr(input_path, output_path)
    if dataset_id == "exposing-csi" and input_path.suffix.lower() == ".mat":
        return convert_exposing_csi_mat(input_path, output_path)
    if dataset_id == "wifi-80mhz" and input_path.suffix.lower() == ".mat":
        return convert_wifi_80mhz_mat(input_path, output_path)
    if dataset_id in {"glasgow-activity-localization", "glasgow-multiuser", "wipe-fall"} and input_path.suffix.lower() == ".csv":
        return convert_usrp_amplitude_csv(dataset_id, input_path, output_path)
    if dataset_id == "wireless-har-wifi-uwb":
        return convert_wireless_har_wifi(input_path, output_path)
    # Remaining profiles retain official processed values and record that no
    # undocumented calibration or axis permutation was inferred.
    return convert_generic(input_path, output_path, dataset_id)
