from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np


CLASS_NAMES = ["lie_down", "fall", "walk", "pickup", "run", "sit_down", "stand_up"]


def normalize_ut_har_arrays(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize SenseFi UT-HAR features to [sample,time,subcarrier,tx,rx]."""
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[1:] == (250, 90):
        links_first = data.reshape(data.shape[0], 250, 3, 30)
    elif data.ndim == 4 and data.shape[1:] == (250, 3, 30):
        links_first = data
    elif data.ndim == 4 and data.shape[1:] == (3, 30, 250):
        links_first = data.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"unsupported UT-HAR shape {data.shape}")
    canonical = links_first.transpose(0, 1, 3, 2)[:, :, :, None, :]
    labels = np.asarray(labels).reshape(-1).astype(np.int16)
    if labels.size != canonical.shape[0]:
        raise ValueError("UT-HAR data and label counts differ")
    return canonical.astype(np.float32), labels


def convert_ut_har_npz(input_path: Path, output_path: Path) -> Path:
    source = np.load(input_path, allow_pickle=False)
    if isinstance(source, np.lib.npyio.NpzFile):
        data_key = next((key for key in ("data", "x", "amplitude") if key in source.files), None)
        label_key = next((key for key in ("label", "labels", "y") if key in source.files), None)
        if data_key is None or label_key is None:
            raise ValueError("UT-HAR NPZ requires data/x/amplitude and label/labels/y arrays")
        data, label = source[data_key], source[label_key]
    else:
        # SenseFi distributes NumPy payloads with a .csv extension under
        # UT_HAR/data and matching files under UT_HAR/label.
        parts = list(input_path.parts)
        try:
            data_index = len(parts) - 1 - parts[::-1].index("data")
        except ValueError as exc:
            raise ValueError("UT-HAR array file must be in a data/ directory") from exc
        label_root = Path(*parts[:data_index], "label", *parts[data_index + 1:-1])
        names = [input_path.name]
        if input_path.name.startswith(("X_", "x_")):
            names.insert(0, "y_" + input_path.name[2:])
        label_path = next((label_root / name for name in names if (label_root / name).exists()), label_root / names[0])
        if not label_path.exists():
            raise ValueError(f"matching UT-HAR label file not found; tried: {', '.join(str(label_root / name) for name in names)}")
        data, label = source, np.load(label_path, allow_pickle=False)
    source_shape = list(np.asarray(data).shape)
    amplitude, labels = normalize_ut_har_arrays(data, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        packet_index=np.arange(amplitude.shape[1], dtype=np.int32),
        amplitude=amplitude,
        valid_mask=np.ones(amplitude.shape[:2], dtype=bool),
        activity_label=labels,
    )
    sidecar = {
        "schema_version": "1.0", "dataset_id": "ut-har", "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_representation": "processed_amplitude", "standard_representation": "amplitude",
        "shape": list(amplitude.shape), "axis_order": ["sample", "time", "subcarrier", "tx_link", "rx_link"],
        "source_shape": source_shape,
        "source_axis_order": ["sample", "time", "flattened_rx_link_subcarrier"],
        "sample_rate_hz": None, "time_axis": "packet_index", "power_unit": "source_amplitude_arbitrary_unit",
        "labels": {"activity": CLASS_NAMES},
        "label_sets": {"Activity": CLASS_NAMES},
        "antenna_layout": {"tx_links": 1, "rx_links": 3},
        "antenna_mapping_evidence": "SenseFi UT-HAR documents 90 channels as 3 antenna streams × 30 subcarriers",
        "source_rate_status": "not reported in the processed SenseFi files",
        "transformations": [
            "reshape 90 flattened channels to 30 subcarriers × 1 Tx × 3 Rx",
            "preserve signed processed feature values",
            "cast float32",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(), "tool": "wisensehub-0.6.0"
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
