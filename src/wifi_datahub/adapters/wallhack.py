from __future__ import annotations

import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

WALLHACK_CLASS_NAMES = {
    "0": "no_presence",
    "1": "walking",
    "2": "arm_waving",
}


def parse_interleaved_imag_real(value: str | Iterable[float]) -> np.ndarray:
    """Parse Wallhack's documented [I, R, I, R, ...] packet representation."""
    if isinstance(value, str):
        value = ast.literal_eval(value.strip())
    vector = np.asarray(list(value), dtype=np.float32)
    if vector.ndim != 1 or vector.size % 2:
        raise ValueError("interleaved I/R vector must have an even number of values")
    imag = vector[0::2]
    real = vector[1::2]
    return real.astype(np.complex64) + 1j * imag.astype(np.complex64)


def select_subcarriers(csi: np.ndarray, include_ht_ltf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Select documented 52 L-LTF subcarriers and optional 56 HT-LTF carriers."""
    indices = list(range(6, 32)) + list(range(33, 59))
    if include_ht_ltf:
        indices += list(range(66, 94)) + list(range(95, 123))
    valid = np.asarray([index for index in indices if index < csi.shape[-1]], dtype=np.int64)
    return csi[..., valid], valid


def convert_wallhack_csv(input_path: Path, output_path: Path, include_ht_ltf: bool = False) -> Path:
    packets = []
    labels = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "data" not in reader.fieldnames:
            raise ValueError("Wallhack CSV must contain a data column")
        for row in reader:
            packets.append(parse_interleaved_imag_real(row["data"]))
            if "class" in row and row["class"] != "":
                labels.append(row["class"])
    if not packets:
        raise ValueError("Wallhack CSV contains no CSI packets")
    lengths = {packet.size for packet in packets}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent CSI vector lengths: {sorted(lengths)}")
    raw = np.stack(packets)
    selected, indices = select_subcarriers(raw, include_ht_ltf)
    complex_csi = selected[:, None, :]
    amplitude = np.abs(complex_csi).astype(np.float32)
    power = amplitude.astype(np.float64) ** 2
    reference = max(float(np.median(power)), np.finfo(np.float32).tiny)
    power_db_rel = (10 * np.log10(np.maximum(power, np.finfo(np.float32).tiny) / reference)).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        timestamp_s=np.arange(complex_csi.shape[0], dtype=np.float64) / 100.0,
        csi_real=complex_csi.real.astype(np.float32),
        csi_imag=complex_csi.imag.astype(np.float32),
        amplitude=amplitude,
        power_db_rel=power_db_rel,
        valid_mask=np.ones(complex_csi.shape[0], dtype=bool),
        source_label=np.asarray(
            [WALLHACK_CLASS_NAMES.get(label, label) for label in labels] if labels else [],
            dtype="U64",
        ),
        subcarrier_index=indices,
    )
    class_name = WALLHACK_CLASS_NAMES.get(labels[0], labels[0]) if labels else None
    sidecar = {
        "schema_version": "1.0", "dataset_id": "wallhack18k",
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_representation": "raw_iq", "standard_representation": "complex_csi",
        "shape": list(complex_csi.shape), "axis_order": ["time", "link", "subcarrier"],
        "sample_rate_hz": 100.0, "duration_s": complex_csi.shape[0] / 100.0,
        "time_unit": "s", "time_axis": "timestamp_s",
        "power_unit": "dB_relative_to_median_valid_sample_power", "valid_fraction": 1.0,
        "labels": {
            "class": labels[0] if labels else None,
            "activity": class_name,
            "vocabulary": WALLHACK_CLASS_NAMES,
        },
        "transformations": ["parse interleaved imaginary/real", "select documented L-LTF/HT-LTF subcarriers", "relative-power conversion"],
        "created_at": datetime.now(timezone.utc).isoformat(), "tool": "wisensehub-0.6.0"
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
