from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


SIGNAL_ARRAYS = {
    "amplitude", "csi_real", "csi_imag", "phase_rad", "power_db_rel",
    "official_normalized_amplitude",
}

# These layouts come from the released tensor/file organization, not from the
# number of flattened links alone. When a release does not expose a reliable
# factorization, every stream is retained as one Tx by L Rx.
DATASET_ANTENNA_LAYOUTS: Dict[str, Tuple[int, int]] = {
    "ehunam": (1, 1),
    "exposing-csi": (1, 4),
    "figshare-csi-har": (2, 2),
    "glasgow-activity-localization": (1, 1),
    "glasgow-multiuser": (1, 1),
    "mm-fi": (1, 3),
    "nist-breathesmart": (2, 3),
    "ntu-fi": (1, 3),
    "operanet": (3, 3),
    "signfi": (1, 3),
    "ut-har": (1, 3),
    "wallhack18k": (1, 1),
    "wifi-80mhz": (1, 4),
    "wifi-presence-movement": (1, 3),
    "wimans": (3, 3),
    "wipe-fall": (1, 1),
    "wireless-har-wifi-uwb": (1, 3),
    "xrf-v2": (3, 3),
    "xrf55": (1, 9),
}


def _reshape_signal(
    value: np.ndarray, axes: list[str], tx_links: int, rx_links: int,
) -> np.ndarray:
    sample_axis = axes.index("sample") if "sample" in axes else None
    time_axis = next((axes.index(name) for name in ("time", "packet") if name in axes), None)
    if time_axis is None:
        raise ValueError(f"cannot identify time axis in {axes}")
    link_axis = axes.index("link")
    subcarrier_axis = axes.index("subcarrier")
    order = ([sample_axis] if sample_axis is not None else []) + [time_axis, subcarrier_axis, link_axis]
    moved = np.transpose(value, order)
    if moved.shape[-1] != tx_links * rx_links:
        raise ValueError(
            f"link count {moved.shape[-1]} does not match {tx_links} Tx × {rx_links} Rx"
        )
    return moved.reshape(moved.shape[:-1] + (tx_links, rx_links))


def canonicalize_csi_output(dataset_id: str, output_path: Path) -> Path:
    """Rewrite flattened CSI links to per-sample [T,S,Tx,Rx] in place.

    Processed feature representations (for example Widar3 BVP and WiFiTAD
    features) are intentionally left on their documented semantic axes.
    """
    sidecar_path = output_path.with_suffix(".json")
    if not output_path.exists() or not sidecar_path.exists():
        return sidecar_path
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    axes = list(metadata.get("axis_order") or [])
    if {"subcarrier", "tx_link", "rx_link"}.issubset(axes):
        return sidecar_path
    if "link" not in axes or "subcarrier" not in axes:
        metadata.setdefault("canonical_tensor_exception", {
            "reason": "This output is a processed feature representation, not raw CSI.",
            "axis_order": axes,
        })
        sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return sidecar_path

    with np.load(output_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    primary = metadata.get("standard_representation")
    if primary not in arrays:
        primary = "amplitude" if "amplitude" in arrays else next(iter(arrays))
    primary_value = np.asarray(arrays[primary])
    if primary_value.ndim != len(axes):
        raise ValueError(
            f"primary array {primary} shape {primary_value.shape} does not match axes {axes}"
        )

    link_count = int(primary_value.shape[axes.index("link")])
    requested = metadata.get("antenna_layout") or {}
    tx_links = int(requested.get("tx_links") or 0) if isinstance(requested, dict) else 0
    rx_links = int(requested.get("rx_links") or 0) if isinstance(requested, dict) else 0
    if not tx_links or not rx_links:
        tx_links, rx_links = DATASET_ANTENNA_LAYOUTS.get(dataset_id, (1, link_count))
    mapping_source = "released antenna layout"
    if tx_links * rx_links != link_count:
        tx_links, rx_links = 1, link_count
        mapping_source = "preserved flattened streams; source factorization unavailable"

    new_axes = (["sample"] if "sample" in axes else []) + [
        "time", "subcarrier", "tx_link", "rx_link",
    ]
    for name, value in list(arrays.items()):
        if name in SIGNAL_ARRAYS and np.asarray(value).shape == primary_value.shape:
            arrays[name] = _reshape_signal(np.asarray(value), axes, tx_links, rx_links)
    arrays["subcarrier_index"] = np.arange(
        primary_value.shape[axes.index("subcarrier")], dtype=np.int32,
    )
    arrays["tx_link_index"] = np.arange(tx_links, dtype=np.int32)
    arrays["rx_link_index"] = np.arange(rx_links, dtype=np.int32)
    np.savez_compressed(output_path, **arrays)

    source_axes = axes
    source_shape = list(primary_value.shape)
    primary_shape = list(np.asarray(arrays[primary]).shape)
    transformations = list(metadata.get("transformations") or [])
    transformations.append(
        f"reshape flattened link axis to canonical [T,S,Tx,Rx] with "
        f"Tx={tx_links}, Rx={rx_links}; preserve all native subcarriers and links"
    )
    metadata.update({
        "source_axis_order": source_axes,
        "source_shape": source_shape,
        "axis_order": new_axes,
        "shape": primary_shape,
        "antenna_layout": {"tx_links": tx_links, "rx_links": rx_links},
        "antenna_mapping": "tx_major_then_rx",
        "antenna_mapping_evidence": mapping_source,
        "transformations": transformations,
    })
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
