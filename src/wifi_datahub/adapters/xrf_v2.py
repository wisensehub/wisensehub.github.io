from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


XRF_V2_SAMPLE_RATE_HZ = 50.0
XRF_V2_SCENES = {1: "dining room", 2: "study room", 3: "bedroom"}
XRF_V2_ACTION_NAMES = {
    0: "stretching", 1: "pouring water", 2: "writing", 3: "cutting fruit",
    4: "eating fruit", 5: "taking medicine", 6: "drinking water", 7: "sitting down",
    8: "turning on/off eye protection lamp", 9: "opening/closing curtains",
    10: "opening/closing windows", 11: "typing", 12: "opening envelope",
    13: "throwing garbage", 14: "picking fruit", 15: "picking up items",
    16: "answering phone", 17: "using mouse", 18: "wiping table",
    19: "writing on blackboard", 20: "washing hands", 21: "using phone", 22: "reading",
    23: "watering plants", 24: "walking to bed", 25: "walking to chair",
    26: "walking to cabinet", 27: "walking to window", 28: "walking to blackboard",
    29: "getting out of bed", 30: "standing up", 31: "lying down",
    32: "standing still", 33: "lying still",
}


def normalize_xrf_v2_wifi(amplitude: np.ndarray, receivers: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, list[int]]:
    """Normalize XRF V2 WiFi amplitude to [time, subcarrier, tx, rx].

    The official loader documents each HDF5 sample as `[time, 3, 3, 30]`:
    three receiver devices, three source channel streams, and thirty
    subcarriers. Keep both released 3-valued axes explicit instead of
    flattening them; the release does not establish that the channel stream
    axis is a physical transmitter-antenna axis.
    """
    amplitude = np.asarray(amplitude)
    if amplitude.ndim != 4 or amplitude.shape[1:] != (3, 3, 30):
        raise ValueError(f"expected XRF V2 WiFi shape [time,3,3,30], got {amplitude.shape}")
    keep = list(receivers) if receivers is not None else [0, 1, 2]
    if not keep or any(index not in (0, 1, 2) for index in keep):
        raise ValueError("receivers must contain indices from 0, 1, 2")
    selected = amplitude[:, keep, :, :]
    canonical = selected.transpose(0, 3, 2, 1)
    return canonical.astype(np.float32), keep


def convert_xrf_v2_h5(input_path: Path, output_path: Path, receivers: Optional[Sequence[int]] = None) -> Path:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("XRF V2 conversion requires the optional 'h5py' dependency") from exc
    with h5py.File(input_path, "r") as source:
        source_amplitude = source["amp"][...]
        source_phase = source["pha"][...]
        amplitude, kept = normalize_xrf_v2_wifi(source_amplitude, receivers)
        phase, _ = normalize_xrf_v2_wifi(source_phase, receivers)
        label = np.asarray(source["label"][...], dtype=np.int64)
    if label.ndim != 2 or label.shape[1] != 4:
        raise ValueError(f"XRF V2 label table must be [segment,4], got {label.shape}")
    packet_labels = np.full(amplitude.shape[0], "unlabeled", dtype="U64")
    segments = []
    for _segment_id, raw_label, start, end in label:
        label_id = int(raw_label)
        if label_id not in XRF_V2_ACTION_NAMES:
            raise ValueError(f"XRF V2 activity id outside 0..33: {label_id}")
        start_index = max(0, int(start))
        end_index = min(amplitude.shape[0], int(end))
        if end_index <= start_index:
            raise ValueError(f"invalid XRF V2 segment {start_index}:{end_index}")
        name = XRF_V2_ACTION_NAMES[label_id]
        packet_labels[start_index:end_index] = name
        segments.append({
            "start_seconds": start_index / XRF_V2_SAMPLE_RATE_HZ,
            "end_seconds": end_index / XRF_V2_SAMPLE_RATE_HZ,
            "start_index": start_index,
            "end_index": end_index,
            "label_id": label_id,
            "label": name,
        })
    setting_match = re.fullmatch(r"(?P<subject>\d+)_(?P<scene>\d+)_(?P<group>\d+)", input_path.stem)
    settings = {}
    if setting_match:
        scene_id = int(setting_match.group("scene"))
        settings = {
            "subject_id": int(setting_match.group("subject")),
            "scene_id": scene_id,
            "scene": XRF_V2_SCENES.get(scene_id, f"scene {scene_id}"),
            "sequence_group": int(setting_match.group("group")),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        packet_index=np.arange(amplitude.shape[0], dtype=np.int32),
        amplitude=amplitude,
        phase_rad=phase,
        valid_mask=np.ones(amplitude.shape[0], dtype=bool),
        source_label=packet_labels,
        segment_table=label,
        subcarrier_index=np.arange(amplitude.shape[1], dtype=np.int32),
        tx_link_index=np.arange(amplitude.shape[2], dtype=np.int32),
        rx_link_index=np.asarray(kept, dtype=np.int32),
    )
    sidecar = {
        "schema_version": "1.0",
        "dataset_id": "xrf-v2",
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_representation": "processed_amplitude",
        "standard_representation": "amplitude",
        "shape": list(amplitude.shape),
        "axis_order": ["packet", "subcarrier", "tx_link", "rx_link"],
        "source_axis_order": ["packet", "rx_link", "tx_link", "subcarrier"],
        "source_shape": [int(value) for value in source_amplitude.shape],
        "sample_rate_hz": XRF_V2_SAMPLE_RATE_HZ,
        "duration_s": amplitude.shape[0] / XRF_V2_SAMPLE_RATE_HZ,
        "time_axis": "packet_index",
        "power_unit": "source_amplitude_arbitrary_unit",
        "receivers_kept": kept,
        "labels": {"activity": {str(key): value for key, value in XRF_V2_ACTION_NAMES.items()}},
        "segments": segments,
        "settings": settings,
        "raw_activity_count": 34,
        "derived_benchmark_activity_count": 30,
        "benchmark_label_note": (
            "The raw HDF5 interval tables retain 34 activity IDs. The derived benchmark "
            "annotation file merges walking to bed, chair, cabinet, window, and blackboard "
            "into one Walking class, producing 30 benchmark classes."
        ),
        "antenna_layout": {"tx_links": 3, "rx_links": len(kept)},
        "antenna_mapping": "source channel stream-as-logical-Tx and receiver-device-as-Rx axes",
        "antenna_mapping_evidence": (
            "The official loader documents [time, receiver device, channel, subcarrier]. "
            "The release does not prove that channel is a physical transmitter antenna."
        ),
        "transformations": [
            "load authentic amplitude and phase tensors", "select receivers",
            "transpose to [time,subcarrier,tx_link,rx_link]", "attach official 50 Hz label intervals",
            "cast float32",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "wisensehub-0.6.0"
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
