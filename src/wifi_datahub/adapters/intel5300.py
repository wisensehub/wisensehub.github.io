from __future__ import annotations

import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np


WIAR_ACTIVITY_NAMES = {
    1: "horizontal arm wave",
    2: "high arm wave",
    3: "two hands wave",
    4: "high throw",
    5: "draw x",
    6: "draw tick",
    7: "toss paper",
    8: "forward kick",
    9: "side kick",
    10: "bend",
    11: "hand clap",
    12: "walk",
    13: "phone call",
    14: "drink water",
    15: "sit down",
    16: "squat",
}

_WIAR_FILENAME = re.compile(r"^csi_a(?P<activity>\d+)_(?P<trial>\d+)\.dat$", re.IGNORECASE)


def _signed_byte(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def parse_bfee(payload_record: bytes) -> Dict[str, object]:
    """Port of the Intel 5300 CSI Tool read_bfee.c record parser."""
    if len(payload_record) < 20:
        raise ValueError("truncated Intel 5300 beamforming record")
    timestamp_low = int.from_bytes(payload_record[0:4], "little")
    bfee_count = int.from_bytes(payload_record[4:6], "little")
    nrx, ntx = payload_record[8], payload_record[9]
    if nrx not in (1, 2, 3) or ntx not in (1, 2, 3):
        raise ValueError(f"invalid Intel 5300 antenna dimensions ntx={ntx}, nrx={nrx}")
    antenna_sel = payload_record[15]
    declared_len = int.from_bytes(payload_record[16:18], "little")
    expected_len = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
    if declared_len != expected_len:
        raise ValueError(f"beamforming payload length {declared_len} != expected {expected_len}")
    payload = payload_record[20:]
    if len(payload) < expected_len:
        raise ValueError("truncated Intel 5300 CSI bit payload")
    csi = np.empty((ntx, nrx, 30), dtype=np.complex64)
    bit_index = 0
    for subcarrier in range(30):
        bit_index += 3
        remainder = bit_index % 8
        for pair in range(nrx * ntx):
            byte_index = bit_index // 8
            real = (payload[byte_index] >> remainder) | (payload[byte_index + 1] << (8 - remainder))
            imag = (payload[byte_index + 1] >> remainder) | (payload[byte_index + 2] << (8 - remainder))
            tx, rx = pair % ntx, pair // ntx
            csi[tx, rx, subcarrier] = complex(_signed_byte(real), _signed_byte(imag))
            bit_index += 16
    permutation = [antenna_sel & 0x3, (antenna_sel >> 2) & 0x3, (antenna_sel >> 4) & 0x3]
    if nrx > 1 and sum(value + 1 for value in permutation[:nrx]) == (1, 3, 6)[nrx - 1]:
        reordered = np.empty_like(csi)
        for source_rx, destination_rx in enumerate(permutation[:nrx]):
            reordered[:, destination_rx, :] = csi[:, source_rx, :]
        csi = reordered
    return {
        "timestamp_low": timestamp_low, "bfee_count": bfee_count, "nrx": nrx, "ntx": ntx,
        "rssi": [payload_record[10], payload_record[11], payload_record[12]],
        "noise": _signed_byte(payload_record[13]), "agc": payload_record[14], "csi": csi,
    }


def read_bf_file(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                # Some files in the official WiAR repository end partway
                # through one final record. The MATLAB reader effectively
                # ignores that incomplete tail, so preserve every complete
                # beamforming record and stop here.
                break
            field_len = struct.unpack(">H", length_bytes)[0]
            if field_len < 1:
                raise ValueError("invalid zero-length Intel 5300 record")
            code = handle.read(1)
            body = handle.read(field_len - 1)
            if len(code) != 1 or len(body) != field_len - 1:
                # A partial final record is present in several official WiAR
                # files. It contains no complete sample and is safe to ignore.
                break
            if code[0] == 187:
                records.append(parse_bfee(body))
    if not records:
        raise ValueError("no Intel 5300 beamforming records (code 187) found")
    return records


def convert_wiar_dat(input_path: Path, output_path: Path) -> Path:
    match = _WIAR_FILENAME.fullmatch(input_path.name)
    if match is None:
        raise ValueError(
            f"WiAR filename must match csi_a<activity>_<trial>.dat, got {input_path.name!r}"
        )
    activity_id = int(match.group("activity"))
    trial_id = int(match.group("trial"))
    if activity_id not in WIAR_ACTIVITY_NAMES:
        raise ValueError(f"WiAR activity id must be 1-16, got {activity_id}")
    activity_name = WIAR_ACTIVITY_NAMES[activity_id]

    records = read_bf_file(input_path)
    shapes = {np.asarray(record["csi"]).shape for record in records}
    if len(shapes) != 1:
        raise ValueError(f"WiAR antenna configuration changes within file: {sorted(shapes)}")
    source_csi = np.stack([np.asarray(record["csi"]) for record in records])
    tx_links, rx_links = int(source_csi.shape[1]), int(source_csi.shape[2])
    # read_bfee returns [Tx, Rx, subcarrier]. Store each packet using the hub's
    # canonical per-sample order [time, subcarrier, tx_link, rx_link].
    csi = np.transpose(source_csi, (0, 3, 1, 2))
    timestamps = np.asarray([record["timestamp_low"] for record in records], dtype=np.float64)
    timestamps = (timestamps - timestamps[0]) / 1_000_000.0
    positive_gaps = np.diff(timestamps)
    positive_gaps = positive_gaps[np.isfinite(positive_gaps) & (positive_gaps > 0)]
    observed_rate_hz = (
        float(1.0 / np.median(positive_gaps)) if positive_gaps.size else None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, timestamp_s=timestamps, csi_real=csi.real.astype(np.float32),
        csi_imag=csi.imag.astype(np.float32), amplitude=np.abs(csi).astype(np.float32),
        phase_rad=np.angle(csi).astype(np.float32),
        rssi=np.asarray([record["rssi"] for record in records], dtype=np.int16),
        noise_db=np.asarray([record["noise"] for record in records], dtype=np.int16),
        agc=np.asarray([record["agc"] for record in records], dtype=np.int16),
        valid_mask=np.ones(csi.shape[0], dtype=bool),
        subcarrier_index=np.arange(csi.shape[1], dtype=np.int16),
        tx_link_index=np.arange(tx_links, dtype=np.int16),
        rx_link_index=np.arange(rx_links, dtype=np.int16),
        source_label=np.asarray(activity_name, dtype="U64"),
        activity_label=np.asarray(activity_name, dtype="U64"),
        activity_id=np.asarray(activity_id, dtype=np.int16),
        trial_id=np.asarray(trial_id, dtype=np.int16),
    )
    sidecar = {
        "schema_version": "1.0", "dataset_id": "wiar", "source_file": input_path.name,
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_representation": "raw_iq", "standard_representation": "amplitude",
        "source_shape": list(source_csi.shape),
        "source_axis_order": ["time", "tx_link", "rx_link", "subcarrier"],
        "shape": list(csi.shape),
        "axis_order": ["time", "subcarrier", "tx_link", "rx_link"],
        "antenna_layout": {"tx_links": tx_links, "rx_links": rx_links},
        "sample_rate_hz": 30.0,
        "official_nominal_sample_rate_hz": 30.0,
        "observed_timestamp_rate_hz": observed_rate_hz,
        "source_rate_evidence": "official WiAR README nominal rate; packet timestamps retained",
        "time_axis": "timestamp_s", "time_unit": "s",
        "power_unit": "source_csi_arbitrary_unit",
        "transformations": [
            "port official read_bf_file/read_bfee bit parser",
            "ignore only an incomplete trailing source record",
            "apply receive-antenna permutation",
            "transpose [T,Tx,Rx,S] to canonical [T,S,Tx,Rx]",
            "retain amplitude, phase, and real/imaginary CSI without antenna reduction",
        ],
        "labels": {
            "activity": activity_name,
            "activity_id": str(activity_id),
            "trial": str(trial_id),
        },
        "label_sets": {"Activity": [activity_name]},
        "created_at": datetime.now(timezone.utc).isoformat(), "tool": "wisensehub-0.6.0",
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar_path
