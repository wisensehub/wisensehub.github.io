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
    for part in input_path.parts:
        if part.startswith("act_"):
            labels["activity"] = part[len("act_") :]
        elif part.startswith("user_"):
            labels["subject"] = part[len("user_") :]
        elif part.startswith("env_"):
            labels["environment"] = part[len("env_") :]
        elif part.startswith("sub_"):
            labels["subject_type"] = part[len("sub_") :]
        elif part.startswith("Diff_"):
            labels["difficulty"] = part[len("Diff_") :]
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
    data = np.asarray(data)
    if data.ndim == 4:
        if data.shape[-1] in (56, 114, 208, 30, 52, 100) or data.shape[1] > data.shape[0]:
            return data, ["sample", "time", "link", "subcarrier"]
        raise ValueError(f"CSI-Bench 4-D tensor has unsupported shape {data.shape}")
    if data.ndim != 3:
        raise ValueError(f"CSI-Bench expects a 3-D or 4-D CSI tensor, got {data.shape}")
    if data.shape[0] <= 32 and data.shape[1] > data.shape[0]:
        return data[:, :, None, :], ["sample", "time", "link", "subcarrier"]
    if data.shape[0] in (56, 114, 208) and data.shape[1] >= data.shape[0]:
        return data.transpose(1, 2, 0), ["time", "link", "subcarrier"]
    if data.shape[-1] in (56, 114, 208, 30, 52, 100) and data.shape[1] >= data.shape[0]:
        return data, ["time", "link", "subcarrier"]
    raise ValueError(f"CSI-Bench tensor layout is not recognized: {data.shape}")


def convert_csi_bench_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    data = _find(mapping, ("CSI_amps", "X", "csi", "CSI"))
    data = data if data is not None else _largest_numeric(mapping, 2)
    data, axes = _canonical_csi_bench_tensor(data)
    arrays = _complex_arrays(data) if np.iscomplexobj(data) else {"amplitude": data.astype(np.float32)}
    path_labels = _path_metadata_labels(input_path)
    if path_labels.get("activity"):
        arrays["activity_label"] = np.asarray(path_labels["activity"], dtype="U64")
    return _save(
        input_path, output_path, "csi-bench", arrays, "amplitude", axes,
        "complex_csi" if np.iscomplexobj(data) else "processed_amplitude",
        ["load official CSI_amps/X tensor", "transpose to canonical axes", "attach path-derived activity label", "cast float32"],
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
        frame = np.asarray(frame).squeeze()
        if frame.ndim == 3 and frame.shape[-1] in (10, 30, 114):
            frame = frame.reshape(-1, frame.shape[-1])
        elif frame.ndim == 2:
            pass
        else:
            raise ValueError(f"unsupported MM-Fi CSIamp frame shape {frame.shape}")
        frames.append(frame)
    amplitude = np.nan_to_num(np.stack(frames), copy=False).astype(np.float32)
    return _save(input_path, output_path, "mm-fi", {"amplitude": amplitude}, "amplitude",
                 ["time", "link", "subcarrier"], "processed_amplitude",
                 ["load sorted frame*.mat CSIamp", "flatten antenna pair axes into link", "replace non-finite values"])


def convert_ntu_fi_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    data = _find(mapping, ("CSIamp",))
    if data is None:
        raise ValueError("NTU-Fi MAT file must contain CSIamp")
    data = np.asarray(data).squeeze()
    if data.ndim == 2 and data.shape[0] == 342:
        data = data.reshape(3, 114, data.shape[1])
    if data.ndim != 3:
        raise ValueError(f"unsupported NTU-Fi CSIamp shape {data.shape}")
    if data.shape[0:2] == (3, 114):
        data = data[:, :, ::4].transpose(2, 0, 1)
    elif data.shape[-2:] == (3, 114):
        data = data[::4]
    else:
        raise ValueError(f"NTU-Fi expected [3,114,time], got {data.shape}")
    return _save(input_path, output_path, "ntu-fi", {"amplitude": data.astype(np.float32)}, "amplitude",
                 ["time", "link", "subcarrier"], "processed_amplitude",
                 ["load official CSIamp", "downsample time by four as SenseFi", "transpose to canonical axes"])


def convert_widar_csv(input_path: Path, output_path: Path) -> Path:
    value = np.genfromtxt(input_path, delimiter=",")
    if value.size != 22 * 20 * 20:
        raise ValueError(f"SenseFi Widar BVP CSV must contain 8800 values, got {value.size}")
    bvp = value.reshape(22, 20, 20).astype(np.float32)
    return _save(input_path, output_path, "widar3", {"bvp": bvp}, "bvp",
                 ["gesture_channel", "doppler_bin", "time_bin"], "processed_bvp",
                 ["load official SenseFi CSV", "reshape 22x400 to 22x20x20"], power_unit="normalized_bvp")


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
    if csi.shape[0] == 200 and csi.shape[1] == 30:
        csi = csi.transpose(3, 0, 2, 1)
    arrays = _complex_arrays(csi)
    label = _find(mapping, ("label_lab", "label_home", "label", "labels"))
    if label is not None:
        arrays["source_label"] = np.asarray(label).reshape(-1)
    return _save(input_path, output_path, "signfi", arrays, "amplitude",
                 ["sample", "time", "link", "subcarrier"], "complex_csi",
                 ["load official csid tensor", "transpose 200x30x3xN to canonical axes", "derive amplitude and phase"])


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
    canonical, axes = _canonical_sequence(value)
    arrays = _complex_arrays(canonical)
    return _save(input_path, output_path, "wimans", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 ["load official WiMANS MAT/amp NPY", "flatten antenna axes when present", "cast float32"])


def convert_xrf55_npy(input_path: Path, output_path: Path) -> Path:
    value = np.load(input_path, allow_pickle=False)
    canonical, axes = _canonical_sequence(value)
    arrays = _complex_arrays(canonical)
    return _save(input_path, output_path, "xrf55", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 ["load official XRF55 WiFi NPY", "canonicalize time/link/subcarrier axes", "cast float32"])


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
    timestamp_delta = _find(mapping, ("Timestamp",))
    if timestamp_delta is not None and np.asarray(timestamp_delta).size == csi.shape[0]:
        arrays["timestamp_s"] = np.cumsum(np.asarray(timestamp_delta).reshape(-1).astype(np.float64)) / 1000.0
    rssi = _find(mapping, ("RSSI",))
    if rssi is not None:
        arrays["rssi_dbm"] = np.asarray(rssi).reshape(-1).astype(np.float32)
    return _save(input_path, output_path, "ehunam", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["load official CSI/BW/Subcarriers metadata", "remove null and pilot carriers using official Subcarrier.m", "derive amplitude and phase"])


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
    return _save(input_path, output_path, "wifi-presence-movement",
                 {"csi_real": real, "csi_imag": imag, "amplitude": amplitude, "timestamp_s": timestamps},
                 "amplitude", ["time", "link", "subcarrier"], "complex_csi",
                 ["stream gzip JSON Lines", "parse official r/i complex fields", "transpose subcarrier/link axes", "normalize epoch timestamps to elapsed seconds"])


def _find_ancestor_child(path: Path, child: str) -> Path | None:
    for parent in (path.parent, *path.parents):
        candidate = parent / child
        if candidate.exists():
            return candidate
    return None


def convert_wifi_tad_npy(input_path: Path, output_path: Path) -> Path:
    data = np.load(input_path, allow_pickle=False).squeeze()
    if data.ndim != 2:
        raise ValueError(f"WiFiTAD NPY must be 2-D, got {data.shape}")
    # The official loader applies np.transpose before feeding a 60-channel
    # temporal tensor. Canonical storage keeps time first.
    if data.shape[0] == 60 and data.shape[1] != 60:
        time_first = data.T
    elif data.shape[1] == 60:
        time_first = data
    else:
        raise ValueError(f"WiFiTAD expects one 60-channel axis, got {data.shape}")
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
                            segments.append((float(row[-2]) * ratio, float(row[-1]) * ratio, int(float(row[2]))))
                        except ValueError:
                            continue
        if segments:
            arrays["segment_start_index"] = np.asarray([item[0] for item in segments], dtype=np.float32)
            arrays["segment_end_index"] = np.asarray([item[1] for item in segments], dtype=np.float32)
            arrays["segment_label"] = np.asarray([item[2] for item in segments], dtype=np.int16)
    return _save(input_path, output_path, "wifi-tad", arrays, "amplitude",
                 ["time", "link", "feature"], "processed_amplitude",
                 ["load official smartwifi NPY", "orient 60 channels after official transpose", "store official amplitude/40 normalization", "attach temporal annotation indices"])


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
    csi = flat.reshape(flat.shape[0], 9, 30)
    arrays = _complex_arrays(csi)
    lowered = {name.lower(): name for name in columns}
    for source_name, target_name in {
        "activity": "source_label", "person_id": "subject", "room_no": "environment", "exp_no": "experiment"
    }.items():
        if source_name in lowered:
            arrays[target_name] = np.asarray(columns[lowered[source_name]]).astype("U64")
    if "timestamp" in lowered:
        timestamp = np.asarray(columns[lowered["timestamp"]], dtype=np.float64)
        arrays["timestamp_s"] = (timestamp - timestamp[0]) / 1000.0
    return _save(input_path, output_path, "operanet", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["load official MATLAB table", "order tx1rx1_sub1 through tx3rx3_sub30", "reshape 270 complex fields to 9 links x 30 subcarriers", "attach activity/person/room metadata"])


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
        links, subcarriers = 9, 114
    elif real.shape[1] == 504:
        links, subcarriers = 9, 56
    elif real.shape[1] % 114 == 0:
        links, subcarriers = real.shape[1] // 114, 114
    elif real.shape[1] % 56 == 0:
        links, subcarriers = real.shape[1] // 56, 56
    else:
        raise ValueError(f"BreatheSmart CSI width is not compatible with 56/114 subcarriers: {real.shape[1]}")
    real = real.reshape(real.shape[0], links, subcarriers).astype(np.float32)
    imag = imag.reshape(imag.shape[0], links, subcarriers).astype(np.float32)
    arrays: Dict[str, np.ndarray] = {
        "csi_real": real, "csi_imag": imag, "amplitude": np.hypot(real, imag).astype(np.float32),
        "timestamp_s": np.arange(real.shape[0], dtype=np.float64) / 10.0,
    }
    config_candidates = sorted(input_path.parent.glob("config*.csv")) + sorted(input_path.parent.glob("config*.cvs"))
    if config_candidates:
        with config_candidates[0].open(newline="", encoding="utf-8-sig", errors="ignore") as handle:
            rows = list(csv.reader(handle))
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                key = re.sub(r"[^a-z0-9]+", "_", row[0].strip().lower()).strip("_")
                if key:
                    arrays[f"config_{key}"] = np.asarray(str(row[1]).strip())
    return _save(input_path, output_path, "nist-breathesmart", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["pair official real and imaginary CSV logs", "reshape 3x3 MIMO x 56/114 subcarriers", "attach config CSV values"],
                 sample_rate_hz=10.0)


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
    amplitude = np.asarray(zarr.open(str(amp_path), mode="r"), dtype=np.float32)
    if amplitude.ndim != 4:
        raise ValueError(f"CSIDA amplitude must be [sample,time,3,114], got {amplitude.shape}")
    arrays: Dict[str, np.ndarray] = {"amplitude": amplitude}
    if phase_path.exists():
        phase = np.asarray(zarr.open(str(phase_path), mode="r"), dtype=np.float32)
        if phase.shape != amplitude.shape:
            raise ValueError("CSIDA amplitude and phase shapes differ")
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
            if values.size != amplitude.shape[0]:
                raise ValueError(f"CSIDA {source_name} count does not match samples")
            arrays[target_name] = values
    return _save(input_path, output_path, "csida", arrays, "amplitude",
                 ["sample", "time", "link", "subcarrier"], "amplitude_phase",
                 ["open official Zarr arrays", "align amplitude/phase and four label arrays", "reconstruct real and imaginary CSI"])


def convert_exposing_csi_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("csi_buff",))
    if csi is None:
        raise ValueError("Exposing the CSI MAT file must contain csi_buff")
    csi = np.asarray(csi)
    if csi.ndim != 2 or csi.shape[1] < 2048:
        raise ValueError(f"expected AX-CSI csi_buff [packet,2048+], got {csi.shape}")
    csi = np.fft.fftshift(csi[:, :2048], axes=1)
    csi = csi[np.sum(np.abs(csi), axis=1) != 0]
    remove = np.asarray([
        *range(0, 12), 509, 510, 511, 512, 513, 514, 1013, 1014, 1015, 1016, 1017,
        1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
        1030, 1031, 1032, 1033, 1034, 1035, 1535, 1536, 1537, 1538, 1539,
        2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047,
    ], dtype=np.int32)
    streams = []
    for stream in range(4):
        value = csi[stream::4]
        value = np.delete(value, remove, axis=1)
        scale = np.mean(np.abs(value), axis=1, keepdims=True)
        streams.append(value / np.maximum(scale, np.finfo(np.float32).eps))
    length = min(value.shape[0] for value in streams)
    canonical = np.stack([value[:length] for value in streams], axis=1)
    arrays = _complex_arrays(canonical)
    activity = re.search(r"(?:^|_)([A-L])(?:_|$)", input_path.stem.upper())
    if activity:
        arrays["source_label"] = np.asarray(activity.group(1))
    arrays["subcarrier_index"] = np.delete(np.arange(2048, dtype=np.int32), remove)
    return _save(input_path, output_path, "exposing-csi", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["load official csi_buff", "FFT-shift 160 MHz AX-CSI", "remove official null carriers", "deinterleave four monitor streams", "normalize each packet by mean amplitude"],
                 sample_rate_hz=1.0 / 0.006)


def convert_wifi_80mhz_mat(input_path: Path, output_path: Path) -> Path:
    mapping = _load_mat(input_path)
    csi = _find(mapping, ("csi_buff", "CFR", "cfr", "CSI", "csi"))
    csi = csi if csi is not None else _largest_numeric(mapping, 2)
    csi = np.asarray(csi)
    if csi.ndim != 2:
        raise ValueError(f"80 MHz CFR trace must be a 2-D complex matrix, got {csi.shape}")
    shifted = False
    if csi.shape[1] == 1024:
        csi = np.fft.fftshift(csi, axes=1)[:, :1024:4]
        shifted = True
    if csi.shape[1] == 256:
        if not shifted:
            csi = np.fft.fftshift(csi, axes=1)
        remove = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255])
        csi = np.delete(csi, remove, axis=1)
    elif csi.shape[1] != 242:
        raise ValueError(f"80 MHz trace expects 242, 256, or 1024 frequency bins, got {csi.shape[1]}")
    csi = csi[np.sum(np.abs(csi), axis=1) != 0]
    if csi.shape[0] < 4:
        raise ValueError("80 MHz trace contains fewer than four monitor-antenna packets")
    length = csi.shape[0] // 4
    streams = [csi[index:length * 4:4] for index in range(4)]
    canonical = np.stack(streams, axis=1)
    arrays = _complex_arrays(canonical)
    activity = re.search(r"(?:^|_)([WRJLSCGE])(?:_|\d|$)", input_path.stem.upper())
    if activity:
        arrays["source_label"] = np.asarray(activity.group(1))
    return _save(input_path, output_path, "wifi-80mhz", arrays, "amplitude",
                 ["time", "link", "subcarrier"], "complex_csi",
                 ["load official csi_buff CFR trace", "FFT-shift and select 242 data subcarriers", "deinterleave four monitor antennas"],
                 sample_rate_hz=173.0)


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
    amplitude = value[:, None, :].astype(np.float32)
    arrays: Dict[str, np.ndarray] = {
        "amplitude": amplitude,
        "source_label": np.asarray(input_path.parent.name),
    }
    return _save(input_path, output_path, dataset_id, arrays, "amplitude",
                 ["time", "link", "subcarrier"], "processed_amplitude",
                 transformations + ["insert singleton USRP link axis", "cast float32"])


def convert_wireless_har_wifi(input_path: Path, output_path: Path) -> Path:
    """Convert only the WiFi_CSI branch of the paired WiFi/UWB release."""
    if "wifi_csi" not in {part.lower() for part in input_path.parts}:
        raise ValueError("wireless HAR adapter only accepts files inside the official WiFi_CSI directory")
    if input_path.suffix.lower() == ".csv":
        return convert_usrp_amplitude_csv("wireless-har-wifi-uwb", input_path, output_path)
    source, _ = load_generic_source(input_path)
    data = _find(source, ("csi", "CSI", "amp", "amplitude", "data", "x", "input"))
    if data is None:
        data = _largest_numeric(source, 1)
    canonical, axes = _canonical_sequence(np.asarray(data))
    arrays = _complex_arrays(canonical)
    return _save(input_path, output_path, "wireless-har-wifi-uwb", arrays, "amplitude", axes,
                 "complex_csi" if np.iscomplexobj(canonical) else "processed_amplitude",
                 ["select official WiFi_CSI branch", "load released numeric CSI array", "canonicalize time/link/subcarrier axes"])


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
