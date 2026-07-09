#!/usr/bin/env python3
"""Build website previews for fetched dataset samples.

For every dataset with a sample under ``site/samples/<id>/`` this script:

1. records the sample file tree,
2. renders a "before" preview (amplitude heatmap + spectrogram) from the
   original sample file,
3. renders an "after" preview from the standardized NPZ produced by
   ``wisensehub prepare``,
4. merges optional collection-setup figures from ``site/assets/figures/``,
5. writes everything to ``site/data/samples.json`` for the website.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
SITE = ROOT / "site"
SAMPLES = SITE / "samples"
PREVIEWS = SITE / "assets" / "previews"
FIGURES = SITE / "assets" / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def normalize_preview_grid(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Keep native ``[subcarrier, time]`` size for display (no forced shared grid)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    native_rows, native_cols = matrix.shape
    return matrix, {
        "native_subcarriers": native_rows,
        "native_time_steps": native_cols,
        "display_subcarriers": native_rows,
        "display_time_steps": native_cols,
    }


# Fallback rates when a sidecar omits sample_rate_hz.
FALLBACK_RATE_HZ = 100.0

# Curated per-dataset collection facts that are not in the catalog JSON.
COLLECTION_SETUP = {
    "csi-bench": "In-the-wild home/office deployments across 26 environments.",
    "wallhack18k": "Five adjacent office rooms; LoS hallway path and through-wall NLoS path.",
    "nist-breathesmart": "Controlled NIST lab; robotic breathing phantom between a 3×3 MIMO WiFi link.",
    "figshare-csi-har": "Three furnished rooms at Ukrainian Catholic University; routers a few metres apart.",
    "operanet": "Two instrumented residential-style rooms with NUC WiFi CSI nodes around a monitored area.",
}

SETUP_DISTANCE = {
    "csi-bench": "Device-dependent (this sample: hex device deployed near a fridge)",
    "wallhack18k": "1.8–18 m along marked LoS/NLoS paths",
    "nist-breathesmart": "Phantom centred between transmitter and receiver (~1 m scale)",
    "figshare-csi-har": "A few metres between transmitter and receiver inside one room",
    "operanet": "Room-scale (5.1 m × 10.0 m and 7.2 m × 7.2 m floor plans)",
}

AXIS_MEANINGS = {
    "sample": "N — clip / sample index",
    "time": "T — time steps (packets)",
    "packet": "T — time steps (packets)",
    "link": "L — Tx-Rx antenna link",
    "subcarrier": "S — OFDM subcarrier",
}


# --- raw "before" loaders: return (matrix[subcarrier, time], fs_hz or None) ---

def raw_csi_bench(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    import h5py

    with h5py.File(path, "r") as handle:
        arrays = []
        handle.visititems(lambda name, obj: arrays.append(np.asarray(obj)) if hasattr(obj, "shape") else None)
    best = max(arrays, key=lambda item: item.size)
    matrix = np.squeeze(best)
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    # official layout is [subcarrier, time]: keep subcarriers on rows
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
    return matrix.astype(np.float64), None


def raw_wallhack(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    from wifi_datahub.adapters.wallhack import parse_interleaved_imag_real

    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            rows.append(np.abs(parse_interleaved_imag_real(row["data"])))
    return np.stack(rows).astype(np.float64).T, 100.0


def raw_nist(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    from wifi_datahub.adapters.official_profiles import _read_numeric_csv

    real = _read_numeric_csv(path)
    imag = _read_numeric_csv(path.with_name(path.name.replace("_csi_real_log.csv", "_csi_imag_log.csv")))
    return np.hypot(real, imag).astype(np.float64).T, 10.0


def raw_figshare(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    matrix = np.atleast_2d(np.genfromtxt(path, delimiter=","))
    matrix = matrix[:, ~np.all(np.isnan(matrix), axis=0)]
    return np.nan_to_num(matrix).astype(np.float64).T, None


def raw_operanet(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    import re

    from scipy.io import loadmat
    from wifi_datahub.adapters.official_profiles import _column_mapping

    columns = _column_mapping(loadmat(path, simplify_cells=True))
    names = sorted(
        (name for name in columns if re.fullmatch(r"tx\d+rx\d+_sub\d+", name)),
        key=lambda name: tuple(int(v) for v in re.findall(r"\d+", name)),
    )
    matrix = np.column_stack([np.abs(np.asarray(columns[name])) for name in names])
    return matrix.astype(np.float64).T, None


RAW_LOADERS = {
    "csi-bench": raw_csi_bench,
    "wallhack18k": raw_wallhack,
    "nist-breathesmart": raw_nist,
    "figshare-csi-har": raw_figshare,
    "operanet": raw_operanet,
}


# --- plotting ---------------------------------------------------------------

def render_preview(matrix: np.ndarray, fs: Optional[float], title: str, destination: Path,
                   ylabel: str = "subcarrier index") -> Dict[str, int]:
    """Paper-style CSI amplitude heatmap: x = time, y = subcarrier index."""
    matrix, dims = normalize_preview_grid(matrix)
    rate = fs or 1.0
    time_unit = "s" if fs else "packets"
    duration = matrix.shape[1] / rate

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=110)
    extent = [0.0, duration, matrix.shape[0], 0.0]
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", extent=extent, interpolation="nearest")
    ax.set_title(f"{title}: CSI amplitude", fontsize=11)
    ax.set_xlabel(f"time ({time_unit})")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="amplitude (a.u.)")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return dims


# --- inventory helpers ------------------------------------------------------

def file_tree(root: Path) -> List[dict]:
    def walk(directory: Path) -> List[dict]:
        entries = []
        for path in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if path.is_dir():
                entries.append({"name": path.name, "type": "dir", "children": walk(path)})
            else:
                entries.append({"name": path.name, "type": "file", "bytes": path.stat().st_size})
        return entries

    return walk(root)


def count_tree_files(entries: List[dict]) -> int:
    total = 0
    for entry in entries:
        if entry.get("type") == "file":
            total += 1
        else:
            total += count_tree_files(entry.get("children") or [])
    return total


def sync_standardized_into_sample(dataset_id: str) -> Optional[Path]:
    """Copy prepare output into the hosted sample so downloads match the UI."""
    source = DATA / dataset_id / "standardized"
    destination = SAMPLES / dataset_id / "standardized"
    if not source.exists():
        if destination.exists():
            shutil.rmtree(destination)
        return None
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def split_sample_trees(sample_dir: Path, dataset_id: Optional[str] = None) -> Dict[str, object]:
    """Split into original release, real prepare output, and by_label use-case trees."""
    original_entries: List[dict] = []
    usecase_entries: List[dict] = []
    sample_standardized: List[dict] = []
    if sample_dir.exists():
        for path in sorted(sample_dir.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if path.name == "by_label" and path.is_dir():
                usecase_entries = file_tree(path)
                continue
            if path.name == "standardized" and path.is_dir():
                sample_standardized = file_tree(path)
                continue
            if path.is_dir():
                original_entries.append({"name": path.name, "type": "dir", "children": file_tree(path)})
            else:
                original_entries.append({"name": path.name, "type": "file", "bytes": path.stat().st_size})

    data_standardized = DATA / (dataset_id or sample_dir.name) / "standardized"
    standardized_children = sample_standardized
    if not standardized_children and data_standardized.exists():
        standardized_children = file_tree(data_standardized)
    standardized_entries: List[dict] = []
    if standardized_children:
        standardized_entries = [{
            "name": "standardized",
            "type": "dir",
            "children": standardized_children,
        }]

    combined = list(original_entries)
    if standardized_entries:
        combined.extend(standardized_entries)
    if usecase_entries:
        combined.append({"name": "by_label", "type": "dir", "children": usecase_entries})
    return {
        "file_tree": combined,
        "original_file_tree": original_entries,
        "standardized_file_tree": standardized_entries,
        "usecase_file_tree": usecase_entries,
        "original_file_count": count_tree_files(original_entries),
        "standardized_file_count": count_tree_files(standardized_entries),
        "usecase_file_count": count_tree_files(usecase_entries),
    }


def slugify_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "unknown"


def converted_records(dataset_id: str) -> List[dict]:
    manifest_path = DATA / dataset_id / "prepare-manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        record for record in manifest.get("records", [])
        if record.get("status") in {"converted", "skipped"} and record.get("output")
    ]


def first_converted_record(dataset_id: str) -> Optional[dict]:
    records = converted_records(dataset_id)
    return records[0] if records else None


def resolve_record_path(record_path: str) -> Path:
    path = Path(record_path)
    return path if path.is_absolute() else ROOT / path


def clip_label_name(sidecar: dict, archive: Optional[dict] = None) -> Optional[str]:
    labels = sidecar.get("labels") or {}
    for key in ("activity", "class", "pattern", "bpm"):
        value = labels.get(key)
        if value not in (None, ""):
            return str(value)
    if archive and "source_label" in archive:
        values = np.asarray(archive["source_label"]).reshape(-1)
        if values.size == 1:
            return str(values[0])
        if values.size > 1 and len({str(item) for item in values}) == 1:
            return str(values[0])
    return None


def collect_label_entries(dataset_id: str) -> Dict[str, List[dict]]:
    catalog: Dict[str, List[dict]] = {}
    for record in converted_records(dataset_id):
        # Prefer native NPZ for label discovery so cropped task views do not drop activities.
        native_path = resolve_record_path(record["native_output"]) if record.get("native_output") else None
        view_path = resolve_record_path(record["output"])
        candidates = []
        if native_path and native_path.exists():
            candidates.append(native_path)
        if view_path.exists() and view_path != native_path:
            candidates.append(view_path)
        for npz_path in candidates:
            sidecar_path = npz_path.with_suffix(".json")
            if not sidecar_path.exists():
                continue
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as archive:
                archive_dict = {name: archive[name] for name in archive.files}
            clip_label = clip_label_name(sidecar, archive_dict)
            if clip_label:
                catalog.setdefault(clip_label, []).append({
                    "npz_path": npz_path,
                    "sidecar_path": sidecar_path,
                    "sidecar": sidecar,
                    "record": record,
                })
            for segment in sidecar.get("segments") or []:
                label = str(segment.get("label") or segment.get("source_label") or "")
                if not label:
                    continue
                catalog.setdefault(label, []).append({
                    "npz_path": npz_path,
                    "sidecar_path": sidecar_path,
                    "sidecar": sidecar,
                    "record": record,
                    "segment": segment,
                })
            # Also recover packet-level labels when segments are missing from the sidecar.
            if "source_label" in archive_dict:
                labels = np.asarray(archive_dict["source_label"]).reshape(-1)
                if labels.size > 1:
                    rate = sidecar.get("sample_rate_hz") or 1.0
                    start = 0
                    current = str(labels[0])
                    for index in range(1, labels.size + 1):
                        if index == labels.size or str(labels[index]) != current:
                            catalog.setdefault(current, []).append({
                                "npz_path": npz_path,
                                "sidecar_path": sidecar_path,
                                "sidecar": sidecar,
                                "record": record,
                                "segment": {
                                    "start_seconds": start / float(rate),
                                    "end_seconds": index / float(rate),
                                    "label": current,
                                    "source_label": current,
                                },
                            })
                            if index < labels.size:
                                start = index
                                current = str(labels[index])
    return catalog


def segment_time_indices(segment: dict, length: int, rate: Optional[float]) -> Tuple[int, int]:
    """Map a sidecar segment onto packet indices.

    When ``sample_rate_hz`` is unknown, adapters store packet indices in the
    ``*_seconds`` fields. Using a fallback Hz would overshoot and skip the crop.
    """
    start_raw = float(segment["start_seconds"])
    end_raw = float(segment["end_seconds"])
    if rate:
        start = int(round(start_raw * float(rate)))
        end = int(round(end_raw * float(rate)))
        if 0 <= start < length and start < end <= length:
            return start, end
    start = int(round(start_raw))
    end = int(round(end_raw))
    return max(0, min(start, length)), max(0, min(end, length))


def choose_label_entry(entries: List[dict]) -> dict:
    segmented = [entry for entry in entries if entry.get("segment")]
    pool = segmented or entries

    def score(entry: dict) -> Tuple[int, float]:
        # Prefer native NPZ over cropped task views so every activity remains available.
        native = 1 if "views" not in entry["npz_path"].parts else 0
        duration = 0.0
        if entry.get("segment"):
            duration = float(entry["segment"]["end_seconds"]) - float(entry["segment"]["start_seconds"])
        return native, duration

    return max(pool, key=score)


def matrix_for_label_entry(entry: dict) -> Tuple[np.ndarray, Optional[float]]:
    matrix, fs = after_matrix(entry["npz_path"], entry["sidecar_path"])
    segment = entry.get("segment")
    if not segment:
        return matrix, fs
    sidecar = entry["sidecar"]
    rate = sidecar.get("sample_rate_hz") or fs
    start, end = segment_time_indices(segment, matrix.shape[1], rate)
    if end > start:
        matrix = matrix[:, start:end]
    return matrix, rate or fs


def native_segment_span(label: str, label_path: Path) -> Optional[Tuple[int, int]]:
    rows = []
    with label_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                rows.append(row[1].strip())
    if not rows:
        return None
    best_start, best_len = 0, 0
    start = 0
    current = rows[0]
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index] != current:
            run = index - start
            if current == label and run > best_len:
                best_start, best_len = start, run
            if index < len(rows):
                start = index
                current = rows[index]
    if best_len == 0:
        return None
    return best_start, best_start + best_len


def tensor_for_label_entry(entry: dict) -> Tuple[np.ndarray, Optional[float]]:
    sidecar = entry["sidecar"]
    axes = sidecar.get("axis_order") or ["time", "link", "subcarrier"]
    with np.load(entry["npz_path"], allow_pickle=False) as archive:
        preferred = sidecar.get("standard_representation") or "amplitude"
        primary = preferred if preferred in archive.files else next(
            (name for name in ("amplitude", "csi_real", "bvp") if name in archive.files),
            archive.files[0],
        )
        tensor = np.asarray(archive[primary], dtype=np.float32)
        extras = {}
        if "source_label" in archive.files:
            extras["source_label"] = np.asarray(archive["source_label"])
    time_axis = 0 if axes and axes[0] == "time" else 1
    segment = entry.get("segment")
    if segment:
        rate = sidecar.get("sample_rate_hz")
        start, end = segment_time_indices(segment, tensor.shape[time_axis], rate)
        if end > start:
            slices = [slice(None)] * tensor.ndim
            slices[time_axis] = slice(start, end)
            tensor = tensor[tuple(slices)]
            if "source_label" in extras and extras["source_label"].ndim == 1:
                extras["source_label"] = extras["source_label"][start:end]
    return tensor, extras, sidecar.get("sample_rate_hz")


def mirror_sample_zip(dataset_id: str) -> Dict[str, int]:
    sample_dir = SAMPLES / dataset_id
    sync_standardized_into_sample(dataset_id)
    archive_path = SAMPLES / f"{dataset_id}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(sample_dir.rglob("*")):
            if path.is_file():
                handle.write(path, Path(dataset_id) / path.relative_to(sample_dir))
    total = sum(path.stat().st_size for path in sample_dir.rglob("*") if path.is_file())
    return {
        "files": sum(1 for path in sample_dir.rglob("*") if path.is_file()),
        "bytes": total,
        "zip_bytes": archive_path.stat().st_size,
    }


def export_label_data_samples(dataset_id: str, catalog: Dict[str, List[dict]]) -> List[dict]:
    """Write one mini sample folder per label under site/samples/<id>/by_label/."""
    label_root = SAMPLES / dataset_id / "by_label"
    if label_root.exists():
        shutil.rmtree(label_root)
    exported: List[dict] = []
    for label in sorted(catalog):
        slug = slugify_label(label)
        entry = choose_label_entry(catalog[label])
        dest = label_root / slug
        dest.mkdir(parents=True, exist_ok=True)
        files: List[str] = []
        record = entry.get("record") or {}
        source = record.get("source")
        source_path = resolve_record_path(source) if source else None
        segment = entry.get("segment")
        start_idx = end_idx = None
        if segment and source_path and source_path.name == "data.csv":
            label_path = source_path.with_name("label.csv")
            data = np.atleast_2d(np.genfromtxt(source_path, delimiter=","))
            rate = entry["sidecar"].get("sample_rate_hz")
            start_idx, end_idx = segment_time_indices(segment, data.shape[0], rate)
            if end_idx <= start_idx:
                span = native_segment_span(label, label_path) if label_path.exists() else None
                if span:
                    start_idx, end_idx = span
            if end_idx > start_idx:
                np.savetxt(dest / "data.csv", data[start_idx:end_idx], delimiter=",")
                files.append("data.csv")
                if label_path.exists():
                    rows = []
                    with label_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                        for row in csv.reader(handle):
                            if len(row) >= 2:
                                rows.append(row[1].strip())
                    with (dest / "label.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        for offset, value in enumerate(rows[start_idx:end_idx]):
                            writer.writerow([start_idx + offset, value])
                    files.append("label.csv")
        elif source_path and source_path.exists():
            target = dest / f"original{source_path.suffix}"
            shutil.copy2(source_path, target)
            files.append(target.name)
        tensor, extras, rate = tensor_for_label_entry(entry)
        np.savez_compressed(dest / "standardized.npz", amplitude=tensor, **extras)
        files.append("standardized.npz")
        meta = {
            "label": label,
            "slug": slug,
            "kind": "segment" if segment else "clip",
            "source_file": entry["sidecar"].get("source_file"),
            "sample_dir": f"samples/{dataset_id}/by_label/{slug}",
            "files": files,
            "tensor_shape": list(tensor.shape),
            "sample_rate_hz": rate,
            "start_packet": start_idx,
            "end_packet": end_idx,
            "segment": segment,
        }
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        files.append("metadata.json")
        meta["files"] = files
        exported.append(meta)
    return exported


def render_label_previews(dataset_id: str) -> Tuple[List[dict], Dict[str, str], Dict[str, List[dict]]]:
    catalog = collect_label_entries(dataset_id)
    previews: List[dict] = []
    errors: Dict[str, str] = {}
    for label in sorted(catalog):
        slug = slugify_label(label)
        destination = PREVIEWS / dataset_id / f"label_{slug}.png"
        try:
            entry = choose_label_entry(catalog[label])
            matrix, fs = matrix_for_label_entry(entry)
            dims = render_preview(matrix, fs or FALLBACK_RATE_HZ, label, destination)
            previews.append({
                "label": label,
                "slug": slug,
                "image": f"assets/previews/{dataset_id}/label_{slug}.png",
                "kind": "segment" if entry.get("segment") else "clip",
                "source_file": entry["sidecar"].get("source_file"),
                "dims": dims,
            })
        except Exception as exc:  # noqa: BLE001
            errors[label] = f"{type(exc).__name__}: {exc}"
    return previews, errors, catalog


def standardized_summary(npz_path: Path, sidecar_path: Path) -> dict:
    arrays = []
    with np.load(npz_path, allow_pickle=False) as archive:
        for name in archive.files:
            value = archive[name]
            arrays.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    return {
        "arrays": arrays,
        "shape": sidecar.get("shape"),
        "axis_order": sidecar.get("axis_order"),
        "standard_representation": sidecar.get("standard_representation"),
        "source_representation": sidecar.get("source_representation"),
        "sample_rate_hz": sidecar.get("sample_rate_hz"),
    }


def after_matrix(npz_path: Path, sidecar_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Return the standardized tensor as [subcarrier/channel, time]."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    axes = sidecar.get("axis_order") or []
    with np.load(npz_path, allow_pickle=False) as archive:
        primary = sidecar.get("standard_representation")
        key = primary if primary in archive.files else "amplitude"
        if key not in archive.files:
            key = "amplitude"
        tensor = np.asarray(archive[key], dtype=np.float64)
    if axes and axes[0] == "sample" and tensor.ndim >= 3:
        per_sample_features = int(np.prod(tensor.shape[2:]))
        if per_sample_features == 1:
            return tensor.reshape(tensor.shape[0], tensor.shape[1]), sidecar.get("sample_rate_hz")
        tensor = tensor[0]
    if len(axes) >= 3 and axes[0] == "time" and tensor.ndim == 3:
        return tensor.transpose(1, 2, 0).reshape(-1, tensor.shape[0]), sidecar.get("sample_rate_hz")
    if len(axes) >= 2 and axes[0] == "time" and tensor.ndim == 2:
        return tensor.T, sidecar.get("sample_rate_hz")
    return tensor.reshape(tensor.shape[0], -1).T, sidecar.get("sample_rate_hz")


def dimension_table(summary: dict) -> List[dict]:
    """Describe each dimension of the primary standardized array."""
    primary = summary.get("standard_representation") or "amplitude"
    array = next((item for item in summary.get("arrays", []) if item["name"] == primary), None)
    if array is None:
        array = next((item for item in summary.get("arrays", []) if item["name"] == "amplitude"), None)
    if array is None or not summary.get("axis_order"):
        return []
    return [
        {"axis": axis, "size": size, "meaning": AXIS_MEANINGS.get(axis, axis)}
        for axis, size in zip(summary["axis_order"], array["shape"])
    ]


def original_profile(dataset_id: str, matrix: np.ndarray, fs: Optional[float], source_file: str,
                     dims: Dict[str, int]) -> Dict[str, object]:
    catalog_path = ROOT / "catalog" / "datasets" / f"{dataset_id}.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    rate = fs
    time_steps = dims.get("native_time_steps")
    info: Dict[str, object] = {
        "source_file": source_file,
        "sampling_rate": f"{rate:g} Hz" if rate else "not reported by the source file",
        "duration": f"{time_steps / rate:.1f} s ({time_steps} packets)" if rate and time_steps else (f"{time_steps} packets" if time_steps else None),
        "frequency_band": (catalog.get("hardware") or {}).get("band"),
        "wifi_standard": (catalog.get("hardware") or {}).get("wifi_standard"),
        "setup_distance": SETUP_DISTANCE.get(dataset_id),
        "collection_setup": COLLECTION_SETUP.get(dataset_id) or (catalog.get("settings") or {}).get("scenario"),
        "tensor_shape": f"{dims['native_subcarriers']} subcarriers × {dims['native_time_steps']} time steps",
    }
    return {key: value for key, value in info.items() if value is not None}


def standardized_profile(dataset_id: str, summary: dict, native_summary: Optional[dict] = None) -> Dict[str, object]:
    catalog_path = ROOT / "catalog" / "datasets" / f"{dataset_id}.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    dims = dimension_table(summary)
    sizes = {item["axis"]: item["size"] for item in dims}
    time_steps = sizes.get("time") or sizes.get("packet")
    subcarriers = sizes.get("subcarrier")
    links = sizes.get("link")
    rate = summary.get("sample_rate_hz") or FALLBACK_RATE_HZ
    profile = summary.get("profile") or (summary.get("view_options") or {}).get("profile") or (catalog.get("standardization") or {}).get("profile")
    info: Dict[str, object] = {
        "task_profile": profile,
        "representation": summary.get("standard_representation") or "amplitude",
        "sampling_rate": f"{rate:g} Hz",
        "duration": f"{time_steps / rate:.1f} s ({time_steps} packets)" if rate and time_steps else None,
        "frequency_band": (catalog.get("hardware") or {}).get("band"),
        "wifi_standard": (catalog.get("hardware") or {}).get("wifi_standard"),
        "setup_distance": SETUP_DISTANCE.get(dataset_id),
        "collection_setup": "Task-profile view: resample/crop/pad time; keep native links and subcarriers",
        "tensor_shape": f"{links or 1} link × {subcarriers} subcarriers × {time_steps} time steps",
        "view_grid": f"{subcarriers} subcarriers × {time_steps} time steps @ {rate:g} Hz",
    }
    if native_summary:
        info["derived_from_shape"] = native_summary.get("shape")
    return {key: value for key, value in info.items() if value is not None}


def sample_info(dataset_id: str, summary: dict, fs: Optional[float], native_dims: Optional[dict] = None) -> Dict[str, object]:
    """Recording facts: duration, sampling rate, band, distance, and collection setup."""
    catalog_path = ROOT / "catalog" / "datasets" / f"{dataset_id}.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    dims = dimension_table(summary)
    sizes = {item["axis"]: item["size"] for item in dims}
    time_steps = sizes.get("time") or sizes.get("packet") or (native_dims or {}).get("native_time_steps")
    subcarriers = sizes.get("subcarrier")
    links = sizes.get("link")
    rate = summary.get("sample_rate_hz") or fs or FALLBACK_RATE_HZ
    info: Dict[str, object] = {
        "sampling_rate": f"{rate:g} Hz",
        "duration": f"{time_steps / rate:.1f} s ({time_steps} packets)" if rate and time_steps else (f"{time_steps} packets" if time_steps else None),
        "frequency_band": (catalog.get("hardware") or {}).get("band"),
        "wifi_standard": (catalog.get("hardware") or {}).get("wifi_standard"),
        "setup_distance": SETUP_DISTANCE.get(dataset_id),
        "collection_setup": COLLECTION_SETUP.get(dataset_id) or (catalog.get("settings") or {}).get("scenario"),
        "native_subcarriers": subcarriers,
        "native_links": links,
        "preview_grid": f"{subcarriers} subcarriers × {time_steps} time steps @ {rate:g} Hz" if subcarriers and time_steps and rate else None,
    }
    if native_dims:
        info["native_shape"] = f"{native_dims['native_subcarriers']} channels × {native_dims['native_time_steps']} time steps"
    return {key: value for key, value in info.items() if value is not None}


def main() -> int:
    fetch_report = {}
    report_path = SAMPLES / "fetch-report.json"
    if report_path.exists():
        fetch_report = json.loads(report_path.read_text(encoding="utf-8"))
    figure_sources = {}
    sources_path = FIGURES / "sources.json"
    if sources_path.exists():
        figure_sources = json.loads(sources_path.read_text(encoding="utf-8"))

    payload: Dict[str, dict] = {}
    for dataset_id, fetch_info in sorted(fetch_report.items()):
        if fetch_info.get("status") != "ok":
            skipped: Dict[str, object] = {"status": "skipped", "reason": fetch_info.get("reason")}
            figure_path = FIGURES / f"{dataset_id}.png"
            if figure_path.exists():
                skipped["setup_figure"] = f"assets/figures/{dataset_id}.png"
                skipped["figure_source"] = figure_sources.get(dataset_id)
            payload[dataset_id] = skipped
            continue
        sample_dir = SAMPLES / dataset_id
        tree_info = split_sample_trees(sample_dir, dataset_id)
        entry: Dict[str, object] = {
            "status": "ok",
            "note": fetch_info.get("note"),
            "sample_zip": f"samples/{dataset_id}.zip",
            "sample_bytes": fetch_info.get("bytes"),
            "zip_bytes": fetch_info.get("zip_bytes"),
            "file_count": fetch_info.get("files"),
            "previews": {},
            **tree_info,
        }
        figure_path = FIGURES / f"{dataset_id}.png"
        if figure_path.exists():
            entry["setup_figure"] = f"assets/figures/{dataset_id}.png"
            entry["figure_source"] = figure_sources.get(dataset_id)

        record = first_converted_record(dataset_id)
        source_path = Path(record["source"]) if record else None
        if source_path and not source_path.is_absolute():
            source_path = ROOT / source_path
        loader = RAW_LOADERS.get(dataset_id)
        raw_fs: Optional[float] = None
        ylabel = "subcarrier index"
        if loader and source_path and source_path.exists():
            try:
                matrix, raw_fs = loader(source_path)
                before_png = PREVIEWS / dataset_id / "before.png"
                before_dims = render_preview(matrix, raw_fs, "Original", before_png, ylabel)
                entry["preview_before_dims"] = before_dims
                entry["previews"]["before"] = f"assets/previews/{dataset_id}/before.png"
                entry["preview_source_file"] = source_path.name
                entry["original"] = {
                    "profile": original_profile(dataset_id, matrix, raw_fs, source_path.name, before_dims),
                    "dimensions": [{
                        "axis": "subcarrier",
                        "size": before_dims["native_subcarriers"],
                        "meaning": "OFDM subcarrier / flattened channel index",
                    }, {
                        "axis": "time",
                        "size": before_dims["native_time_steps"],
                        "meaning": "T — time steps (packets)",
                    }],
                }
            except Exception as exc:  # noqa: BLE001 - keep the page usable without a preview
                entry["preview_error"] = f"before: {type(exc).__name__}: {exc}"

        native_summary = None
        if record:
            npz_path = resolve_record_path(record["output"])
            native_path = resolve_record_path(record["native_output"]) if record.get("native_output") else None
            sidecar_path = npz_path.with_suffix(".json")
            if native_path and native_path.exists():
                native_summary = standardized_summary(native_path, native_path.with_suffix(".json"))
            if npz_path.exists():
                summary = standardized_summary(npz_path, sidecar_path)
                entry["standardized"] = summary
                entry["dimensions"] = dimension_table(summary)
                entry["standardized_view"] = {
                    "profile": standardized_profile(dataset_id, summary, native_summary),
                    "dimensions": dimension_table(summary),
                }
                entry["native_output"] = str(native_path.relative_to(ROOT)) if native_path else None
                try:
                    matrix, fs = after_matrix(npz_path, sidecar_path)
                    after_png = PREVIEWS / dataset_id / "after.png"
                    after_dims = render_preview(matrix, fs or FALLBACK_RATE_HZ, "Standardized view", after_png, ylabel)
                    entry["preview_after_dims"] = after_dims
                    entry["previews"]["after"] = f"assets/previews/{dataset_id}/after.png"
                except Exception as exc:  # noqa: BLE001
                    entry["preview_error"] = f"after: {type(exc).__name__}: {exc}"
            elif loader and source_path and source_path.exists():
                try:
                    matrix, raw_fs = loader(source_path)
                    entry["original"] = entry.get("original") or {
                        "profile": original_profile(dataset_id, matrix, raw_fs, source_path.name, normalize_preview_grid(matrix)[1]),
                        "dimensions": [],
                    }
                except Exception:
                    pass
        label_previews, label_errors, label_catalog = render_label_previews(dataset_id)
        if label_catalog:
            label_samples = export_label_data_samples(dataset_id, label_catalog)
            entry["label_samples"] = label_samples
            entry["label_coverage"] = {
                "expected": len(label_catalog),
                "exported": len(label_samples),
                "complete": len(label_samples) == len(label_catalog),
            }
            for preview in label_previews:
                sample = next((item for item in label_samples if item["label"] == preview["label"]), None)
                if sample:
                    preview["sample_dir"] = sample["sample_dir"]
                    preview["sample_files"] = sample["files"]
            try:
                zip_stats = mirror_sample_zip(dataset_id)
                entry.update(split_sample_trees(SAMPLES / dataset_id, dataset_id))
                entry.update({
                    "file_count": zip_stats["files"],
                    "sample_bytes": zip_stats["bytes"],
                    "zip_bytes": zip_stats["zip_bytes"],
                })
            except Exception as exc:  # noqa: BLE001
                entry["sample_zip_error"] = f"{type(exc).__name__}: {exc}"
        if label_previews:
            entry["label_previews"] = label_previews
            entry["previews"]["by_label"] = [item["image"] for item in label_previews]
        if label_errors:
            entry["label_preview_errors"] = label_errors
        std = entry.get("standardized") or {}
        dims = {item["axis"]: item["size"] for item in entry.get("dimensions") or []}
        entry["preview_window"] = {
            "profile": std.get("profile") or (std.get("view_options") or {}).get("profile"),
            "time_steps": dims.get("time") or dims.get("packet"),
            "subcarriers": dims.get("subcarrier"),
            "links": dims.get("link"),
            "sample_rate_hz": std.get("sample_rate_hz"),
            "axes": "y = subcarrier index, x = time",
            "note": "Native files are preserved. Task-profile views resample/crop/pad time only; links and subcarriers stay native.",
        }
        payload[dataset_id] = entry

    destination = SITE / "data" / "samples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": "1.0", "datasets": payload}, indent=2) + "\n",
                           encoding="utf-8")
    ready = sum(1 for item in payload.values() if item.get("status") == "ok")
    print(f"Wrote {destination.relative_to(ROOT)} ({ready} dataset samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
