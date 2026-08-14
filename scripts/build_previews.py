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

import argparse
import csv
import gzip
import hashlib
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
HOSTED_LABEL_MAX_TIME_STEPS = 200


def versioned_site_asset(path: Path) -> str:
    """Return a content-versioned site URL so rebuilt images bypass caches."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.relative_to(SITE).as_posix()}?v={digest}"

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
    "nist-breathesmart": "Controlled NIST lab; robotic breathing phantom between a 2 Tx × 3 Rx MIMO WiFi link.",
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
    "sample": "clip / sample index",
    "time": "T — time steps (packets)",
    "packet": "T — time steps (packets)",
    "link": "L — Tx-Rx antenna link",
    "subcarrier": "S — OFDM subcarrier",
    "tx_link": "Tx — transmit antenna / device link",
    "rx_link": "Rx — receive antenna / device link",
    "time_bin": "T — semantic BVP time snapshot",
    "velocity_x_bin": "Vx — body-coordinate x-velocity bin",
    "velocity_y_bin": "Vy — body-coordinate y-velocity bin",
    "flattened_velocity_bin": "F — flattened Vx × Vy bin",
}


# --- raw "before" loaders: return (matrix[subcarrier, time], fs_hz or None) ---

def raw_aril(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Read one ARIL sample without flattening the sample axis into channels."""
    from scipy.io import loadmat

    payload = loadmat(path, simplify_cells=True)
    key = next((name for name in ("train_data", "test_data") if name in payload), None)
    if key is None:
        raise ValueError(f"no ARIL train_data or test_data array found in {path.name}")

    data = np.abs(np.asarray(payload[key])).squeeze()
    if data.ndim != 3:
        raise ValueError(f"expected ARIL [sample, 52, 192] data, got {data.shape}")

    sample = data[0]
    if sample.shape == (192, 52):
        sample = sample.T
    elif sample.shape != (52, 192):
        raise ValueError(f"expected ARIL sample shape 52×192 or 192×52, got {sample.shape}")
    return sample.astype(np.float64), None


def raw_csida_rx_grids(path: Path, sample_index: int = 0) -> Tuple[np.ndarray, Optional[float]]:
    """Read one official CSIDA clip as [Rx, subcarrier, time]."""
    import zarr

    value = np.asarray(zarr.open(str(path), mode="r"), dtype=np.float64)
    if value.ndim != 4 or value.shape[2:] != (3, 114):
        raise ValueError(f"expected CSIDA [sample,time,3 Rx,114 subcarrier], got {value.shape}")
    if sample_index < 0 or sample_index >= value.shape[0]:
        raise IndexError(f"CSIDA sample index {sample_index} is outside {value.shape[0]} clips")
    return np.abs(value[sample_index]).transpose(1, 2, 0), 1000.0


def standardized_csida_rx_grids(
    npz_path: Path, sidecar_path: Path, sample_index: int = 0,
) -> Tuple[np.ndarray, Optional[float]]:
    """Read a canonical [sample,T,S,Tx,Rx] view as [Rx,S,T]."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    axes = list(sidecar.get("axis_order") or [])
    with np.load(npz_path, allow_pickle=False) as archive:
        primary = sidecar.get("standard_representation") or "amplitude"
        tensor = np.asarray(archive[primary], dtype=np.float64)
    if axes and axes[0] == "sample":
        tensor = tensor[sample_index]
        axes = axes[1:]
    expected = ["time", "subcarrier", "tx_link", "rx_link"]
    if axes != expected or tensor.ndim != 4:
        raise ValueError(f"expected canonical CSIDA axes {expected}, got {axes} and {tensor.shape}")
    if tensor.shape[2] < 1:
        raise ValueError("CSIDA standardized tensor has no Tx link")
    return np.abs(tensor[:, :, 0, :]).transpose(2, 1, 0), sidecar.get("sample_rate_hz")


def canonical_csi_tensor(
    npz_path: Path, sidecar_path: Path, sample_index: int = 0,
) -> Tuple[np.ndarray, Optional[float]]:
    """Read one sample as canonical [T,S,Tx,Rx]."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    axes = list(sidecar.get("axis_order") or [])
    with np.load(npz_path, allow_pickle=False) as archive:
        preferred = sidecar.get("standard_representation") or "amplitude"
        primary = preferred if preferred in archive.files else next(
            (name for name in ("amplitude", "csi_real") if name in archive.files), archive.files[0],
        )
        tensor = np.asarray(archive[primary], dtype=np.float64)
    if axes and axes[0] == "sample":
        if sample_index < 0 or sample_index >= tensor.shape[0]:
            raise IndexError(f"sample index {sample_index} is outside {tensor.shape[0]} samples")
        tensor = tensor[sample_index]
        axes = axes[1:]
    expected = ["time", "subcarrier", "tx_link", "rx_link"]
    packet_expected = ["packet", "subcarrier", "tx_link", "rx_link"]
    if axes not in (expected, packet_expected) or tensor.ndim != 4:
        raise ValueError(f"expected canonical CSI axes {expected} (time may be packet), got {axes} and {tensor.shape}")
    # Amplitude arrays may be signed processed features (for example UT-HAR).
    # Preserve the stored values; complex CSI adapters already provide a true
    # non-negative amplitude array as their standard representation.
    return tensor, sidecar.get("sample_rate_hz")


def semantic_feature_matrix(
    npz_path: Path, sidecar_path: Path, sample_index: int = 0,
) -> Tuple[np.ndarray, Optional[float], Dict[str, object]]:
    """Project a documented non-CSI feature tensor to [feature, time]."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    axes = list(sidecar.get("axis_order") or [])
    with np.load(npz_path, allow_pickle=False) as archive:
        preferred = sidecar.get("standard_representation")
        primary = preferred if preferred in archive.files else next(
            (name for name in ("bvp", "official_normalized_amplitude", "amplitude") if name in archive.files),
            archive.files[0],
        )
        tensor = np.asarray(archive[primary], dtype=np.float64)
    if axes and axes[0] == "sample":
        tensor = tensor[sample_index]
        axes = axes[1:]

    if primary == "bvp" or "doppler_bin" in axes:
        if {"time_bin", "velocity_x_bin", "velocity_y_bin"}.issubset(axes):
            time_axis = axes.index("time_bin")
            x_axis = axes.index("velocity_x_bin")
            y_axis = axes.index("velocity_y_bin")
            tensor = np.transpose(tensor, (time_axis, x_axis, y_axis))
            # A compact fallback: preserve time and x velocity while summing
            # over y velocity. The main Widar preview uses the full 3-D stack.
            matrix = tensor.sum(axis=2).T
            return matrix, None, {
                "quantity": "body-velocity profile (y-marginal)",
                "ylabel": "x-velocity bin", "colorbar": "BVP power",
                "cmap": "magma", "signed": False, "time_label": "time bin",
                "display_gamma": .35,
            }
        time_name = "time_bin" if "time_bin" in axes else "time"
        time_axis = axes.index(time_name)
        feature_axis = axes.index("doppler_bin")
        # Widar BVP's gesture_channel is a bank of derived channels. Show one
        # channel while retaining the real Doppler-by-time surface.
        select = [slice(None)] * tensor.ndim
        for index, axis in enumerate(axes):
            if index not in {time_axis, feature_axis}:
                select[index] = 0
        matrix = np.asarray(tensor[tuple(select)])
        remaining = [axis for index, axis in enumerate(axes) if isinstance(select[index], slice)]
        matrix = np.transpose(matrix, (remaining.index("doppler_bin"), remaining.index(time_name)))
        return matrix, sidecar.get("sample_rate_hz"), {
            "quantity": "body-velocity profile", "ylabel": "Doppler bin",
            "colorbar": "normalized BVP", "cmap": "magma", "signed": False,
            "time_label": "time bin",
        }

    time_axis = next((axes.index(name) for name in ("time", "packet") if name in axes), None)
    feature_axis = axes.index("feature") if "feature" in axes else None
    if time_axis is None or feature_axis is None:
        raise ValueError(f"cannot project semantic feature axes {axes}")
    select = [slice(None)] * tensor.ndim
    for index in range(tensor.ndim):
        if index not in {time_axis, feature_axis}:
            select[index] = 0
    matrix = np.asarray(tensor[tuple(select)])
    remaining = [axis for index, axis in enumerate(axes) if isinstance(select[index], slice)]
    matrix = np.transpose(matrix, (remaining.index("feature"), remaining.index(axes[time_axis])))
    return matrix, sidecar.get("sample_rate_hz"), {
        "quantity": "processed temporal features", "ylabel": "feature channel",
        "colorbar": "feature value", "cmap": "viridis", "signed": bool(np.nanmin(matrix) < 0),
    }


def semantic_bvp_tensor(
    npz_path: Path, sidecar_path: Path, sample_index: int = 0,
) -> np.ndarray:
    """Load Widar BVP as [time_bin, velocity_x_bin, velocity_y_bin]."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    axes = list(sidecar.get("axis_order") or [])
    with np.load(npz_path, allow_pickle=False) as archive:
        if "bvp" not in archive.files:
            raise ValueError("Widar semantic preview requires a bvp array")
        tensor = np.asarray(archive["bvp"], dtype=np.float64)
    if axes and axes[0] == "sample":
        tensor = tensor[sample_index]
        axes = axes[1:]
    expected = {"time_bin", "velocity_x_bin", "velocity_y_bin"}
    if set(axes) != expected or tensor.ndim != 3:
        raise ValueError(f"expected Widar BVP axes {sorted(expected)}, got {axes} and {tensor.shape}")
    return np.transpose(tensor, (
        axes.index("time_bin"), axes.index("velocity_x_bin"), axes.index("velocity_y_bin"),
    ))

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
    rate: Optional[float] = None
    for candidate in sorted(path.parent.glob("config*.csv")) + sorted(path.parent.glob("config*.cvs")):
        with candidate.open(newline="", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2 and row[0].strip().lower() == "msgfreq":
                    try:
                        rate = float(row[1])
                    except ValueError:
                        pass
                    break
        if rate:
            break
    return np.hypot(real, imag).astype(np.float64).T, rate


def raw_nist_stored_tensor(path: Path) -> np.ndarray:
    """Return the unchanged NIST storage envelope as [T,S,1,stored_link]."""
    from wifi_datahub.adapters.official_profiles import _read_numeric_csv

    real = _read_numeric_csv(path)
    imag = _read_numeric_csv(path.with_name(path.name.replace("_csi_real_log.csv", "_csi_imag_log.csv")))
    amplitude = np.hypot(real, imag).astype(np.float64)
    if amplitude.shape[1] == 1026:
        stored_links, stored_subcarriers = 9, 114
    elif amplitude.shape[1] == 504:
        stored_links, stored_subcarriers = 9, 56
    else:
        raise ValueError(f"unsupported NIST stored CSI width {amplitude.shape[1]}")
    stored = amplitude.reshape(amplitude.shape[0], stored_links, stored_subcarriers)
    return stored.transpose(0, 2, 1)[:, :, None, :]


def raw_ntu_fi_tensor(path: Path) -> np.ndarray:
    """Return the unchanged NTU-Fi CSIamp payload as [T,S,1,3]."""
    from scipy.io import loadmat

    value = np.asarray(loadmat(path, simplify_cells=True)["CSIamp"]).squeeze()
    if value.ndim == 2 and value.shape[0] == 342:
        source = value.reshape(3, 114, value.shape[1])
    elif value.ndim == 3 and value.shape[:2] == (3, 114):
        source = value
    elif value.ndim == 3 and value.shape[-2:] == (3, 114):
        source = value.transpose(1, 2, 0)
    else:
        raise ValueError(f"unsupported NTU-Fi CSIamp shape {value.shape}")
    return source.transpose(2, 1, 0)[:, :, None, :].astype(np.float64)


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


def _preview_grid_from_array(value: np.ndarray) -> np.ndarray:
    """Project common official tensor layouts to [channel, time] for display."""
    value = np.abs(np.asarray(value)).squeeze()
    if value.ndim == 0:
        return value.reshape(1, 1).astype(np.float64)
    if value.ndim == 1:
        return value[None, :].astype(np.float64)
    if value.ndim == 4:
        if value.shape[0] <= 8 and value.shape[1] > 8:
            value = value[0]
        else:
            return value.reshape(value.shape[0], -1).T.astype(np.float64)
    if value.ndim == 3:
        if value.shape[0] <= 8 and value.shape[1] > 16 and value.shape[2] > 16:
            # Either [sample,feature,time] or [sample,time,feature]; one sample
            # is sufficient for a truthful raw-layout preview.
            return _preview_grid_from_array(value[0])
        return value.reshape(value.shape[0], -1).T.astype(np.float64)
    if value.ndim > 4:
        value = value.reshape(value.shape[0], -1)
    rows, cols = value.shape
    known_channels = {30, 51, 52, 56, 60, 64, 90, 100, 114, 168, 192, 208, 232, 242, 270, 342, 504, 1026, 1990}
    if rows in known_channels and cols > rows:
        return value.astype(np.float64)
    return value.T.astype(np.float64)


def raw_generic(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """Read deterministic adapter fixtures and simple official numeric sources."""
    if path.is_dir():
        if path.name == "csi_data_amp":
            import zarr
            return _preview_grid_from_array(np.asarray(zarr.open(str(path), mode="r"))), None
        from scipy.io import loadmat
        frames = []
        for frame in sorted(path.glob("frame*.mat")):
            payload = loadmat(frame, simplify_cells=True)
            if "CSIamp" in payload:
                frames.append(np.asarray(payload["CSIamp"]))
        if frames:
            return _preview_grid_from_array(np.stack(frames)), None
        raise ValueError(f"no previewable arrays found in {path}")
    name = path.name.lower()
    if name.endswith(".json.gz"):
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                packet = record.get("csi") or []
                rows.append([abs(complex(item["r"], item["i"])) for carrier in packet for item in carrier])
        return _preview_grid_from_array(np.asarray(rows)), None
    if path.suffix.lower() == ".json":
        return _preview_grid_from_array(np.asarray(json.loads(path.read_text(encoding="utf-8")))), None
    if path.suffix.lower() == ".dat":
        from wifi_datahub.adapters.intel5300 import read_bf_file
        records = read_bf_file(path)
        return _preview_grid_from_array(np.stack([record["csi"] for record in records])), 30.0
    if path.suffix.lower() in {".npy", ".npz"}:
        payload = np.load(path, allow_pickle=False)
        if isinstance(payload, np.lib.npyio.NpzFile):
            key = next((key for key in ("data", "amplitude", "x") if key in payload.files), payload.files[0])
            value = payload[key]
        else:
            value = payload
        return _preview_grid_from_array(value), None
    if path.suffix.lower() == ".csv":
        value = np.genfromtxt(path, delimiter=",")
        value = np.atleast_2d(value)
        value = value[~np.all(np.isnan(value), axis=1)]
        value = value[:, ~np.all(np.isnan(value), axis=0)]
        return _preview_grid_from_array(np.nan_to_num(value)), None
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        payload = loadmat(path, simplify_cells=True)
        candidates = [np.asarray(value) for key, value in payload.items()
                      if not key.startswith("__") and np.asarray(value).dtype.kind in "biufc" and np.asarray(value).ndim >= 2]
        if not candidates:
            raise ValueError(f"no numeric matrix found in {path.name}")
        return _preview_grid_from_array(max(candidates, key=lambda value: value.size)), None
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        arrays = []
        with h5py.File(path, "r") as handle:
            handle.visititems(lambda _name, obj: arrays.append(np.asarray(obj)) if hasattr(obj, "shape") and obj.shape else None)
        if not arrays:
            raise ValueError(f"no HDF5 arrays found in {path.name}")
        return _preview_grid_from_array(max(arrays, key=lambda value: value.size)), None
    raise ValueError(f"no generic raw preview loader for {path.name}")


RAW_LOADERS = {
    "aril": raw_aril,
    "csi-bench": raw_csi_bench,
    "wallhack18k": raw_wallhack,
    "nist-breathesmart": raw_nist,
    "figshare-csi-har": raw_figshare,
    "operanet": raw_operanet,
}


# --- plotting ---------------------------------------------------------------

def preview_color_limits(matrix: np.ndarray, *, signed: bool = False) -> Tuple[float, float]:
    """Return robust display limits without changing the stored measurements."""
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        limit = float(np.percentile(np.abs(finite), 99)) or 1.0
        return -limit, limit
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def render_preview(matrix: np.ndarray, fs: Optional[float], title: str, destination: Path,
                   ylabel: str = "subcarrier index", *, quantity: str = "CSI amplitude",
                   colorbar: str = "amplitude (a.u.)", cmap: str = "viridis",
                   signed: bool = False, watermark: Optional[str] = None,
                   time_label: Optional[str] = None,
                   display_gamma: Optional[float] = None) -> Dict[str, int]:
    """Paper-style time/channel heatmap with dataset-aware semantics."""
    matrix, dims = normalize_preview_grid(matrix)
    rate = fs or 1.0
    time_unit = "s" if fs else "packets"
    duration = matrix.shape[1] / rate

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=110)
    extent = [0.0, duration, matrix.shape[0], 0.0]
    image_options: Dict[str, object] = {"aspect": "auto", "cmap": cmap, "extent": extent, "interpolation": "nearest"}
    vmin, vmax = preview_color_limits(matrix, signed=signed)
    if display_gamma is not None and not signed:
        image_options["norm"] = matplotlib.colors.PowerNorm(
            gamma=display_gamma, vmin=max(0.0, vmin), vmax=vmax,
        )
    else:
        image_options.update({"vmin": vmin, "vmax": vmax})
    image = ax.imshow(matrix, **image_options)
    ax.set_title(f"{title}: {quantity}", fontsize=11)
    ax.set_xlabel(time_label or f"time ({time_unit})")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label=colorbar)
    if watermark:
        ax.text(.5, .5, watermark, transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color="white", alpha=.38, weight="bold", rotation=-18)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return dims


def render_bvp_preview(
    tensor: np.ndarray, title: str, destination: Path, *, watermark: Optional[str] = None,
) -> Dict[str, int]:
    """Render a Widar BVP sequence as stacked body-velocity maps."""
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"BVP preview expects [time,Vx,Vy], got {tensor.shape}")
    time_bins, velocity_x_bins, velocity_y_bins = tensor.shape
    snapshot_indices = np.unique(
        np.linspace(0, time_bins - 1, min(5, time_bins)).round().astype(int)
    )
    vmin, vmax = preview_color_limits(tensor, signed=False)
    norm = matplotlib.colors.PowerNorm(gamma=.35, vmin=max(0.0, vmin), vmax=vmax)
    cmap = plt.get_cmap("magma")
    vx, vy = np.meshgrid(
        np.arange(velocity_x_bins), np.arange(velocity_y_bins), indexing="ij",
    )

    fig = plt.figure(figsize=(6.0, 3.6), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    for layer, time_index in enumerate(snapshot_indices):
        values = tensor[time_index]
        z = np.full_like(vx, float(layer), dtype=np.float64)
        ax.plot_surface(
            vx, vy, z, facecolors=cmap(norm(values)),
            rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
        )
        ax.plot(
            [0, velocity_x_bins - 1, velocity_x_bins - 1, 0, 0],
            [0, 0, velocity_y_bins - 1, velocity_y_bins - 1, 0],
            [layer] * 5, color="#475569", linewidth=.5,
        )
    ax.set_title(f"{title}: BVP velocity maps over time", fontsize=10, pad=4)
    ax.set_xlabel("x-velocity bin", labelpad=3)
    ax.set_ylabel("y-velocity bin", labelpad=3)
    ax.set_zticks(
        list(range(len(snapshot_indices))),
        [f"t{index + 1}" for index in snapshot_indices],
    )
    ax.set_zlabel("time snapshot", labelpad=3)
    ax.view_init(elev=40, azim=-58)
    ax.set_box_aspect((1.2, 1.0, .9))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor("#d4d4d8")
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(
        scalar, ax=ax, shrink=.57, pad=.07, aspect=18,
        label="BVP power (nonlinear display)",
    )
    if watermark:
        fig.text(.5, .50, watermark, ha="center", va="center", fontsize=18,
                 color="#334155", alpha=.28, weight="bold", rotation=-16)
    fig.subplots_adjust(left=.01, right=.90, bottom=.03, top=.91)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return {
        "native_time_steps": int(time_bins),
        "native_velocity_x_bins": int(velocity_x_bins),
        "native_velocity_y_bins": int(velocity_y_bins),
        "display_time_steps": int(len(snapshot_indices)),
        "display_velocity_x_bins": int(velocity_x_bins),
        "display_velocity_y_bins": int(velocity_y_bins),
    }


def _plot_spectrogram_stack(
    ax, grids: np.ndarray, rate: float, color_norm, color_map, layer_name: str, subtitle: str,
    layer_labels: Optional[List[str]] = None, *, time_unit: str = "s",
    ylabel: str = "subcarrier", layer_ticks: Optional[List[float]] = None,
) -> None:
    """Plot real spectrogram grids as offset planes on one 3D axis."""
    grids = np.asarray(grids, dtype=np.float64)
    if grids.ndim != 3:
        raise ValueError(f"stacked preview expects [layer,subcarrier,time], got {grids.shape}")
    layer_count, subcarriers, time_steps = grids.shape
    time_index = np.unique(np.linspace(0, time_steps - 1, min(time_steps, 120)).round().astype(int))
    subcarrier_index = np.unique(
        np.linspace(0, subcarriers - 1, min(subcarriers, 57)).round().astype(int)
    )
    time_values = time_index / rate
    x_grid, y_grid = np.meshgrid(time_values, subcarrier_index)
    for layer in range(layer_count):
        values = grids[layer][np.ix_(subcarrier_index, time_index)]
        z_grid = np.full_like(x_grid, float(layer))
        ax.plot_surface(
            x_grid, y_grid, z_grid,
            facecolors=color_map(color_norm(values)),
            rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
        )
        # A thin edge makes the three real heatmap layers legible at card size.
        ax.plot(
            [time_values[0], time_values[-1], time_values[-1], time_values[0], time_values[0]],
            [subcarrier_index[0], subcarrier_index[0], subcarrier_index[-1], subcarrier_index[-1], subcarrier_index[0]],
            [layer] * 5,
            color="#334155", linewidth=.55,
        )
    ax.set_title(subtitle, fontsize=9, pad=3)
    ax.set_xlabel(f"time ({time_unit})", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=4)
    labels = layer_labels or [f"{layer_name} {index + 1}" for index in range(layer_count)]
    ax.set_zticks(layer_ticks or list(range(layer_count)), labels)
    ax.set_xlim(0.0, time_steps / rate)
    ax.set_ylim(0, subcarriers - 1)
    ax.set_zlim(-.25, max(.45, layer_count - .55))
    ax.view_init(elev=48, azim=-58)
    ax.set_box_aspect((2.05, 1.05, 1.0 if layer_count > 4 else .78))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.pane.set_edgecolor("#d4d4d8")


def mimo_link_stacks(tensor: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Return every Tx-Rx plane in Tx-major and Rx-major display order."""
    time_steps, subcarriers, tx_count, rx_count = tensor.shape
    tx_major = tensor.transpose(2, 3, 1, 0).reshape(tx_count * rx_count, subcarriers, time_steps)
    rx_major = tensor.transpose(3, 2, 1, 0).reshape(rx_count * tx_count, subcarriers, time_steps)
    tx_labels = [f"T{tx + 1}R{rx + 1}" for tx in range(tx_count) for rx in range(rx_count)]
    rx_labels = [f"R{rx + 1}T{tx + 1}" for rx in range(rx_count) for tx in range(tx_count)]
    return tx_major, tx_labels, rx_major, rx_labels


def flattened_source_link_tensor(tensor: np.ndarray) -> np.ndarray:
    """Return canonical CSI as the source's un-factorized 1 x L link layout."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"source link preview expects [T,S,Tx,Rx], got {tensor.shape}")
    time_steps, subcarriers, tx_count, rx_count = tensor.shape
    return tensor.reshape(time_steps, subcarriers, 1, tx_count * rx_count)


def render_antenna_preview(
    tensor: np.ndarray, fs: Optional[float], title: str, destination: Path,
    *, signed: bool = False, watermark: Optional[str] = None,
    normalize_links_for_display: bool = False,
) -> Dict[str, int]:
    """Render [T,S,Tx,Rx] as a heatmap, one stack, or two side-by-side stacks."""
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 4:
        raise ValueError(f"antenna preview expects [T,S,Tx,Rx], got {tensor.shape}")
    time_steps, subcarriers, tx_count, rx_count = tensor.shape
    if tx_count == 1 and rx_count == 1:
        dims = render_preview(
            tensor[:, :, 0, 0].T, fs, title, destination,
            cmap="coolwarm" if signed else "viridis", signed=signed, watermark=watermark,
            quantity="processed feature" if signed else "CSI amplitude",
            colorbar="feature value" if signed else "amplitude (a.u.)",
        )
        return {**dims, "native_tx_links": 1, "native_rx_links": 1}

    if normalize_links_for_display and not signed:
        # Real antenna paths have different gains. Normalize each link only
        # for rendering; stored NPZ values and preprocessing stay untouched.
        display_tensor = np.empty_like(tensor)
        for tx_index in range(tx_count):
            for rx_index in range(rx_count):
                link = tensor[:, :, tx_index, rx_index]
                low, high = preview_color_limits(link, signed=False)
                display_tensor[:, :, tx_index, rx_index] = np.clip(
                    (link - low) / (high - low), 0.0, 1.0,
                )
        tensor = display_tensor
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = preview_color_limits(tensor, signed=signed)
    color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = plt.get_cmap("coolwarm" if signed else "viridis")
    rate = fs or 1.0
    time_unit = "s" if fs else "packets"
    stack_y = "feature channel" if signed else "subcarrier"

    # Preserve the existing compact card size. For MIMO, the two 3D axes are
    # deliberately placed in the same figure rather than enlarging the page.
    fig = plt.figure(figsize=(6.0, 3.6), dpi=110)
    if tx_count == 2 and rx_count >= 2:
        # Show each physical link once.  One subplot belongs to each Tx and
        # contains its Rx planes, so every physical link appears exactly once.
        left = fig.add_subplot(121, projection="3d")
        right = fig.add_subplot(122, projection="3d")
        for tx_index, ax in enumerate((left, right)):
            grids = tensor[:, :, tx_index, :].transpose(2, 1, 0)
            _plot_spectrogram_stack(
                ax, grids, rate, color_norm, color_map, "Rx",
                f"Tx {tx_index + 1} · {rx_count} Rx links",
                time_unit=time_unit, ylabel=stack_y,
            )
        fig.suptitle(f"{title}: {tx_count} Tx × {rx_count} Rx", fontsize=10, y=.98)
        colorbar_axes = [left, right]
        fig.subplots_adjust(left=.005, right=.90, bottom=.02, top=.90, wspace=-.02)
    elif tx_count > 1 and rx_count > 1:
        left = fig.add_subplot(121, projection="3d")
        right = fig.add_subplot(122, projection="3d")
        tx_major, tx_labels, rx_major, rx_labels = mimo_link_stacks(tensor)
        _plot_spectrogram_stack(
            left, tx_major, rate, color_norm, color_map, "link", f"All {tx_count * rx_count} links · grouped by Tx",
            [f"Tx {index + 1}" for index in range(tx_count)], time_unit=time_unit, ylabel=stack_y,
            layer_ticks=[index * rx_count + (rx_count - 1) / 2 for index in range(tx_count)],
        )
        _plot_spectrogram_stack(
            right, rx_major, rate, color_norm, color_map, "link", f"All {tx_count * rx_count} links · grouped by Rx",
            [f"Rx {index + 1}" for index in range(rx_count)], time_unit=time_unit, ylabel=stack_y,
            layer_ticks=[index * tx_count + (tx_count - 1) / 2 for index in range(rx_count)],
        )
        fig.suptitle(f"{title}: {'feature' if signed else 'antenna'} views", fontsize=10, y=.98)
        colorbar_axes = [left, right]
        fig.subplots_adjust(left=.005, right=.90, bottom=.02, top=.90, wspace=-.02)
    else:
        ax = fig.add_subplot(111, projection="3d")
        if rx_count > 1:
            grids, layer_name = tensor[:, :, 0, :].transpose(2, 1, 0), "Rx"
        else:
            grids, layer_name = tensor[:, :, :, 0].transpose(2, 1, 0), "Tx"
        _plot_spectrogram_stack(
            ax, grids, rate, color_norm, color_map, layer_name,
            f"{title}: stacked {layer_name} {'feature maps' if signed else 'spectrograms'}",
            time_unit=time_unit, ylabel=stack_y,
        )
        colorbar_axes = [ax]
        fig.subplots_adjust(left=.01, right=.91, bottom=.02, top=.92)
    scalar = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    scalar.set_array([])
    colorbar_label = (
        "feature value" if signed else
        "per-link normalized amplitude (display only)" if normalize_links_for_display else
        "amplitude (a.u.)"
    )
    fig.colorbar(scalar, ax=colorbar_axes, shrink=.55, pad=.05, aspect=18,
                 label=colorbar_label)
    if watermark:
        fig.text(.5, .50, watermark, ha="center", va="center", fontsize=18,
                 color="#334155", alpha=.28, weight="bold", rotation=-16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)
    return {
        "native_subcarriers": int(subcarriers),
        "native_time_steps": int(time_steps),
        "native_tx_links": int(tx_count),
        "native_rx_links": int(rx_count),
        "display_subcarriers": int(subcarriers),
        "display_time_steps": int(time_steps),
    }


def render_multi_rx_preview(
    grids: np.ndarray, fs: Optional[float], title: str, destination: Path,
) -> Dict[str, int]:
    """Backward-compatible wrapper for the original CSIDA preview helper."""
    grids = np.asarray(grids)
    tensor = grids.transpose(2, 1, 0)[:, :, None, :]
    return render_antenna_preview(tensor, fs, title, destination)


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
    # Skip derived views/ — the sample shows native standardized clips only.
    def _ignore(directory: str, names: List[str]) -> set:
        if Path(directory).resolve() == source.resolve():
            return {"views"}
        return set()
    shutil.copytree(source, destination, ignore=_ignore)
    return destination


def simplified_standardized_tree(standardized_root: Path) -> List[dict]:
    """Flat-folder schema: pattern + one real example pair; no fake file rows."""
    if not standardized_root.exists():
        return []
    npz_files = sorted(standardized_root.glob("*.npz"))
    if not npz_files:
        return []

    example = npz_files[0]
    example_json = example.with_suffix(".json")
    example_payload: Dict[str, object] = {
        "npz": example.name,
        "npz_bytes": example.stat().st_size,
    }
    if example_json.exists():
        example_payload["json"] = example_json.name
        example_payload["json_bytes"] = example_json.stat().st_size
    return [{
        "name": "standardized",
        "type": "dir",
        "children": [],
        "clip_count": len(npz_files),
        "layout": "flat",
        "pattern": "{label}__{…}__{setting}.npz + .json",
        "example": example_payload,
    }]


def _packaged_leaf_dirs(package_root: Path) -> List[Path]:
    """Leaf folders that contain a packaged sample (standardized.npz)."""
    if not package_root.exists():
        return []
    return [path.parent for path in sorted(package_root.rglob("standardized.npz"))]


def packaging_schema(
    package_root: Path,
    *,
    root_name: str,
    slot_name: str,
) -> Optional[dict]:
    """Structured by_label / by_setting layout for the website schema card."""
    leaves = _packaged_leaf_dirs(package_root)
    if not leaves:
        return None

    nested = any(len(leaf.relative_to(package_root).parts) >= 2 for leaf in leaves)
    path_pattern = (
        f"{root_name}/[task]/[{slot_name}]/"
        if nested
        else f"{root_name}/[{slot_name}]/"
    )
    example = leaves[0]
    rel = example.relative_to(package_root)
    example_files = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(example.iterdir())
        if path.is_file()
    ]
    return {
        "root": root_name,
        "slot": slot_name,
        "nested": nested,
        "pattern": path_pattern,
        "leaf_files": ["original.*", "standardized.npz", "metadata.json"],
        "folder_count": len(leaves),
        "example": {
            "path": rel.as_posix(),
            "files": example_files,
        },
    }


def simplified_packaging_tree(
    package_root: Path,
    *,
    slot_name: str,
    more_word: str,
) -> List[dict]:
    """Compact display tree for by_label / by_setting: pattern dirs + one example."""
    schema = packaging_schema(
        package_root,
        root_name="by_label" if more_word == "label" else "by_setting",
        slot_name=slot_name.strip("[]"),
    )
    if not schema:
        return []

    leaf_files = [
        {"name": name, "type": "file", "bytes": 0}
        for name in schema["leaf_files"]
    ]
    if schema["nested"]:
        pattern = {
            "name": "[task]",
            "type": "dir",
            "children": [{"name": slot_name, "type": "dir", "children": leaf_files}],
        }
    else:
        pattern = {"name": slot_name, "type": "dir", "children": leaf_files}

    example = schema["example"]
    example_files = [
        {"name": item["name"], "type": "file", "bytes": item["bytes"]}
        for item in example["files"]
    ]
    children: List[dict] = [
        pattern,
        {
            "name": example["path"],
            "type": "dir",
            "children": example_files,
            "role": "example",
        },
    ]
    remaining = schema["folder_count"] - 1
    if remaining > 0:
        children.append({
            "name": f"… +{remaining} more {more_word} folders",
            "type": "file",
            "bytes": 0,
            "role": "more",
        })
    return children


def split_sample_trees(sample_dir: Path, dataset_id: Optional[str] = None) -> Dict[str, object]:
    """Split into original release, prepare output, and by_label / by_setting use cases."""
    original_entries: List[dict] = []
    label_root: Optional[Path] = None
    setting_root: Optional[Path] = None
    if sample_dir.exists():
        for path in sorted(sample_dir.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if path.name == "by_label" and path.is_dir():
                label_root = path
                continue
            if path.name == "by_setting" and path.is_dir():
                setting_root = path
                continue
            if path.name == "standardized" and path.is_dir():
                continue
            if path.is_dir():
                original_entries.append({"name": path.name, "type": "dir", "children": file_tree(path)})
            else:
                original_entries.append({"name": path.name, "type": "file", "bytes": path.stat().st_size})

    data_standardized = DATA / (dataset_id or sample_dir.name) / "standardized"
    sample_standardized = sample_dir / "standardized"
    source_for_demo = sample_standardized if sample_standardized.exists() else data_standardized
    standardized_entries = simplified_standardized_tree(source_for_demo)
    native_clip_count = len(list(source_for_demo.glob("*.npz"))) if source_for_demo.exists() else 0
    standardized_meta = standardized_entries[0] if standardized_entries else {}

    usecase_entries = simplified_packaging_tree(label_root, slot_name="[label]", more_word="label") if label_root else []
    setting_entries = simplified_packaging_tree(setting_root, slot_name="[setting]", more_word="setting") if setting_root else []
    usecase_schema = packaging_schema(label_root, root_name="by_label", slot_name="label") if label_root else None
    setting_schema = packaging_schema(setting_root, root_name="by_setting", slot_name="setting") if setting_root else None
    label_count = len(_packaged_leaf_dirs(label_root)) if label_root else 0
    setting_count = len(_packaged_leaf_dirs(setting_root)) if setting_root else 0

    combined = list(original_entries)
    if standardized_entries:
        combined.extend(standardized_entries)
    if usecase_entries:
        combined.append({"name": "by_label", "type": "dir", "children": usecase_entries})
    if setting_entries:
        combined.append({"name": "by_setting", "type": "dir", "children": setting_entries})
    return {
        "file_tree": combined,
        "original_file_tree": original_entries,
        "standardized_file_tree": standardized_entries,
        "usecase_file_tree": usecase_entries,
        "setting_file_tree": setting_entries,
        "original_file_count": count_tree_files(original_entries),
        "standardized_file_count": native_clip_count,
        "usecase_file_count": label_count,
        "setting_file_count": setting_count,
        "standardized_pattern": standardized_meta.get("pattern"),
        "standardized_example": standardized_meta.get("example"),
        "usecase_schema": usecase_schema,
        "setting_schema": setting_schema,
    }

def slugify_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "unknown"


def split_catalog_key(value: str) -> Tuple[Optional[str], str]:
    """Split only the task separator, preserving slashes inside label text."""
    if "/" not in value:
        return None, value
    task, label = value.split("/", 1)
    return task.strip() or None, label.strip()


def label_relative_dir(label: str) -> Path:
    """Keep the task/label hierarchy without treating label punctuation as a path."""
    task, display_label = split_catalog_key(str(label))
    parts = [slugify_label(display_label)]
    if task:
        parts.insert(0, slugify_label(task))
    return Path(*parts) if all(parts) else Path("unknown")


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
    if dataset_id == "widar3":
        preferred = next((
            record for record in records
            if "1-Push&Pull" in resolve_record_path(record["source"]).parts
        ), None)
        if preferred:
            return preferred
    return records[0] if records else None


def resolve_record_path(record_path: str) -> Path:
    path = Path(record_path)
    return path if path.is_absolute() else ROOT / path


def clip_label_name(sidecar: dict, archive: Optional[dict] = None) -> Optional[str]:
    labels = sidecar.get("labels") or {}
    for key in ("activity", "class", "pattern", "bpm"):
        value = labels.get(key)
        if isinstance(value, (list, tuple, dict)):
            continue
        if value not in (None, ""):
            return str(value)
    if archive and "source_label" in archive:
        values = np.asarray(archive["source_label"]).reshape(-1)
        if values.size == 1:
            return str(values[0])
        if values.size > 1 and len({str(item) for item in values}) == 1:
            return str(values[0])
    return None


DISPLAY_LABELS = {
    "operanet": {
        "background": "background", "noactivity": "no activity",
        "liedown": "lie down", "standfromlie": "stand from floor",
        "bodyrotate": "body rotate",
    },
    "wifi-80mhz": {
        "W": "walking", "R": "running", "J": "jumping", "L": "sitting still",
        "S": "standing", "C": "sit down / stand up", "G": "arm exercises", "E": "empty room",
    },
    "wireless-har-wifi-uwb": {
        "liedown": "lie down", "pickup": "pick up",
        "sitrotate": "sit and rotate", "standrotate": "stand and rotate",
    },
}

WIAR_LABELS = {
    1: "horizontal arm wave", 2: "high arm wave", 3: "two hands wave",
    4: "high throw", 5: "draw x", 6: "draw tick", 7: "toss paper",
    8: "forward kick", 9: "side kick", 10: "bend", 11: "hand clap",
    12: "walk", 13: "phone call", 14: "drink water", 15: "sit down",
    16: "squat",
}

WIDAR_LABELS = [
    "Push&Pull", "Sweep", "Clap", "Slide", "Draw-N(H)", "Draw-O(H)",
    "Draw-Rectangle(H)", "Draw-Triangle(H)", "Draw-Zigzag(H)",
    "Draw-Zigzag(V)", "Draw-N(V)", "Draw-O(V)", "Draw-1", "Draw-2",
    "Draw-3", "Draw-4", "Draw-5", "Draw-6", "Draw-7", "Draw-8",
    "Draw-9", "Draw-10",
]


def clean_dataset_label(dataset_id: str, value: object, record: Optional[dict] = None) -> Optional[str]:
    """Return a short human label, rejecting padded/empty values."""
    label = str(value).strip()
    if not label or label.lower() in {"nan", "none", "null"}:
        return None
    label = DISPLAY_LABELS.get(dataset_id, {}).get(label, label)
    source = resolve_record_path(record["source"]) if record and record.get("source") else None
    if dataset_id == "xrf-v2" and label.isdigit() and source:
        label = source.stem
    elif dataset_id == "wifi-tad" and label.isdigit():
        label = f"class {label}"
    elif dataset_id == "widar3":
        label = re.sub(r"^gesture\s*(\d+)$", r"gesture \1", label, flags=re.IGNORECASE)
    return label.replace("_", " ").strip()


def inferred_record_label(dataset_id: str, record: dict) -> Optional[str]:
    source = resolve_record_path(record["source"]) if record.get("source") else None
    if not source:
        return None
    if dataset_id == "mm-fi":
        return next((part for part in reversed(source.parts) if re.fullmatch(r"A\d+", part)), None)
    if dataset_id == "wiar":
        match = re.fullmatch(r"csi_a(\d+)_\d+", source.stem, re.IGNORECASE)
        return WIAR_LABELS.get(int(match.group(1))) if match else None
    if dataset_id == "ntu-fi":
        return clean_dataset_label(dataset_id, source.parent.name, record)
    if dataset_id == "wireless-har-wifi-uwb":
        return clean_dataset_label(dataset_id, source.parent.name, record)
    if dataset_id in {"wimans", "xrf55", "xrf-v2"}:
        return clean_dataset_label(dataset_id, source.stem.split("_")[0], record)
    return None


def _csi_bench_coverage_catalogs(records: List[dict]) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    """Return (label_catalog, setting_catalog) from CSI-Bench coverage map."""
    coverage_path = DATA / "csi-bench" / "original" / "sample_coverage.json"
    if not coverage_path.exists():
        coverage_path = ROOT / "scripts" / "csi_bench_sample_coverage.json"
    if not coverage_path.exists():
        return {}, {}
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    by_source = {}
    for record in records:
        source = resolve_record_path(record["source"])
        try:
            rel = str(source.relative_to(DATA / "csi-bench" / "original")).replace("\\", "/")
        except ValueError:
            continue
        by_source[rel] = record
        # Website samples may be extracted with one extra csi-bench/ wrapper.
        # Index both forms so the official coverage map still resolves tasks.
        if rel.startswith("csi-bench/"):
            by_source[rel.removeprefix("csi-bench/")] = record

    def add_entry(catalog: Dict[str, List[dict]], key: str, remote_path: str) -> None:
        record = by_source.get(remote_path)
        if not record:
            return
        native_path = resolve_record_path(record["native_output"]) if record.get("native_output") else None
        if not native_path or not native_path.exists():
            native_path = resolve_record_path(record["output"])
        sidecar_path = native_path.with_suffix(".json")
        if not sidecar_path.exists():
            return
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        catalog.setdefault(key, []).append({
            "npz_path": native_path,
            "sidecar_path": sidecar_path,
            "sidecar": sidecar,
            "record": record,
        })

    labels: Dict[str, List[dict]] = {}
    settings: Dict[str, List[dict]] = {}
    for task, info in (payload.get("coverage") or {}).items():
        task_name = task.split("/")[-1]
        for name, remote_path in (info.get("labels") or {}).items():
            add_entry(labels, f"{task_name}/{name}", remote_path)
        for name, remote_path in (info.get("settings") or {}).items():
            add_entry(settings, f"{task_name}/{name}", remote_path)
    return labels, settings


def collect_label_entries(dataset_id: str) -> Dict[str, List[dict]]:
    records = converted_records(dataset_id)
    if dataset_id == "csi-bench":
        labels, _settings = _csi_bench_coverage_catalogs(records)
        if labels:
            return labels
    catalog: Dict[str, List[dict]] = {}
    for record in records:
        # Prefer native NPZ for label discovery so task-view windowing does not hide labels.
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
            structured_labels = sidecar.get("label_sets") or {}
            for task_name, values in structured_labels.items():
                if not isinstance(values, (list, tuple)):
                    values = [values]
                for raw_value in values:
                    display_label = clean_dataset_label(dataset_id, raw_value, record)
                    if not display_label:
                        continue
                    catalog.setdefault(f"{task_name}/{display_label}", []).append({
                        "npz_path": npz_path,
                        "sidecar_path": sidecar_path,
                        "sidecar": sidecar,
                        "record": record,
                    })
            # Some releases store many labeled clips in one tensor. Convert
            # those per-sample arrays into the same task/label catalog used by
            # clip-level and time-segment datasets.
            batch_specs = {
                "aril": {
                    "activity_label": ("Gesture", "activity", None),
                    "location_label": ("Location", "location", "location"),
                },
                "csida": {
                    "activity_label": ("Gesture", "activity", "class"),
                    "environment": ("Room", "environment", "room"),
                    "location": ("Position", "location", "position"),
                    "subject": ("Identity", "subject", "user"),
                },
                "ut-har": {
                    "activity_label": ("Activity", "activity", None),
                },
            }.get(dataset_id, {})
            axes = sidecar.get("axis_order") or []
            primary_name = sidecar.get("standard_representation") or "amplitude"
            primary_value = archive_dict.get(primary_name)
            if primary_value is None:
                primary_value = archive_dict.get("amplitude")
            sample_count = int(primary_value.shape[0]) if primary_value is not None and axes and axes[0] == "sample" else 0
            vocabularies = sidecar.get("labels") or {}
            for array_name, (task_name, vocabulary_key, fallback_prefix) in batch_specs.items():
                values = archive_dict.get(array_name)
                if values is None or sample_count <= 0:
                    continue
                values = np.asarray(values).reshape(-1)
                if values.size != sample_count:
                    continue
                vocabulary = vocabularies.get(vocabulary_key)
                for sample_index, raw_value in enumerate(values):
                    label_value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                    display_label: str
                    if isinstance(vocabulary, (list, tuple)):
                        try:
                            display_label = str(vocabulary[int(label_value)])
                        except (ValueError, TypeError, IndexError):
                            display_label = str(label_value)
                    elif isinstance(vocabulary, dict):
                        display_label = str(vocabulary.get(str(label_value), label_value))
                    elif fallback_prefix:
                        display_label = f"{fallback_prefix} {label_value}"
                    else:
                        display_label = str(label_value)
                    display_label = clean_dataset_label(dataset_id, display_label, record) or "unknown"
                    catalog.setdefault(f"{task_name}/{display_label}", []).append({
                        "npz_path": npz_path,
                        "sidecar_path": sidecar_path,
                        "sidecar": sidecar,
                        "record": record,
                        "sample_index": sample_index,
                    })
            clip_label = clean_dataset_label(dataset_id, clip_label_name(sidecar, archive_dict), record)
            if not clip_label:
                clip_label = inferred_record_label(dataset_id, record)
            if clip_label and not structured_labels and not (
                dataset_id == "xrf-v2" and sidecar.get("segments")
            ):
                clip_task = {
                    "exposing-csi": "Activity",
                }.get(dataset_id)
                clip_key = f"{clip_task}/{clip_label}" if clip_task else clip_label
                catalog.setdefault(clip_key, []).append({
                    "npz_path": npz_path,
                    "sidecar_path": sidecar_path,
                    "sidecar": sidecar,
                    "record": record,
                })
            for segment in sidecar.get("segments") or []:
                label = clean_dataset_label(
                    dataset_id, segment.get("label") or segment.get("source_label") or "", record,
                )
                if not label:
                    continue
                task = {
                    "figshare-csi-har": "Activity",
                    "operanet": "Activity",
                    "xrf-v2": "Activity",
                    "wifi-tad": "Activity",
                }.get(dataset_id)
                key = f"{task}/{label}" if task else label
                catalog.setdefault(key, []).append({
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
                    current = str(labels[0]).strip()
                    for index in range(1, labels.size + 1):
                        next_label = str(labels[index]).strip() if index < labels.size else None
                        if index == labels.size or next_label != current:
                            display = clean_dataset_label(dataset_id, current, record)
                            if display:
                                task = {
                                    "figshare-csi-har": "Activity",
                                    "operanet": "Activity",
                                    "xrf-v2": "Activity",
                                    "wifi-tad": "Activity",
                                }.get(dataset_id)
                                key = f"{task}/{display}" if task else display
                                catalog.setdefault(key, []).append({
                                    "npz_path": npz_path,
                                    "sidecar_path": sidecar_path,
                                    "sidecar": sidecar,
                                    "record": record,
                                    "segment": {
                                        "start_seconds": start / float(rate),
                                        "end_seconds": index / float(rate),
                                        "label": display,
                                    },
                                })
                            if index < labels.size:
                                start = index
                                current = next_label or ""
    return catalog


def collect_setting_entries(dataset_id: str) -> Dict[str, List[dict]]:
    """Collect experimental-setting buckets (e.g. CSI-Bench easy/medium/hard)."""
    if dataset_id != "csi-bench":
        return {}
    _labels, settings = _csi_bench_coverage_catalogs(converted_records(dataset_id))
    return settings


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

    def score(entry: dict) -> Tuple[int, int, float]:
        # Prefer native NPZ over task views so every activity remains available.
        native = 1 if "views" not in entry["npz_path"].parts else 0
        label_cardinality = 0
        if entry["sidecar"].get("dataset_id") == "wimans":
            label_cardinality = sum(
                len(values) if isinstance(values, (list, tuple)) else 1
                for values in (entry["sidecar"].get("label_sets") or {}).values()
            )
        duration = 0.0
        if entry.get("segment"):
            duration = float(entry["segment"]["end_seconds"]) - float(entry["segment"]["start_seconds"])
        return native, -label_cardinality, duration

    return max(pool, key=score)


def matrix_for_label_entry(entry: dict) -> Tuple[np.ndarray, Optional[float]]:
    matrix, fs = after_matrix(
        entry["npz_path"], entry["sidecar_path"], sample_index=entry.get("sample_index"),
    )
    segment = entry.get("segment")
    if not segment:
        return matrix, fs
    sidecar = entry["sidecar"]
    rate = sidecar.get("sample_rate_hz") or fs
    start, end = segment_time_indices(segment, matrix.shape[1], rate)
    if end > start:
        matrix = matrix[:, start:end]
    return matrix, rate or fs


def canonical_tensor_for_label_entry(entry: dict) -> Tuple[np.ndarray, Optional[float]]:
    tensor, fs = canonical_csi_tensor(
        entry["npz_path"], entry["sidecar_path"], sample_index=entry.get("sample_index", 0),
    )
    segment = entry.get("segment")
    if segment:
        rate = entry["sidecar"].get("sample_rate_hz") or fs
        start, end = segment_time_indices(segment, tensor.shape[0], rate)
        if end > start:
            tensor = tensor[start:end]
    return tensor, fs


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


def tensor_for_label_entry(entry: dict) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[float]]:
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
        for name in archive.files:
            if name == "source_label" or name.endswith("_label"):
                extras[name] = np.asarray(archive[name])
    sample_index = entry.get("sample_index")
    if sample_index is not None and axes and axes[0] == "sample":
        source_sample_count = tensor.shape[0]
        tensor = tensor[int(sample_index)]
        for name, value in list(extras.items()):
            if value.ndim >= 1 and value.shape[0] == source_sample_count:
                extras[name] = value[int(sample_index)]
        axes = axes[1:]
    time_axis = next((index for index, axis in enumerate(axes) if axis in {"time", "packet"}), 0)
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
    # Public per-label downloads are compact, contiguous real slices. Full
    # native and task-view recordings remain under data/<id>/standardized/.
    # Capping only time keeps every source subcarrier and link while making
    # the complete website release practical for GitHub Pages.
    if tensor.ndim and tensor.shape[time_axis] > HOSTED_LABEL_MAX_TIME_STEPS:
        slices = [slice(None)] * tensor.ndim
        slices[time_axis] = slice(0, HOSTED_LABEL_MAX_TIME_STEPS)
        tensor = tensor[tuple(slices)]
        for name, value in list(extras.items()):
            if value.ndim == 1 and value.shape[0] > HOSTED_LABEL_MAX_TIME_STEPS:
                extras[name] = value[:HOSTED_LABEL_MAX_TIME_STEPS]
    return tensor, extras, sidecar.get("sample_rate_hz")


def mirror_sample_zip(dataset_id: str) -> Dict[str, int]:
    sample_dir = SAMPLES / dataset_id
    catalog_path = ROOT / "catalog" / "datasets" / f"{dataset_id}.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    original = catalog.get("original") or {}
    package_files = sorted(
        path for package in ("by_label", "by_setting")
        for path in (sample_dir / package).rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "kind": "compact-real-sample",
        "description": (
            "Contiguous real signal windows selected from authentic released recordings. "
            "Values are unchanged; only the time span is capped for compact hosting."
        ),
        "canonical_axes": ["time", "subcarrier", "tx_link", "rx_link"],
        "max_time_steps": HOSTED_LABEL_MAX_TIME_STEPS,
        "source": original.get("download_page") or original.get("landing_page"),
        "license": original.get("license"),
        "redistribution": original.get("redistribution"),
        "files": len(package_files),
    }
    manifest_path = sample_dir / "sample-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    package_files.append(manifest_path)
    archive_path = SAMPLES / f"{dataset_id}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in package_files:
            handle.write(path, Path(dataset_id) / path.relative_to(sample_dir))
    total = sum(path.stat().st_size for path in package_files)
    return {
        "files": len(package_files),
        "bytes": total,
        "zip_bytes": archive_path.stat().st_size,
    }


def export_packaged_samples(
    dataset_id: str,
    catalog: Dict[str, List[dict]],
    package_name: str,
    *,
    key_field: str = "label",
) -> List[dict]:
    """Write one mini sample folder per catalog key under site/samples/<id>/<package_name>/."""
    package_root = SAMPLES / dataset_id / package_name
    if package_root.exists():
        shutil.rmtree(package_root)
    exported: List[dict] = []
    for key in catalog:
        rel = label_relative_dir(key)
        slug = "/".join(rel.parts)
        entry = choose_label_entry(catalog[key])
        dest = package_root / rel
        dest.mkdir(parents=True, exist_ok=True)
        files: List[str] = []
        segment = entry.get("segment")
        tensor, extras, rate = tensor_for_label_entry(entry)
        np.savez_compressed(dest / "standardized.npz", amplitude=tensor, **extras)
        files.append("standardized.npz")
        meta = {
            key_field: key,
            "slug": slug,
            "kind": "segment" if segment else "clip",
            "source_file": entry["sidecar"].get("source_file"),
            "sample_dir": f"samples/{dataset_id}/{package_name}/{slug}",
            "files": files,
            "tensor_shape": list(tensor.shape),
            "sample_rate_hz": rate,
            "compact_real_window": True,
            "max_time_steps": HOSTED_LABEL_MAX_TIME_STEPS,
            "values_unchanged": True,
            "segment": segment,
            "source_sample_index": entry.get("sample_index"),
        }
        (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        files.append("metadata.json")
        meta["files"] = files
        exported.append(meta)
    return exported


def export_label_data_samples(dataset_id: str, catalog: Dict[str, List[dict]]) -> List[dict]:
    """Write one mini sample folder per label under site/samples/<id>/by_label/."""
    return export_packaged_samples(dataset_id, catalog, "by_label", key_field="label")


def export_setting_data_samples(dataset_id: str, catalog: Dict[str, List[dict]]) -> List[dict]:
    """Write one mini sample folder per setting under site/samples/<id>/by_setting/."""
    return export_packaged_samples(dataset_id, catalog, "by_setting", key_field="setting")


def render_label_previews(dataset_id: str) -> Tuple[List[dict], Dict[str, str], Dict[str, List[dict]]]:
    catalog = collect_label_entries(dataset_id)
    previews: List[dict] = []
    errors: Dict[str, str] = {}
    preview_root = PREVIEWS / dataset_id
    if preview_root.exists():
        for stale in preview_root.glob("label_*.png"):
            stale.unlink()
    labels = list(catalog)
    if dataset_id == "wiar":
        order = {name: index for index, name in WIAR_LABELS.items()}
        labels.sort(key=lambda key: order.get(split_catalog_key(key)[1], 10_000))
    elif dataset_id == "widar3":
        order = {name: index for index, name in enumerate(WIDAR_LABELS)}
        labels.sort(key=lambda key: order.get(split_catalog_key(key)[1], 10_000))
    elif dataset_id == "wifi-presence-movement":
        # Show the states in one natural room-entry cycle instead of
        # alphabetical order, which makes the transition story harder to read.
        order = {
            name: index for index, name in enumerate(
                ["Gone", "Approach", "Enter", "Mobile", "Stationary", "Exit", "Departure"]
            )
        }
        labels.sort(key=lambda key: order.get(split_catalog_key(key)[1], 10_000))
    elif dataset_id == "wifi-tad":
        order = {
            name: index for index, name in enumerate(
                ["run", "walk", "jump", "wave", "bend", "stand", "sit"]
            )
        }
        labels.sort(key=lambda key: order.get(split_catalog_key(key)[1], 10_000))
    elif dataset_id == "wimans":
        task_order = {
            "Activity": 0, "Occupancy": 1, "Location": 2,
            "Environment": 3, "WiFi band": 4,
        }
        value_order = {
            name: index for index, name in enumerate([
                "nothing", "walk", "rotation", "jump", "wave", "lie down",
                "pick up", "sit down", "stand up",
                "0 people", "1 person", "2 people", "3 people", "4 people", "5 people",
                "a", "b", "c", "d", "e",
                "classroom", "meeting room", "empty room", "2.4 GHz", "5 GHz",
            ])
        }
        labels.sort(key=lambda key: (
            task_order.get(split_catalog_key(key)[0] or "", 10_000),
            value_order.get(split_catalog_key(key)[1], 10_000),
            split_catalog_key(key)[1],
        ))
    for label in labels:
        rel = label_relative_dir(label)
        slug = "__".join(rel.parts)
        destination = PREVIEWS / dataset_id / f"label_{slug}.png"
        try:
            entry = choose_label_entry(catalog[label])
            task, display_label = split_catalog_key(label)
            preview_title = f"{task}: {display_label}" if task else display_label
            watermark = "SYNTHETIC SHAPE DEMO" if (DATA / dataset_id / "original" / ".wisensehub-fixture.json").exists() else None
            try:
                if dataset_id == "widar3":
                    tensor = semantic_bvp_tensor(entry["npz_path"], entry["sidecar_path"])
                    dims = render_bvp_preview(tensor, preview_title, destination, watermark=watermark)
                else:
                    tensor, fs = canonical_tensor_for_label_entry(entry)
                    signed = bool(np.nanmin(tensor) < 0 and entry["sidecar"].get("source_representation") == "processed_amplitude")
                    display_fs = fs if dataset_id == "ntu-fi" else (fs or FALLBACK_RATE_HZ)
                    dims = render_antenna_preview(
                        tensor, display_fs, preview_title, destination,
                        signed=signed, watermark=watermark,
                        normalize_links_for_display=dataset_id == "wimans",
                    )
            except ValueError:
                try:
                    matrix, fs, spec = semantic_feature_matrix(entry["npz_path"], entry["sidecar_path"])
                    display_fs = None if spec.get("time_label") else (fs or FALLBACK_RATE_HZ)
                    dims = render_preview(
                        matrix, display_fs, preview_title, destination,
                        watermark=watermark, **spec,
                    )
                except ValueError:
                    matrix, fs = matrix_for_label_entry(entry)
                    dims = render_preview(matrix, fs or FALLBACK_RATE_HZ, preview_title, destination, watermark=watermark)
            previews.append({
                "catalog_key": label,
                "task": task,
                "label": display_label,
                "slug": slug,
                "image": versioned_site_asset(destination),
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
        "target_rate_assumed": sidecar.get("target_rate_assumed", False),
        "profile": sidecar.get("profile"),
        "view_options": sidecar.get("view_options"),
    }


def after_matrix(
    npz_path: Path, sidecar_path: Path, sample_index: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[float]]:
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
        index = 0 if sample_index is None else int(sample_index)
        if index < 0 or index >= tensor.shape[0]:
            raise IndexError(f"sample index {index} is outside tensor with {tensor.shape[0]} samples")
        tensor = tensor[index]
        axes = axes[1:]
    if len(axes) >= 3 and axes[0] in {"time", "packet"} and tensor.ndim == 3:
        return tensor.transpose(1, 2, 0).reshape(-1, tensor.shape[0]), sidecar.get("sample_rate_hz")
    if len(axes) >= 2 and axes[0] in {"time", "packet"} and tensor.ndim == 2:
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


def per_sample_dimensions(axes: List[str], shape: List[int]) -> List[dict]:
    """Describe one sample only; the website intentionally never displays N."""
    if axes and axes[0] == "sample":
        axes, shape = axes[1:], shape[1:]
    return [
        {"axis": axis, "size": int(size), "meaning": AXIS_MEANINGS.get(axis, axis)}
        for axis, size in zip(axes, shape)
    ]


def xrf_v2_axis_evidence(dimensions: List[dict]) -> List[dict]:
    """Use source-supported names for XRF V2's two released 3-valued axes."""
    meanings = {
        "tx_link": "Tx — logical source channel stream (not proven to be a physical Tx antenna)",
        "rx_link": "Rx — receiver-device stream",
    }
    return [{**item, "meaning": meanings.get(item["axis"], item["meaning"])} for item in dimensions]


def compact_shape(dimensions: List[dict]) -> str:
    labels = {"time": "T", "packet": "T", "subcarrier": "S", "link": "L", "tx_link": "Tx", "rx_link": "Rx"}
    return " × ".join(f"{labels.get(item['axis'], item['axis'])}={item['size']}" for item in dimensions)


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
        "tensor_shape": (
            f"T={dims['native_time_steps']} × S={dims['native_subcarriers']} × "
            f"Tx={dims.get('native_tx_links', 1)} × Rx={dims.get('native_rx_links', 1)}"
            if "native_tx_links" in dims
            else f"S={dims['native_subcarriers']} × T={dims['native_time_steps']}"
        ),
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
    tx_links = sizes.get("tx_link")
    rx_links = sizes.get("rx_link")
    rate = summary.get("sample_rate_hz")
    profile = summary.get("profile") or (summary.get("view_options") or {}).get("profile") or (catalog.get("standardization") or {}).get("profile")
    duration = None
    if rate and time_steps:
        seconds = time_steps / rate
        duration = f"{seconds:.3g} s ({time_steps} packets)"
    info: Dict[str, object] = {
        "task_profile": profile,
        "representation": summary.get("standard_representation") or "amplitude",
        "sampling_rate": (
            f"{rate:g} Hz (assumed target grid)" if rate and summary.get("target_rate_assumed")
            else f"{rate:g} Hz" if rate
            else "not defined for semantic time bins"
        ),
        "duration": duration,
        "frequency_band": (catalog.get("hardware") or {}).get("band"),
        "wifi_standard": (catalog.get("hardware") or {}).get("wifi_standard"),
        "setup_distance": SETUP_DISTANCE.get(dataset_id),
        "collection_setup": "Task-profile view: resample the full timeline, split it into consecutive windows, and pad only the final remainder",
        "tensor_shape": (
            f"{time_steps} time × {subcarriers} subcarriers × {tx_links or 1} Tx × {rx_links or 1} Rx"
            if tx_links is not None or rx_links is not None
            else f"{links or 1} link × {subcarriers} subcarriers × {time_steps} time steps"
            if subcarriers is not None
            else " × ".join(f"{item['size']} {item['axis']}" for item in dims)
        ),
        "view_grid": (
            f"{subcarriers} subcarriers × {time_steps} time steps @ {rate:g} Hz"
            if subcarriers and time_steps and rate else None
        ),
    }
    if native_summary:
        info["derived_from_shape"] = native_summary.get("shape")
    if dataset_id == "widar3":
        info["collection_setup"] = (
            "Reshape 400 flattened velocity bins into a 20 × 20 body-velocity map; "
            "keep all 22 semantic snapshots with no time resampling or padding"
        )
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*",
        help="Dataset IDs to rebuild (default: every dataset in the fetch report)",
    )
    args = parser.parse_args(argv)
    requested = set(args.datasets)
    fetch_report = {}
    report_path = SAMPLES / "fetch-report.json"
    if report_path.exists():
        fetch_report = json.loads(report_path.read_text(encoding="utf-8"))
    figure_sources = {}
    sources_path = FIGURES / "sources.json"
    if sources_path.exists():
        figure_sources = json.loads(sources_path.read_text(encoding="utf-8"))

    destination = SITE / "data" / "samples.json"
    payload: Dict[str, dict] = {}
    if requested and destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        payload.update(existing.get("datasets") or {})
    missing = requested - set(fetch_report)
    if missing:
        parser.error(f"unknown dataset ID(s): {', '.join(sorted(missing))}")
    for dataset_id, fetch_info in sorted(fetch_report.items()):
        if requested and dataset_id not in requested:
            continue
        if fetch_info.get("status") != "ok":
            skipped: Dict[str, object] = {
                "status": "skipped",
                "kind": fetch_info.get("kind", "real-sample-unavailable"),
                "reason": fetch_info.get("reason"),
                "source_url": fetch_info.get("source_url"),
                "license": fetch_info.get("license"),
                "redistribution": fetch_info.get("redistribution"),
            }
            figure_path = FIGURES / f"{dataset_id}.png"
            if figure_path.exists():
                skipped["setup_figure"] = f"assets/figures/{dataset_id}.png"
                skipped["figure_source"] = figure_sources.get(dataset_id)
            payload[dataset_id] = skipped
            continue
        sample_dir = SAMPLES / dataset_id
        tree_info = split_sample_trees(sample_dir, dataset_id)
        catalog_path = ROOT / "catalog" / "datasets" / f"{dataset_id}.json"
        catalog_entry = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
        original_entry = catalog_entry.get("original") or {}
        entry: Dict[str, object] = {
            "status": "ok",
            "kind": fetch_info.get("kind", "official-mini-sample"),
            "delivery": fetch_info.get("delivery", "hosted-sample"),
            "note": fetch_info.get("note"),
            "source_url": fetch_info.get("source_url") or original_entry.get("landing_page") or original_entry.get("download_page"),
            "license": fetch_info.get("license") or original_entry.get("license"),
            "redistribution": fetch_info.get("redistribution") or original_entry.get("redistribution"),
            "sample_zip": f"samples/{dataset_id}.zip" if fetch_info.get("delivery") != "official-fetch" else None,
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
        loader = RAW_LOADERS.get(dataset_id, raw_generic)
        raw_fs: Optional[float] = None
        ylabel = "subcarrier index"
        watermark = "SYNTHETIC SHAPE DEMO" if entry["kind"] == "generated-adapter-fixture" else None
        if source_path and source_path.exists():
            try:
                before_png = PREVIEWS / dataset_id / "before.png"
                native_path_for_preview = (
                    resolve_record_path(record["native_output"])
                    if record and record.get("native_output") else None
                )
                native_sidecar_for_preview = (
                    native_path_for_preview.with_suffix(".json") if native_path_for_preview else None
                )
                source_dimensions: List[dict] = []
                try:
                    if not native_path_for_preview or not native_sidecar_for_preview:
                        raise ValueError("native canonical output unavailable")
                    native_meta = json.loads(native_sidecar_for_preview.read_text(encoding="utf-8"))
                    if dataset_id == "widar3":
                        source_bvp = np.genfromtxt(source_path, delimiter=",")
                        if source_bvp.shape != (22, 400):
                            raise ValueError(f"expected Widar source [22,400], got {source_bvp.shape}")
                        matrix = source_bvp.T
                        before_dims = render_preview(
                            matrix, None, "Source CSV [T, 400]", before_png,
                            ylabel="flattened x/y velocity bin", quantity="BVP power",
                            colorbar="BVP power (nonlinear display)", cmap="magma",
                            time_label="time bin", display_gamma=.35,
                            watermark=watermark,
                        )
                        raw_fs = None
                        source_dimensions = [
                            {"axis": "time_bin", "size": 22, "meaning": AXIS_MEANINGS["time_bin"]},
                            {"axis": "flattened_velocity_bin", "size": 400, "meaning": AXIS_MEANINGS["flattened_velocity_bin"]},
                        ]
                    elif native_meta.get("canonical_tensor_exception"):
                        matrix, raw_fs, spec = semantic_feature_matrix(
                            native_path_for_preview, native_sidecar_for_preview,
                        )
                        before_dims = render_preview(
                            matrix, raw_fs, "Adapter-native", before_png,
                            watermark=watermark, **spec,
                        )
                    else:
                        tensor, raw_fs = canonical_csi_tensor(
                            native_path_for_preview, native_sidecar_for_preview,
                        )
                        if dataset_id == "nist-breathesmart":
                            # Before standardization, show the official 9 x 114
                            # reserved storage envelope unchanged. This makes
                            # the all-zero slots visibly disappear only in the
                            # standardized 2 Tx x 3 Rx x 56 result.
                            tensor = raw_nist_stored_tensor(source_path)
                        elif dataset_id == "ntu-fi":
                            # NTU-Fi's source is 2000 packets. The native
                            # standardized tensor has already applied the
                            # official every-fourth-packet selection, so load
                            # the MAT payload here to show a true before view.
                            tensor = raw_ntu_fi_tensor(source_path)
                            raw_fs = None
                        elif dataset_id == "figshare-csi-har":
                            tensor = flattened_source_link_tensor(tensor)
                        signed = bool(
                            np.nanmin(tensor) < 0
                            and native_meta.get("source_representation") == "processed_amplitude"
                        )
                        if dataset_id == "ut-har":
                            source_array = np.load(source_path, allow_pickle=False)
                            if isinstance(source_array, np.lib.npyio.NpzFile):
                                source_array = source_array[next(
                                    key for key in ("data", "x", "amplitude") if key in source_array.files
                                )]
                            source_array = np.asarray(source_array)
                            matrix = source_array[0].reshape(250, 90).T
                            before_dims = render_preview(
                                matrix, None, "Source [T, 90] processed features", before_png,
                                ylabel="flattened feature channel", cmap="coolwarm", signed=True,
                                quantity="signed processed feature", colorbar="feature value",
                                watermark=watermark,
                            )
                        else:
                            matrix = tensor[:, :, 0, 0].T
                            before_title = {
                                "figshare-csi-har": "Adapter-native 1 Tx × 4 flattened links",
                                "nist-breathesmart": "Source envelope 1 Tx × 9 stored link slots",
                                "ntu-fi": "Source CSIamp · 1 Tx × 3 antenna streams",
                                "operanet": "Source MATLAB CSI table · 3 Tx × 3 Rx",
                            }.get(dataset_id, "Adapter-native")
                            before_dims = render_antenna_preview(
                                tensor, raw_fs, before_title, before_png,
                                signed=signed, watermark=watermark,
                                normalize_links_for_display=dataset_id == "wimans",
                            )
                    source_axes = list(native_meta.get("source_axis_order") or native_meta.get("axis_order") or [])
                    source_shape = list(native_meta.get("source_shape") or native_meta.get("shape") or [])
                    source_dimensions = per_sample_dimensions(source_axes, source_shape)
                    if dataset_id == "xrf-v2":
                        source_dimensions = xrf_v2_axis_evidence(source_dimensions)
                    elif dataset_id == "xrf55":
                        source_dimensions = [{
                            "axis": "flattened_channel",
                            "size": 270,
                            "meaning": "C — 9 Rx streams × 30 subcarriers, stored flat",
                        }, {
                            "axis": "time",
                            "size": 1000,
                            "meaning": "T — source time steps",
                        }]
                    if dataset_id == "ut-har":
                        source_dimensions = [{
                            "axis": "time",
                            "size": 250,
                            "meaning": "T — source packet index",
                        }, {
                            "axis": "flattened_channel",
                            "size": 90,
                            "meaning": "C — 3 Rx streams × 30 subcarriers, stored flat",
                        }]
                except ValueError:
                    if not loader:
                        raise
                    matrix, raw_fs = loader(source_path)
                    before_dims = render_preview(
                        matrix, raw_fs, "Source file", before_png, ylabel,
                        watermark=watermark,
                    )
                    source_dimensions = [{
                        "axis": "subcarrier",
                        "size": before_dims["native_subcarriers"],
                        "meaning": "S — OFDM subcarrier / feature channel",
                    }, {
                        "axis": "time",
                        "size": before_dims["native_time_steps"],
                        "meaning": "T — time steps (packets)",
                    }]
                entry["preview_before_dims"] = before_dims
                entry["previews"]["before"] = versioned_site_asset(before_png)
                entry["preview_source_file"] = source_path.name
                source_profile = original_profile(dataset_id, matrix, raw_fs, source_path.name, before_dims)
                if source_dimensions:
                    source_profile["tensor_shape"] = compact_shape(source_dimensions)
                if dataset_id == "ut-har":
                    source_profile["tensor_shape"] = "T=250 × C=90"
                elif dataset_id == "wireless-har-wifi-uwb":
                    source_profile["tensor_shape"] = (
                        f"T={source_shape[0]} × C={source_shape[1]} (3 Rx × 30 S)"
                        if len(source_shape) == 2 else source_profile.get("tensor_shape")
                    )
                elif dataset_id == "xrf55":
                    source_profile["tensor_shape"] = "C=270 × T=1000 (9 Rx × 30 S)"
                elif dataset_id == "widar3":
                    source_profile["sampling_rate"] = "not defined for semantic time snapshots"
                    source_profile["duration"] = "22 semantic time snapshots"
                entry["original"] = {
                    "profile": source_profile,
                    "dimensions": source_dimensions,
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
                display_dimensions = dimension_table(summary)
                if dataset_id == "xrf-v2":
                    display_dimensions = xrf_v2_axis_evidence(display_dimensions)
                entry["dimensions"] = display_dimensions
                entry["standardized_view"] = {
                    "profile": standardized_profile(dataset_id, summary, native_summary),
                    "dimensions": display_dimensions,
                }
                entry["native_output"] = str(native_path.relative_to(ROOT)) if native_path else None
                try:
                    after_png = PREVIEWS / dataset_id / "after.png"
                    try:
                        view_meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
                        if dataset_id == "widar3":
                            tensor = semantic_bvp_tensor(npz_path, sidecar_path)
                            after_dims = render_bvp_preview(
                                tensor, "Standardized view", after_png, watermark=watermark,
                            )
                        elif view_meta.get("canonical_tensor_exception") or "feature" in (view_meta.get("axis_order") or []):
                            matrix, fs, spec = semantic_feature_matrix(npz_path, sidecar_path)
                            display_fs = None if spec.get("time_label") else (fs or FALLBACK_RATE_HZ)
                            after_dims = render_preview(
                                matrix, display_fs, "Standardized view", after_png,
                                watermark=watermark, **spec,
                            )
                        else:
                            tensor, fs = canonical_csi_tensor(npz_path, sidecar_path)
                            signed = bool(
                                np.nanmin(tensor) < 0
                                and view_meta.get("source_representation") == "processed_amplitude"
                            )
                            after_dims = render_antenna_preview(
                                tensor, fs or FALLBACK_RATE_HZ, "Standardized view", after_png,
                                signed=signed, watermark=watermark,
                                normalize_links_for_display=dataset_id == "wimans",
                            )
                    except ValueError:
                        matrix, fs = after_matrix(npz_path, sidecar_path)
                        after_dims = render_preview(
                            matrix, fs or FALLBACK_RATE_HZ, "Standardized view", after_png,
                            ylabel, watermark=watermark,
                        )
                    entry["preview_after_dims"] = after_dims
                    entry["previews"]["after"] = versioned_site_asset(after_png)
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
        setting_catalog = collect_setting_entries(dataset_id)
        official_fetch_only = entry.get("delivery") == "official-fetch"
        if official_fetch_only:
            for package_name in ("by_label", "by_setting"):
                package_root = SAMPLES / dataset_id / package_name
                if package_root.exists():
                    shutil.rmtree(package_root)
            sample_manifest = SAMPLES / dataset_id / "sample-manifest.json"
            if sample_manifest.exists():
                sample_manifest.unlink()
            hosted_zip = SAMPLES / f"{dataset_id}.zip"
            if hosted_zip.exists():
                hosted_zip.unlink()
        if label_catalog and not official_fetch_only:
            label_samples = export_label_data_samples(dataset_id, label_catalog)
            entry["label_samples"] = label_samples
            entry["label_coverage"] = {
                "expected": len(label_catalog),
                "exported": len(label_samples),
                "complete": len(label_samples) == len(label_catalog),
            }
            for preview in label_previews:
                sample = next((item for item in label_samples if item["label"] == preview["catalog_key"]), None)
                if sample:
                    preview["sample_dir"] = sample["sample_dir"]
                    preview["sample_files"] = sample["files"]
        elif label_catalog:
            entry["label_coverage"] = {
                "expected": len(label_catalog),
                "exported": 0,
                "previewed": len(label_previews),
                "complete": len(label_previews) == len(label_catalog),
                "delivery": "official-fetch",
            }
        if setting_catalog and not official_fetch_only:
            setting_samples = export_setting_data_samples(dataset_id, setting_catalog)
            entry["setting_samples"] = setting_samples
            entry["setting_coverage"] = {
                "expected": len(setting_catalog),
                "exported": len(setting_samples),
                "complete": len(setting_samples) == len(setting_catalog),
            }
        elif (SAMPLES / dataset_id / "by_setting").exists():
            shutil.rmtree(SAMPLES / dataset_id / "by_setting")
        if (label_catalog or setting_catalog) and not official_fetch_only:
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
            "tx_links": dims.get("tx_link"),
            "rx_links": dims.get("rx_link"),
            "sample_rate_hz": std.get("sample_rate_hz"),
            "axes": "y = subcarrier index, x = time",
            "note": "Native files are preserved. Task-profile views keep the full resampled timeline, split it into consecutive windows, and pad only the final remainder.",
        }
        payload[dataset_id] = entry

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": "1.0", "datasets": payload}, indent=2) + "\n",
                           encoding="utf-8")
    ready = sum(1 for item in payload.values() if item.get("status") == "ok")
    print(f"Wrote {destination.relative_to(ROOT)} ({ready} dataset samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
