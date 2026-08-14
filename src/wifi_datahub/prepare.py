from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters.aril import convert_aril_mat
from .adapters.canonical import canonicalize_csi_output
from .adapters.generic import convert_generic
from .adapters.profile_registry import DATASET_ADAPTERS, convert_dataset
from .adapters.intel5300 import convert_wiar_dat
from .adapters.ut_har import convert_ut_har_npz
from .adapters.wallhack import convert_wallhack_csv
from .adapters.xrf_v2 import convert_xrf_v2_h5
from .quality import write_quality_report
from .registry import load_adapter_registry
from .splits import generate_split
from .views import ViewOptions, resolve_task_profile, create_standard_view, view_source_info
from .catalog import load_datasets


def _catalog_view_options(dataset_id: str) -> ViewOptions:
    for entry in load_datasets():
        if entry.get("id") == dataset_id:
            return resolve_task_profile((entry.get("standardization") or {}).get("profile"))
    return resolve_task_profile(None)


def _resolve_prepare_view(dataset_id: str, view_options: Optional[ViewOptions]) -> ViewOptions:
    """Prefer explicit CLI view options; otherwise use the dataset's task profile."""
    if view_options is None or not view_options.requested():
        return _catalog_view_options(dataset_id)
    if (
        dataset_id == "csi-bench"
        and view_options.profile is None
        and view_options.target_rate_hz is None
        and view_options.duration_s is None
        and view_options.target_length is None
    ):
        # Dimension-only requests retain CSI-Bench's automatic task time profile.
        base = _catalog_view_options(dataset_id)
        return replace(
            base,
            interpolation=view_options.interpolation,
            layout=view_options.layout,
            links=view_options.links,
            subcarriers=view_options.subcarriers,
            tx_links=view_options.tx_links,
            rx_links=view_options.rx_links,
            window_mode=view_options.window_mode,
        )
    if (
        view_options.profile
        and view_options.target_rate_hz is None
        and view_options.duration_s is None
        and view_options.target_length is None
    ):
        base = resolve_task_profile(view_options.profile)
        return ViewOptions(
            target_rate_hz=base.target_rate_hz,
            duration_s=base.duration_s,
            target_length=base.target_length,
            interpolation=view_options.interpolation,
            layout=view_options.layout,
            links=view_options.links,
            subcarriers=view_options.subcarriers,
            tx_links=view_options.tx_links,
            rx_links=view_options.rx_links,
            profile=base.profile,
            window_mode=view_options.window_mode,
        )
    return view_options


@dataclass
class ConversionRecord:
    source: str
    output: Optional[str]
    report: Optional[str]
    status: str
    error: Optional[str] = None
    native_output: Optional[str] = None
    view_report: Optional[str] = None
    task: Optional[str] = None
    profile: Optional[str] = None
    view_options: Optional[dict] = None


CSI_BENCH_TASK_PROFILES = {
    "BreathingDetection": "vital-sign",
    "FallDetection": "general-sensing",
    "Localization": "general-sensing",
    "MotionSourceRecognition": "general-sensing",
    "HumanActivityRecognition": "general-sensing",
    "HumanIdentification": "general-sensing",
    "ProximityRecognition": "general-sensing",
    "Multitask": "general-sensing",
}


def _csi_bench_task(source: Path, native_output: Path) -> str:
    try:
        metadata = json.loads(native_output.with_suffix(".json").read_text(encoding="utf-8"))
        task = (metadata.get("labels") or {}).get("task")
        if task in CSI_BENCH_TASK_PROFILES:
            return str(task)
    except (OSError, json.JSONDecodeError):
        pass
    source_text = source.as_posix()
    return next((task for task in CSI_BENCH_TASK_PROFILES if task in source_text), "Multitask")


def _csi_bench_task_view(task: str, requested: ViewOptions) -> ViewOptions:
    base = resolve_task_profile(CSI_BENCH_TASK_PROFILES.get(task, "general-sensing"))
    return ViewOptions(
        target_rate_hz=requested.target_rate_hz if requested.target_rate_hz is not None else base.target_rate_hz,
        duration_s=requested.duration_s if requested.duration_s is not None else base.duration_s,
        target_length=requested.target_length,
        interpolation=requested.interpolation,
        layout=requested.layout,
        links=requested.links,
        subcarriers=requested.subcarriers,
        tx_links=requested.tx_links,
        rx_links=requested.rx_links,
        profile=base.profile,
        window_mode=requested.window_mode,
    )


def _view_matches(path: Path, options: ViewOptions, native_path: Path) -> bool:
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists() or path.stat().st_mtime < native_path.stat().st_mtime:
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("view_options") == asdict(options) and metadata.get("full_source_preserved") is True


def _safe_stem(path: Path, original_root: Path) -> str:
    relative = path.resolve().relative_to(original_root.resolve())
    if path.is_file():
        relative = relative.with_suffix("")
    return "__".join(relative.parts).replace(" ", "_")


def _discover(dataset_id: str, original: Path) -> List[Path]:
    registry = load_adapter_registry()
    if dataset_id not in registry:
        raise ValueError(f"no prepare adapter registered for {dataset_id}")
    found: List[Path] = []
    for pattern in registry[dataset_id]["patterns"]:
        found.extend(original.glob(pattern))
    accept_directories = registry[dataset_id].get("accept_directories", False)
    unique = sorted({
        path.resolve() for path in found
        if path.is_file() or path.name.endswith(".zarr") or (accept_directories and path.is_dir())
    })
    if dataset_id == "wallhack18k":
        unique = [path for path in unique if "data" in path.read_text(encoding="utf-8", errors="ignore")[:2048]]
    return unique


def registered_datasets() -> List[str]:
    return sorted(load_adapter_registry())


def _convert(dataset_id: str, source: Path, output: Path) -> Path:
    handler = load_adapter_registry()[dataset_id]["handler"]
    result: Path
    if handler == "aril":
        name = source.name.lower()
        split = "train" if "train" in name else "test" if "test" in name else None
        if split is None:
            raise ValueError("ARIL filename must contain train or test")
        result = convert_aril_mat(source, output, split)
    elif handler == "xrf-v2":
        result = convert_xrf_v2_h5(source, output)
    elif handler == "wallhack18k":
        result = convert_wallhack_csv(source, output)
    elif handler == "ut-har":
        result = convert_ut_har_npz(source, output)
    elif handler == "generic":
        result = convert_generic(source, output, dataset_id)
    elif handler in DATASET_ADAPTERS:
        result = convert_dataset(handler, source, output)
    elif handler == "official-profile":
        result = convert_dataset(dataset_id, source, output)
    elif handler == "wiar-intel5300":
        result = convert_wiar_dat(source, output)
    else:
        raise ValueError(f"unknown adapter handler {handler!r} for {dataset_id}")
    canonicalize_csi_output(dataset_id, output)
    return result


def prepare_dataset(
    dataset_id: str, data_root: Path, limit: Optional[int] = None, force: bool = False,
    setting: Optional[str] = None, seed: int = 42, ratios: Optional[Sequence[float]] = None,
    holdout: Optional[Sequence[str]] = None,
    view_options: Optional[ViewOptions] = None,
) -> dict:
    if dataset_id not in registered_datasets():
        raise ValueError(f"unsupported dataset {dataset_id}; supported: {', '.join(registered_datasets())}")
    dataset_root = data_root / dataset_id
    original = dataset_root / "original"
    standardized = dataset_root / "standardized"
    view_root = standardized / "views"
    reports = dataset_root / "reports"
    standardized.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    sources = _discover(dataset_id, original)
    if limit is not None:
        sources = sources[:limit]
    records: List[ConversionRecord] = []
    effective_view = _resolve_prepare_view(dataset_id, view_options)
    native_items: list[tuple[Path, str, Path, Path, bool]] = []
    for source in sources:
        stem = _safe_stem(source, original)
        native_output = standardized / f"{stem}.npz"
        native_report = reports / f"{stem}.quality.json"
        try:
            native_changed = False
            if not native_output.exists() or force:
                _convert(dataset_id, source, native_output)
                write_quality_report(native_output, native_report)
                native_changed = True
            native_items.append((source, stem, native_output, native_report, native_changed))
        except Exception as exc:
            records.append(ConversionRecord(str(source), None, None, "failed", f"{type(exc).__name__}: {exc}"))

    explicit_window = bool(view_options and (view_options.target_length is not None or view_options.duration_s is not None))
    window_policy: dict[str, object] | None = None
    item_views: dict[str, ViewOptions] = {item[1]: effective_view for item in native_items}
    item_tasks: dict[str, str] = {}
    task_profiles: dict[str, dict[str, object]] = {}
    if native_items and not records and dataset_id == "csi-bench" and effective_view.profile == "task-auto":
        grouped: dict[str, list[tuple[Path, str, Path, Path, bool]]] = {}
        for item in native_items:
            task = _csi_bench_task(item[0], item[2])
            item_tasks[item[1]] = task
            grouped.setdefault(task, []).append(item)
        for task, task_items in grouped.items():
            task_view = _csi_bench_task_view(task, effective_view)
            source_infos = [view_source_info(item[2], task_view) for item in task_items]
            task_min_length = min(int(info["rate_length"]) for info in source_infos)
            profile_limit = task_view.target_length
            if profile_limit is None and task_view.duration_s is not None and task_view.target_rate_hz is not None:
                profile_limit = max(1, int(round(float(task_view.duration_s) * float(task_view.target_rate_hz))))
            selected_length = (
                int(task_view.target_length) if explicit_window and task_view.target_length is not None
                else min(task_min_length, int(profile_limit)) if profile_limit
                else task_min_length
            )
            selected_view = replace(task_view, target_length=selected_length, duration_s=None)
            for item in task_items:
                item_views[item[1]] = selected_view
            task_profiles[task] = {
                **asdict(selected_view),
                "task_min_resampled_length": task_min_length,
                "profile_max_length": profile_limit,
                "selected_window_length": selected_length,
                "source_count": len(task_items),
            }
        window_policy = {
            "mode": "task-min-full-coverage",
            "task_profiles": task_profiles,
            "full_source_preserved": True,
            "remainder": "pad final window and mark padding false in valid_mask",
        }
    elif native_items and not records and not explicit_window:
        source_infos = [view_source_info(item[2], effective_view) for item in native_items]
        dataset_min_length = min(int(info["rate_length"]) for info in source_infos)
        profile_limit = effective_view.target_length
        if profile_limit is None and effective_view.duration_s is not None:
            rate = effective_view.target_rate_hz
            if rate is not None:
                profile_limit = max(1, int(round(float(effective_view.duration_s) * float(rate))))
        selected_length = min(dataset_min_length, int(profile_limit)) if profile_limit else dataset_min_length
        effective_view = replace(effective_view, target_length=selected_length, duration_s=None)
        item_views = {item[1]: effective_view for item in native_items}
        window_policy = {
            "mode": "dataset-min-full-coverage",
            "dataset_min_resampled_length": dataset_min_length,
            "profile_max_length": profile_limit,
            "selected_window_length": selected_length,
            "full_source_preserved": True,
            "remainder": "pad final window and mark padding false in valid_mask",
        }
    elif native_items:
        item_views = {item[1]: effective_view for item in native_items}
        window_policy = {
            "mode": "explicit-full-coverage",
            "selected_window_length": effective_view.target_length,
            "duration_s": effective_view.duration_s,
            "full_source_preserved": True,
            "remainder": "pad final window and mark padding false in valid_mask",
        }

    for source, stem, native_output, native_report, native_changed in native_items:
        item_view = item_views[stem]
        view_output = view_root / f"{stem}.npz"
        report = reports / "views" / f"{stem}.quality.json"
        try:
            view_current = not force and _view_matches(view_output, item_view, native_output)
            if not view_current:
                create_standard_view(native_output, view_output, item_view)
                write_quality_report(view_output, report)
            records.append(ConversionRecord(
                str(source), str(view_output), str(report), "converted" if native_changed or not view_current else "skipped",
                native_output=str(native_output), view_report=str(report),
                task=item_tasks.get(stem), profile=item_view.profile, view_options=asdict(item_view),
            ))
        except Exception as exc:
            records.append(ConversionRecord(str(source), None, None, "failed", f"{type(exc).__name__}: {exc}"))
    summary = {
        "schema_version": "1.0", "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "directories": {"original": str(original), "standardized": str(standardized), "reports": str(reports)},
        "view_options": asdict(effective_view) if effective_view.requested() else None,
        "task_profiles": task_profiles or None,
        "window_policy": window_policy,
        "source_count": len(sources),
        "converted": sum(record.status == "converted" for record in records),
        "skipped": sum(record.status == "skipped" for record in records),
        "failed": sum(record.status == "failed" for record in records),
        "records": [asdict(record) for record in records],
    }
    manifest = dataset_root / "prepare-manifest.json"
    if not sources:
        manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise FileNotFoundError(f"no recognized source files found in {original}; manifest written to {manifest}")
    if summary["failed"]:
        manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{summary['failed']} conversion(s) failed; inspect {manifest}")
    try:
        split = generate_split(dataset_id, dataset_root, summary["records"], setting, seed, ratios, holdout)
        summary["split"] = {
            "setting": split["setting"], "provenance": split["provenance"],
            "partition_counts": split["partition_counts"], "manifest": split["manifest"],
        }
    except ValueError as exc:
        summary["split_error"] = str(exc)
        manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise ValueError(f"conversion succeeded, but split generation failed: {exc}") from exc
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
