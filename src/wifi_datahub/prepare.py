from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from .adapters.aril import convert_aril_mat
from .adapters.generic import convert_generic
from .adapters.profile_registry import DATASET_ADAPTERS, convert_dataset
from .adapters.intel5300 import convert_wiar_dat
from .adapters.ut_har import convert_ut_har_npz
from .adapters.wallhack import convert_wallhack_csv
from .adapters.xrf_v2 import convert_xrf_v2_h5
from .quality import write_quality_report
from .registry import load_adapter_registry
from .splits import generate_split
from .views import ViewOptions, resolve_task_profile, create_standard_view
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
            profile=base.profile,
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
    if handler == "aril":
        name = source.name.lower()
        split = "train" if "train" in name else "test" if "test" in name else None
        if split is None:
            raise ValueError("ARIL filename must contain train or test")
        return convert_aril_mat(source, output, split)
    if handler == "xrf-v2":
        return convert_xrf_v2_h5(source, output)
    if handler == "wallhack18k":
        return convert_wallhack_csv(source, output)
    if handler == "ut-har":
        return convert_ut_har_npz(source, output)
    if handler == "generic":
        return convert_generic(source, output, dataset_id)
    if handler in DATASET_ADAPTERS:
        return convert_dataset(handler, source, output)
    if handler == "official-profile":
        return convert_dataset(dataset_id, source, output)
    if handler == "wiar-intel5300":
        return convert_wiar_dat(source, output)
    raise ValueError(f"unknown adapter handler {handler!r} for {dataset_id}")


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
    for source in sources:
        stem = _safe_stem(source, original)
        native_output = standardized / f"{stem}.npz"
        view_output = view_root / f"{stem}.npz"
        output = view_output
        report = reports / "views" / f"{stem}.quality.json"
        native_report = reports / f"{stem}.quality.json"
        if output.exists() and native_output.exists() and not force:
            records.append(ConversionRecord(
                str(source), str(output), str(report) if report.exists() else None, "skipped",
                native_output=str(native_output),
            ))
            continue
        try:
            if not native_output.exists() or force:
                _convert(dataset_id, source, native_output)
                write_quality_report(native_output, native_report)
            if not output.exists() or force:
                create_standard_view(native_output, output, effective_view)
                write_quality_report(output, report)
            records.append(ConversionRecord(
                str(source), str(output), str(report), "converted",
                native_output=str(native_output), view_report=str(report),
            ))
        except Exception as exc:
            records.append(ConversionRecord(str(source), None, None, "failed", f"{type(exc).__name__}: {exc}"))
    summary = {
        "schema_version": "1.0", "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "directories": {"original": str(original), "standardized": str(standardized), "reports": str(reports)},
        "view_options": asdict(effective_view) if effective_view.requested() else None,
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
