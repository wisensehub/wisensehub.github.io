from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .registry import get_split_setting


GROUP_PATTERNS = {
    "subject": r"(?:subject|subj|person|user|participant|volunteer|p)[-_ ]?0*(\d+)",
    "environment": r"(?:environment|env|room|scene)[-_ ]?([a-z0-9]+)",
    "scenario": r"(?:scenario|los|nlos|through[-_ ]?wall)[-_ ]?([a-z0-9]+)?",
    "device": r"(?:device|receiver|rx|deployment)[-_ ]?([a-z0-9]+)|\b(bq|pifa)\b",
    "location": r"(?:location|loc|position|pos)[-_ ]?([a-z0-9]+)",
    "orientation": r"(?:orientation|orient|angle)[-_ ]?([a-z0-9]+)",
    "day": r"(?:day|date|session)[-_ ]?([a-z0-9]+)",
    "band": r"(?:band|freq)[-_ ]?([a-z0-9.]+)|\b(2\.4g|5g|6g)\b",
    "trial": r"(?:trial|repeat|repetition|rep|r)[-_ ]?0*(\d+)",
    "occupancy": r"(?:occupancy|people|persons|count)[-_ ]?0*(\d+)",
    "experiment": r"(?:experiment|exp)[-_ ]?([a-z0-9]+)",
}


def _natural(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value.lower()))


def infer_metadata(source: str) -> Dict[str, str]:
    text = source.replace("\\", "/").lower()
    metadata: Dict[str, str] = {}
    mmfi = re.search(r"(?:^|/)e0*(\d+)/s0*(\d+)/a0*(\d+)(?:/|$)", text)
    if mmfi:
        metadata.update(environment=mmfi.group(1), subject=mmfi.group(2), activity=mmfi.group(3))
    for key, pattern in GROUP_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = next((item for item in match.groups() if item), match.group(0))
            metadata[key] = value.lower()
    for split, pattern in {
        "train": r"(?:^|[/_.-])train(?:ing)?(?:[/_.-]|$)",
        "val": r"(?:^|[/_.-])val(?:idation)?(?:[/_.-]|$)",
        "test": r"(?:^|[/_.-])test(?:ing)?(?:[/_.-]|$)",
    }.items():
        if re.search(pattern, text):
            metadata["predefined_split"] = split
            break
    wipe_fall_folder = re.search(
        r"(?:^|/)(low|med|high)(_unseen)?(?:/|$)", text, flags=re.IGNORECASE,
    )
    if wipe_fall_folder:
        metadata["activity"] = {
            "low": "low", "med": "medium", "high": "high",
        }[wipe_fall_folder.group(1).lower()]
        metadata["predefined_split"] = "test" if wipe_fall_folder.group(2) else "train"
    if "los" in text and "nlos" not in text:
        metadata["scenario"] = "los"
    elif "nlos" in text or "through-wall" in text or "through_wall" in text:
        metadata["scenario"] = "nlos"
    if "bq" in text:
        metadata["device"] = "bq"
    elif "pifa" in text:
        metadata["device"] = "pifa"
    for release_group in ("lab+home276", "home276", "lab276", "lab150"):
        if release_group in text:
            metadata["release_group"] = release_group
            break
    return metadata


def _load_metadata(original: Path) -> Dict[str, Dict[str, str]]:
    """Load optional user metadata keyed by source_file or sample_id."""
    result: Dict[str, Dict[str, str]] = {}
    csv_path = original / "metadata.csv"
    jsonl_path = original / "metadata.jsonl"
    rows: Iterable[Mapping[str, Any]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        key = str(row.get("sample_id") or row.get("source_file") or "")
        if key:
            result[key.replace("\\", "/")] = {str(k): str(v) for k, v in row.items() if v not in (None, "")}
    return result


def build_inventory(dataset_root: Path, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    original = dataset_root / "original"
    overrides = _load_metadata(original)
    samples: List[Dict[str, Any]] = []
    for record in records:
        if record.get("status") not in {"converted", "skipped"} or not record.get("output"):
            continue
        output = Path(str(record["output"]))
        source = Path(str(record["source"]))
        try:
            source_rel = source.resolve().relative_to(original.resolve()).as_posix()
        except ValueError:
            source_rel = source.name
        sidecar_path = output.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
        shape, axes = sidecar.get("shape", []), sidecar.get("axis_order", [])
        count = int(shape[0]) if axes and axes[0] == "sample" and shape else 1
        array_metadata: Dict[str, np.ndarray] = {}
        if count > 1:
            with np.load(output, allow_pickle=False) as archive:
                for key in GROUP_PATTERNS:
                    if key in archive and np.asarray(archive[key]).reshape(-1).size == count:
                        array_metadata[key] = np.asarray(archive[key]).reshape(-1)
        inferred = infer_metadata(source_rel)
        inferred.update(overrides.get(source_rel, overrides.get(source.name, {})))
        output_rel = output.resolve().relative_to(dataset_root.resolve()).as_posix()
        for index in range(count):
            sample_id = f"{output_rel}::{index}" if count > 1 else output_rel
            metadata = dict(inferred)
            for key, values in array_metadata.items():
                value = values[index]
                metadata[key] = str(value.item() if hasattr(value, "item") else value)
            metadata.update(overrides.get(sample_id, {}))
            samples.append({"id": sample_id, "source_file": source_rel, "index": index if count > 1 else None, "metadata": metadata})
    return samples


def _ratios(values: Sequence[float] | None, fallback: Sequence[float] | None) -> List[float]:
    ratios = list(values or fallback or [0.7, 0.15, 0.15])
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must be three non-negative values summing to 1")
    return ratios


def _random_split(samples: Sequence[Dict[str, Any]], ratios: Sequence[float], seed: int) -> Dict[str, List[str]]:
    order = np.random.default_rng(seed).permutation(len(samples))
    exact = np.asarray(ratios) * len(samples)
    counts = np.floor(exact).astype(int)
    remainder = len(samples) - int(counts.sum())
    for index in np.argsort(-(exact - counts), kind="stable")[:remainder]:
        counts[index] += 1
    n_train, n_val = int(counts[0]), int(counts[1])
    cuts = (n_train, n_train + n_val)
    return {
        "train": [samples[i]["id"] for i in order[:cuts[0]]],
        "val": [samples[i]["id"] for i in order[cuts[0]:cuts[1]]],
        "test": [samples[i]["id"] for i in order[cuts[1]:]],
    }


def _predefined_split(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    parts = {"train": [], "val": [], "test": []}
    missing = []
    for sample in samples:
        split = sample["metadata"].get("predefined_split")
        if split in parts:
            parts[split].append(sample["id"])
        else:
            missing.append(sample["source_file"])
    if missing:
        examples = ", ".join(sorted(set(missing))[:3])
        raise ValueError(f"official split cannot be inferred for {examples}; keep train/val/test directories or add original/metadata.csv")
    return parts


def _group_split(
    samples: Sequence[Dict[str, Any]], group_by: str, ratios: Sequence[float], seed: int,
    holdout: Sequence[str] | None,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing = []
    for sample in samples:
        group = sample["metadata"].get(group_by)
        if group is None:
            missing.append(sample["source_file"])
        else:
            groups[str(group)].append(sample)
    if missing:
        examples = ", ".join(sorted(set(missing))[:3])
        raise ValueError(f"{group_by} metadata missing for {examples}; add original/metadata.csv with source_file,{group_by}")
    names = sorted(groups, key=_natural)
    if len(names) < 2:
        raise ValueError(f"cross-{group_by} requires at least two distinct groups, found {names}")
    requested = [str(value) for value in (holdout or [])]
    unknown = sorted(set(requested) - set(names), key=_natural)
    if unknown:
        raise ValueError(f"unknown holdout {group_by} values {unknown}; available: {names}")
    test_groups = requested or [names[-1]]
    remaining = [name for name in names if name not in test_groups]
    val_count = 0 if ratios[1] == 0 or len(remaining) < 2 else max(1, round(len(names) * ratios[1]))
    rng = np.random.default_rng(seed)
    shuffled = [remaining[i] for i in rng.permutation(len(remaining))]
    val_groups = shuffled[:val_count]
    train_groups = shuffled[val_count:]
    parts = {
        "train": [item["id"] for name in train_groups for item in groups[name]],
        "val": [item["id"] for name in val_groups for item in groups[name]],
        "test": [item["id"] for name in test_groups for item in groups[name]],
    }
    return parts, {"train": train_groups, "val": val_groups, "test": test_groups}


def _trial_order_split(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    parts = {"train": [], "val": [], "test": []}
    for sample in samples:
        trial = sample["metadata"].get("trial")
        if trial is None:
            raise ValueError("trial metadata missing; use trial/repetition in paths or add original/metadata.csv")
        parts["train" if int(trial) <= 14 else "test"].append(sample["id"])
    return parts


def _flatten_ids(value: Any) -> List[str]:
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_flatten_ids(item))
        return result
    if isinstance(value, dict):
        for key in ("ids", "samples", "files", "data"):
            if key in value:
                return _flatten_ids(value[key])
    return []


def _external_json_split(paths: Sequence[Path]) -> Dict[str, List[str]]:
    parts = {"train": [], "val": [], "test": []}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        used = False
        if isinstance(payload, dict):
            for target, aliases in {"train": ("train", "training"), "val": ("val", "valid", "validation"), "test": ("test", "testing")}.items():
                key = next((name for name in aliases if name in payload), None)
                if key:
                    parts[target].extend(_flatten_ids(payload[key]))
                    used = True
        if not used:
            inferred = infer_metadata(path.name).get("predefined_split")
            if inferred:
                parts[inferred].extend(_flatten_ids(payload))
    return parts


def generate_split(
    dataset_id: str, dataset_root: Path, records: Sequence[Mapping[str, Any]], setting_id: str | None = None,
    seed: int = 42, ratios: Sequence[float] | None = None, holdout: Sequence[str] | None = None,
) -> Dict[str, Any]:
    setting = get_split_setting(dataset_id, setting_id)
    samples = build_inventory(dataset_root, records)
    if not samples:
        raise ValueError("no standardized samples are available for splitting")
    selected_ratios = _ratios(ratios, setting.get("ratios"))
    kind = setting["kind"]
    held_out: Dict[str, List[str]] | None = None
    external_files: List[str] | None = None
    if kind == "random":
        parts = _random_split(samples, selected_ratios, seed)
    elif kind == "predefined":
        parts = _predefined_split(samples)
    elif kind == "group_holdout":
        parts, held_out = _group_split(samples, setting["group_by"], selected_ratios, seed, holdout)
    elif kind == "trial_order":
        parts = _trial_order_split(samples)
    elif kind == "external_json":
        paths = sorted((dataset_root / "original").glob(setting["pattern"]))
        if not paths:
            raise ValueError(f"official split file not found: original/{setting['pattern']}")
        external_files = [path.resolve().relative_to(dataset_root.resolve()).as_posix() for path in paths]
        parts = _external_json_split(paths)
    elif kind == "source_groups":
        group_by = setting["group_by"]
        groups: Dict[str, List[str]] = defaultdict(list)
        for sample in samples:
            value = sample["metadata"].get(group_by)
            if value is None:
                raise ValueError(f"{group_by} metadata missing for source group setting")
            groups[value].append(sample["id"])
        parts = {name: ids for name, ids in sorted(groups.items())}
    else:
        raise ValueError(f"unsupported split kind {kind}")
    payload: Dict[str, Any] = {
        "schema_version": "1.0", "dataset_id": dataset_id, "setting": setting["id"],
        "kind": kind, "provenance": setting["provenance"], "seed": seed,
        "ratios": selected_ratios, "sample_count": len(samples), "partitions": parts,
        "partition_counts": {name: len(items) for name, items in parts.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if setting.get("source"):
        payload["source"] = setting["source"]
    if held_out is not None:
        payload["groups"] = held_out
    if external_files is not None:
        payload["official_split_files"] = external_files
        payload["note"] = "Official JSON files are preserved verbatim; IDs are resolved by the upstream benchmark loader."
    destination = dataset_root / "splits" / f"{setting['id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["manifest"] = str(destination)
    return payload
