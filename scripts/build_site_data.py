#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wifi_datahub.catalog import load_tasks, validate_catalog  # noqa: E402


def main() -> int:
    datasets, errors = validate_catalog(ROOT)
    if errors:
        print("Catalog validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    tasks = load_tasks(ROOT)
    adapters = json.loads((ROOT / "catalog" / "adapters.json").read_text(encoding="utf-8"))["datasets"]
    splits = json.loads((ROOT / "catalog" / "splits.json").read_text(encoding="utf-8"))["datasets"]
    examples = json.loads((ROOT / "catalog" / "examples.json").read_text(encoding="utf-8"))["datasets"]
    sample_plans_path = ROOT / "catalog" / "sample-plans.json"
    sample_plans = json.loads(sample_plans_path.read_text(encoding="utf-8"))["datasets"] if sample_plans_path.exists() else {}
    collections_path = ROOT / "catalog" / "collections.json"
    collections = json.loads(collections_path.read_text(encoding="utf-8"))["datasets"] if collections_path.exists() else {}
    for dataset in datasets:
        dataset["conversion"] = adapters[dataset["id"]]
        dataset["split_settings"] = splits[dataset["id"]]
        dataset["conversion_example"] = examples[dataset["id"]]
        dataset["sample_plan"] = sample_plans.get(dataset["id"])
        if not dataset.get("collection") and dataset["id"] in collections:
            dataset["collection"] = collections[dataset["id"]]
    task_counts = Counter(task for dataset in datasets for task in dataset["tasks"])
    payload = {
        "generated_at": "2026-07-01",
        "datasets": datasets,
        "tasks": [{**item, "dataset_count": task_counts[item["id"]]} for item in tasks.values()],
        "stats": {
            "datasets": len(datasets),
            "tasks": len(tasks),
            "hardware_platforms": len({dataset["hardware"]["platform"] for dataset in datasets}),
            "open_or_direct": sum(dataset["original"]["access"] == "direct" for dataset in datasets),
        },
    }
    destination = ROOT / "site" / "data" / "catalog.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination.relative_to(ROOT)} with {len(datasets)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
