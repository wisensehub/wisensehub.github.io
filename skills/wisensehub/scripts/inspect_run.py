#!/usr/bin/env python3
"""Inspect a WiSenseHub prepare manifest and verify its referenced artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def resolve(repo: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    data_root = args.data_root if args.data_root.is_absolute() else repo / args.data_root
    dataset_root = data_root / args.dataset
    manifest_path = dataset_root / "prepare-manifest.json"
    if not manifest_path.is_file():
        print(json.dumps({"ok": False, "error": f"manifest not found: {manifest_path}"}, indent=2))
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    errors = Counter(
        str(record.get("error") or "unknown error")
        for record in records
        if record.get("status") == "failed"
    )
    referenced_outputs = [resolve(repo, record.get("output")) for record in records]
    missing_outputs = [str(path) for path in referenced_outputs if path and not path.is_file()]
    split = manifest.get("split") or {}
    split_path = resolve(repo, split.get("manifest"))
    split_members: list[str] = []
    if split_path and split_path.is_file():
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        for members in (split_payload.get("partitions") or {}).values():
            split_members.extend(str(member) for member in members)
    missing_split_files = []
    for member in split_members:
        relative_path, _, _window_index = member.partition("::")
        member_path = Path(relative_path)
        if not member_path.is_absolute():
            member_path = dataset_root / member_path
        if not member_path.is_file():
            missing_split_files.append(str(member_path))
    converted = int(manifest.get("converted") or 0)
    failed = int(manifest.get("failed") or 0)
    report = {
        "ok": converted > 0 and failed == 0 and not missing_outputs and not missing_split_files and bool(split_path and split_path.is_file()),
        "dataset_id": manifest.get("dataset_id"),
        "manifest": str(manifest_path),
        "source_count": manifest.get("source_count"),
        "converted": converted,
        "skipped": int(manifest.get("skipped") or 0),
        "failed": failed,
        "view_options": manifest.get("view_options"),
        "split": {
            "setting": split.get("setting"),
            "provenance": split.get("provenance"),
            "partition_counts": split.get("partition_counts"),
            "manifest": str(split_path) if split_path else None,
            "exists": bool(split_path and split_path.is_file()),
            "member_count": len(split_members),
            "missing_file_count": len(missing_split_files),
            "missing_files": missing_split_files[:10],
        },
        "standardized": str(dataset_root / "standardized"),
        "missing_output_count": len(missing_outputs),
        "missing_outputs": missing_outputs[:10],
        "failure_groups": dict(errors.most_common()),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
