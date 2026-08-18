#!/usr/bin/env python3
"""Download and safely extract any hosted WiSenseHub catalog sample."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


SAMPLE_BASE = "https://wisensehub.github.io/samples"


def obtain_archive(source: str, temporary: Path) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        archive = temporary / "csi-bench-sample.zip"
        request = urllib.request.Request(source, headers={"User-Agent": "WiSenseHub-skill/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        return archive
    archive = Path(source).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"sample archive not found: {archive}")
    return archive


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe ZIP member: {info.filename}")
        if not info.is_dir():
            members.append(info)
    return members


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="WiSenseHub repository root")
    parser.add_argument("--dataset", default="csi-bench", help="Catalog dataset ID")
    parser.add_argument("--source", help="Sample URL or local ZIP for testing")
    parser.add_argument("--force", action="store_true", help="Replace existing sample files")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if not (repo / "pyproject.toml").is_file() or not (repo / "catalog").is_dir():
        raise SystemExit(f"Not a WiSenseHub repository: {repo}")

    catalog_path = repo / "catalog" / "datasets" / f"{args.dataset}.json"
    if not catalog_path.is_file():
        raise SystemExit(f"Unknown catalog dataset: {args.dataset}")
    source_value = args.source or (
        f"{SAMPLE_BASE}/csi-bench-original-subset.zip"
        if args.dataset == "csi-bench"
        else f"{SAMPLE_BASE}/{args.dataset}.zip"
    )
    destination = repo / "data" / args.dataset / "original"
    sample_kind = "catalog sample"
    plans_path = repo / "catalog" / "sample-plans.json"
    if plans_path.is_file():
        plans = json.loads(plans_path.read_text(encoding="utf-8")).get("datasets", {})
        plan = plans.get(args.dataset) or {}
        sample_kind = plan.get("sample_kind", sample_kind)
        if sample_kind != "official-mini-sample" and not args.source:
            official_source = plan.get("official_source") or json.loads(
                catalog_path.read_text(encoding="utf-8")
            ).get("original", {}).get("download_page")
            raise SystemExit(
                "No authentic hosted mini-sample is available for "
                f"{args.dataset}. Use the official release: {official_source or 'see the dataset catalog'}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0
    tasks: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="wisensehub-sample-") as temporary_name:
        archive_path = obtain_archive(source_value, Path(temporary_name))
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_members(archive)
            for info in members:
                relative = Path(info.filename)
                parts = relative.parts
                if parts and parts[0] == args.dataset:
                    relative = Path(*parts[1:])
                if not relative.parts:
                    continue
                if relative.parts[0] in {"standardized", "by_label", "by_setting"}:
                    continue
                if len(relative.parts) > 1 and not relative.name.startswith("."):
                    tasks.add(relative.parts[0])
                target = destination / relative
                if target.exists() and not args.force:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source_file, target.open("wb") as output:
                    shutil.copyfileobj(source_file, output, length=1024 * 1024)
                downloaded += 1

    result = {
        "ok": True,
        "dataset": args.dataset,
        "kind": sample_kind,
        "source": source_value,
        "destination": str(destination),
        "tasks": sorted(tasks),
        "downloaded_files": downloaded,
        "existing_files": skipped,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
