#!/usr/bin/env python3
"""Download a small, structure-preserving subset of CSI-Bench from Kaggle.

The script downloads individual files instead of the complete dataset. By
default it covers all seven tasks, including one recording per label plus the
representative easy/medium/hard recordings listed in the repository coverage
map. Official relative paths are preserved under the output directory.

Setup:
    pip install kaggle
    # Configure Kaggle authentication if the dataset requires it.

Examples:
    python scripts/download_csi_bench_subset.py
    python scripts/download_csi_bench_subset.py --task breathing --task fall
    python scripts/download_csi_bench_subset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DATASET = "guozhenjennzhu/csi-bench"
COVERAGE_FILE = ROOT / "scripts" / "csi_bench_sample_coverage.json"
DEFAULT_OUTPUT = ROOT / "data" / "csi-bench" / "original"

TASKS = {
    "breathing": "BreathingDetection",
    "fall": "FallDetection",
    "localization": "Localization",
    "motion-source": "MotionSourceRecognition",
    "activity": "Multitask/HumanActivityRecognition",
    "identity": "Multitask/HumanIdentification",
    "proximity": "Multitask/ProximityRecognition",
}


def load_coverage() -> dict:
    if not COVERAGE_FILE.exists():
        raise FileNotFoundError(f"coverage map not found: {COVERAGE_FILE}")
    return json.loads(COVERAGE_FILE.read_text(encoding="utf-8"))


def selected_files(coverage: dict, task_slugs: Iterable[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    available = coverage.get("coverage") or {}
    for slug in task_slugs:
        task = TASKS[slug]
        task_info = available.get(task)
        if not task_info:
            raise ValueError(f"coverage map has no entry for {task}")
        files = list(dict.fromkeys(task_info.get("files") or []))
        metadata = f"{task}/metadata/label_mapping.json"
        result[task] = [metadata, *files]
    return result


def kaggle_executable() -> str:
    executable = shutil.which("kaggle")
    if not executable:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with: pip install kaggle"
        )
    return executable


def extract_payload(payload: Path, remote_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if payload.suffix.lower() != ".zip":
        shutil.move(str(payload), destination)
        return

    with zipfile.ZipFile(payload) as archive:
        safe_names = [
            name for name in archive.namelist()
            if not Path(name).is_absolute() and ".." not in Path(name).parts
        ]
        exact = next((name for name in safe_names if name == remote_path), None)
        by_name = [name for name in safe_names if Path(name).name == Path(remote_path).name]
        member = exact or (by_name[0] if len(by_name) == 1 else None)
        if member is None:
            raise RuntimeError(
                f"could not find {remote_path!r} inside {payload.name}; "
                f"archive contains {len(safe_names)} file(s)"
            )
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def download_file(kaggle: str, remote_path: str, output: Path, force: bool) -> str:
    destination = output / remote_path
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "skipped"

    with tempfile.TemporaryDirectory(prefix="csi-bench-") as scratch_name:
        scratch = Path(scratch_name)
        command = [
            kaggle,
            "datasets",
            "download",
            "-d",
            DATASET,
            "-f",
            remote_path,
            "-p",
            str(scratch),
            "--force",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Kaggle download failed for {remote_path}: {message}")
        payloads = [path for path in scratch.rglob("*") if path.is_file()]
        if len(payloads) != 1:
            raise RuntimeError(
                f"expected one downloaded payload for {remote_path}, found {len(payloads)}"
            )
        extract_payload(payloads[0], remote_path, destination)
    return "downloaded"


def write_manifest(output: Path, tasks: Dict[str, List[str]]) -> Path:
    files = [path for task_files in tasks.values() for path in task_files]
    manifest = {
        "schema_version": "1.0",
        "dataset": DATASET,
        "source": "https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Representative subset only; official relative paths are preserved.",
        "tasks": {task: task_files for task, task_files in tasks.items()},
        "file_count": len(files),
    }
    path = output / "subset-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(TASKS),
        help="Task to download; repeat for multiple tasks (default: all tasks).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--force", action="store_true", help="Download existing files again.")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading.")
    args = parser.parse_args()

    task_slugs = args.task or list(TASKS)
    coverage = load_coverage()
    tasks = selected_files(coverage, task_slugs)
    unique_files = list(dict.fromkeys(path for files in tasks.values() for path in files))

    print(f"CSI-Bench subset: {len(tasks)} task(s), {len(unique_files)} unique file(s)")
    for task, files in tasks.items():
        print(f"\n{task} ({len(files)} files)")
        for remote_path in files:
            print(f"  {remote_path}")

    if args.dry_run:
        return 0

    kaggle = kaggle_executable()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0}
    for index, remote_path in enumerate(unique_files, start=1):
        print(f"\n[{index}/{len(unique_files)}] {remote_path}")
        status = download_file(kaggle, remote_path, args.output, args.force)
        counts[status] += 1
        print(f"  {status}: {args.output / remote_path}")

    manifest = write_manifest(args.output, tasks)
    print(
        f"\nDone: {counts['downloaded']} downloaded, {counts['skipped']} skipped."
        f"\nManifest: {manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
