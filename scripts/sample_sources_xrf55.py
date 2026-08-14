#!/usr/bin/env python3
"""Acquire one authentic XRF55 WiFi clip for each of the 55 actions.

The official Kaggle release is public, so the files can be fetched one at a
time without downloading the roughly 97 GB multimodal collection. The
authors' official project page licenses XRF55 under CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
try:
    from wifi_datahub.adapters.xrf55_schema import (
        XRF55_ACTION_NAMES,
        XRF55_RX_LINKS,
        XRF55_SUBCARRIERS,
        XRF55_TIME_SAMPLES,
        parse_xrf55_clip_id,
    )
except ModuleNotFoundError:  # Support running directly from an uninstalled checkout.
    sys.path.insert(0, str(ROOT / "src"))
    from wifi_datahub.adapters.xrf55_schema import (  # type: ignore[no-redef]
        XRF55_ACTION_NAMES,
        XRF55_RX_LINKS,
        XRF55_SUBCARRIERS,
        XRF55_TIME_SAMPLES,
        parse_xrf55_clip_id,
    )


KAGGLE_DATASET = "xrfdataset/xrf55"
OFFICIAL_DATASET_URL = "https://www.kaggle.com/datasets/xrfdataset/xrf55"
OFFICIAL_PROJECT_URL = "https://aiotgroup.github.io/XRF55/"
OFFICIAL_REPOSITORY_URL = "https://github.com/airslab2020/XRF55-repo"
OFFICIAL_QA_URL = "https://github.com/airslab2020/XRF55-repo/blob/main/Q%26A.md"

# Kaggle has exposed the WiFi branch under different leading folders across
# XRF55 versions.  Probe only these official-layout names, remember the first
# one that succeeds, and use it for the remaining 54 clips.
REMOTE_MEMBER_TEMPLATES = (
    "Scene1/Scene1/WiFi/{filename}",
    "WiFi/{filename}",
    "Raw_dataset/WiFi/{filename}",
    "dataset/Raw_dataset/WiFi/{filename}",
    "{filename}",
)


def representative_xrf55_filenames(
    subject_id: int = 1, repetition_id: int = 1,
) -> dict[int, str]:
    """Return a deterministic one-real-clip-per-action coverage plan."""

    if subject_id < 1 or repetition_id < 1:
        raise ValueError("XRF55 subject and repetition IDs must be positive")
    return {
        action_id: f"{subject_id:02d}_{action_id:02d}_{repetition_id:02d}.npy"
        for action_id in XRF55_ACTION_NAMES
    }


def candidate_remote_members(filename: str) -> tuple[str, ...]:
    """Return the official-layout Kaggle member candidates for a clip."""

    return tuple(template.format(filename=filename) for template in REMOTE_MEMBER_TEMPLATES)


def validate_xrf55_wifi(path: Path, expected_action_id: int | None = None) -> None:
    """Reject a wrong modality, label, or tensor before it enters the sample."""

    clip = parse_xrf55_clip_id(path)
    if expected_action_id is not None and clip.action_id != expected_action_id:
        raise ValueError(
            f"downloaded XRF55 action {clip.action_id}, expected {expected_action_id}: {path}"
        )
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    expected_shape = (XRF55_RX_LINKS * XRF55_SUBCARRIERS, XRF55_TIME_SAMPLES)
    if value.shape != expected_shape:
        raise ValueError(
            f"XRF55 WiFi sample must have official shape {expected_shape}, got {value.shape}: {path}"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"XRF55 WiFi sample is not numeric: {path}")


def kaggle_executable(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend([
        Path(sys.executable).with_name("kaggle"),
        ROOT.parent / ".venv" / "bin" / "kaggle",
    ])
    on_path = shutil.which("kaggle")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Kaggle CLI not found. Install it with: pip install kaggle")


def _extract_single_file(payload: Path, filename: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if payload.suffix.lower() != ".zip":
        shutil.move(str(payload), destination)
        return

    with zipfile.ZipFile(payload) as archive:
        safe_files = [
            item for item in archive.infolist()
            if not item.is_dir()
            and not Path(item.filename).is_absolute()
            and ".." not in Path(item.filename).parts
        ]
        matches = [item for item in safe_files if Path(item.filename).name == filename]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {filename!r} member in {payload.name}, found {len(matches)}"
            )
        with archive.open(matches[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def _download_member(kaggle: Path, remote_member: str, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="wisensehub-xrf55-") as scratch_name:
        scratch = Path(scratch_name)
        completed = subprocess.run(
            [
                str(kaggle), "datasets", "download", "-d", KAGGLE_DATASET,
                "-f", remote_member, "-p", str(scratch), "--force",
            ],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(message or f"Kaggle could not fetch {remote_member}")
        payloads = [path for path in scratch.rglob("*") if path.is_file()]
        if len(payloads) != 1:
            raise RuntimeError(
                f"Kaggle returned {len(payloads)} payloads for {remote_member}, expected one"
            )
        _extract_single_file(payloads[0], Path(remote_member).name, destination)


def write_xrf55_manifest(original: Path, selected: dict[int, dict[str, object]]) -> Path:
    manifest = {
        "schema_version": "1.0",
        "kind": "official-mini-sample",
        "dataset_id": "xrf55",
        "source": OFFICIAL_DATASET_URL,
        "official_project": OFFICIAL_PROJECT_URL,
        "official_repository": OFFICIAL_REPOSITORY_URL,
        "layout_and_labels_reference": OFFICIAL_QA_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": "one authentic WiFi recording for every official action label",
        "selected_files": selected,
        "label_vocabulary": XRF55_ACTION_NAMES,
        "source_tensor": {
            "shape": [XRF55_RX_LINKS * XRF55_SUBCARRIERS, XRF55_TIME_SAMPLES],
            "axes": ["flattened_rx_link_subcarrier", "time"],
        },
        "license": "CC BY-NC 4.0",
        "redistribution": "allowed for noncommercial use with attribution",
        "redistribution_note": (
            "Attribute the XRF55 authors and keep use/redistribution noncommercial under CC BY-NC 4.0."
        ),
    }
    path = original / "official-sample-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def fetch_xrf55(
    original: Path,
    *,
    kaggle: str | Path | None = None,
    force: bool = False,
    subject_id: int = 1,
    repetition_id: int = 1,
) -> str:
    """Fetch 55 real WiFi clips while preserving XRF55's official structure."""

    executable = kaggle_executable(kaggle)
    coverage = representative_xrf55_filenames(subject_id, repetition_id)
    wifi_root = Path(original) / "Scene1" / "Scene1" / "WiFi"
    selected: dict[int, dict[str, object]] = {}
    working_template: str | None = None

    for action_id, filename in coverage.items():
        target = wifi_root / filename
        remote_member: str | None = None
        status = "downloaded"
        if target.exists() and target.stat().st_size > 0 and not force:
            validate_xrf55_wifi(target, action_id)
            status = "existing"
        else:
            templates = (working_template,) if working_template else REMOTE_MEMBER_TEMPLATES
            errors: list[str] = []
            for template in templates:
                candidate = template.format(filename=filename)
                try:
                    _download_member(executable, candidate, target)
                    validate_xrf55_wifi(target, action_id)
                    remote_member = candidate
                    working_template = template
                    break
                except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                    if target.exists():
                        target.unlink()
                    errors.append(f"{candidate}: {exc}")
            if remote_member is None:
                raise RuntimeError(
                    f"could not fetch authentic XRF55 action {action_id} ({filename}); "
                    + " | ".join(errors)
                )
        if remote_member is None:
            template = working_template or REMOTE_MEMBER_TEMPLATES[0]
            remote_member = template.format(filename=filename)
        selected[action_id] = {
            "action_name": XRF55_ACTION_NAMES[action_id],
            "remote_member": remote_member,
            "local_path": target.relative_to(original).as_posix(),
            "bytes": target.stat().st_size,
            "status": status,
        }

    write_xrf55_manifest(Path(original), selected)
    return (
        "Fifty-five authentic XRF55 WiFi clips, one per official action label, "
        "at fixed Scene 1, subject 1, repetition 1. Licensed CC BY-NC 4.0."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "xrf55" / "original",
        help="Dataset original/ directory",
    )
    parser.add_argument("--kaggle", type=Path, help="Path to the Kaggle executable")
    parser.add_argument("--force", action="store_true", help="Replace validated local clips")
    parser.add_argument("--dry-run", action="store_true", help="Print the 55-file plan only")
    args = parser.parse_args()

    coverage = representative_xrf55_filenames()
    if args.dry_run:
        for action_id, filename in coverage.items():
            print(f"{action_id:02d}  {XRF55_ACTION_NAMES[action_id]}  {candidate_remote_members(filename)[0]}")
        return 0

    print(fetch_xrf55(args.output, kaggle=args.kaggle, force=args.force))
    print(f"Saved under: {args.output / 'Scene1' / 'Scene1' / 'WiFi'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
