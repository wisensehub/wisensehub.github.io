#!/usr/bin/env python3
"""Install the repository-owned WiSenseHub skill into a Codex skill directory."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "skills" / "wisensehub"


def default_target_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    return Path(codex_home).expanduser() / "skills" if codex_home else Path.home() / ".codex" / "skills"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=default_target_root(), help="Parent skills directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing wisensehub skill")
    parser.add_argument("--dry-run", action="store_true", help="Show the destination without changing files")
    args = parser.parse_args()

    if not (SOURCE / "SKILL.md").is_file():
        raise SystemExit(f"Skill source not found: {SOURCE}")

    target_root = args.target.expanduser().resolve()
    destination = target_root / "wisensehub"
    print(f"Source: {SOURCE}")
    print(f"Destination: {destination}")
    if args.dry_run:
        return 0

    target_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        raise SystemExit(f"Destination already exists: {destination}\nRerun with --force to replace it.")

    with tempfile.TemporaryDirectory(prefix=".wisensehub-install-", dir=target_root) as temp_name:
        staged = Path(temp_name) / "wisensehub"
        shutil.copytree(SOURCE, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        if destination.exists():
            backup = Path(temp_name) / "previous-wisensehub"
            destination.rename(backup)
            try:
                staged.rename(destination)
            except Exception:
                backup.rename(destination)
                raise
        else:
            staged.rename(destination)

    print(
        "Installed WiSenseHub skill. Restart Codex, select WiSenseHub in Skills "
        "or type $wisensehub, then ask it to download and prepare a sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
