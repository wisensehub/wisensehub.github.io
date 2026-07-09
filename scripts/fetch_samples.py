#!/usr/bin/env python3
"""Fetch small, structure-preserving dataset samples for the website.

For each pilot dataset this script downloads a tiny subset of the official
release into ``data/<id>/original/`` (preserving the official relative paths),
mirrors the subset to ``site/samples/<id>/``, and zips it as
``site/samples/<id>.zip`` for one-click download from the website.

Datasets that cannot be reached (auth failure, gate, network) are skipped and
the reason is recorded in ``site/samples/fetch-report.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "_cache"
SITE_SAMPLES = ROOT / "site" / "samples"

KAGGLE_CSI_BENCH_FILES = [
    "BreathingDetection/metadata/label_mapping.json",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_empty/Dpl_HC_near_Fridge/device_Hex_1cd6be1df30d/Diff_hard/session_1002__freq208_5G.h5",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_sleep/Dpl_HC_near_Faucet/device_Hex_1cd6be1df323/Diff_easy/2023_01_17/session_6166008__freq208_5G.h5",
]

WALLHACK_CLASS_FILES = {
    "0": "wallhack1.8k/LOS/BQ/b1.csv",
    "1": "wallhack1.8k/LOS/BQ/w1.csv",
    "2": "wallhack1.8k/LOS/BQ/ww1.csv",
}

FIGSHARE_SESSION = "room_1/1"
FIGSHARE_MAX_ROWS = 5200
NIST_PATTERNS = [f"Table 8/FrameRate3/BreathingPattern{index}.tar.gz" for index in range(1, 10)]


def log(message: str) -> None:
    print(f"[fetch-samples] {message}", flush=True)


def download(url: str, destination: Path, expected_bytes: Optional[int] = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_bytes is None or destination.stat().st_size == expected_bytes):
        log(f"cache hit: {destination.name}")
        return destination
    log(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "wisensehub-sample-fetcher"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    return destination


def truncate_csv(path: Path, max_rows: int) -> bool:
    """Keep the first max_rows lines of a CSV. Returns True when truncated."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = []
        for index, line in enumerate(handle):
            if index >= max_rows:
                break
            lines.append(line)
        else:
            return False
    path.write_text("".join(lines), encoding="utf-8")
    return True


def selective_extract(archive: Path, members: List[str], destination: Path) -> List[Path]:
    written = []
    with zipfile.ZipFile(archive) as handle:
        for member in members:
            info = handle.getinfo(member)
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise ValueError(f"unsafe archive member path: {member}")
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            written.append(target)
    return written


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "wisensehub-sample-fetcher"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def load_download_manifest(dataset_id: str) -> List[dict]:
    manifest = json.loads((ROOT / "catalog" / "downloads.json").read_text(encoding="utf-8"))["downloads"]
    return manifest[dataset_id]


# --- per-dataset fetchers -------------------------------------------------

def fetch_csi_bench(original: Path) -> str:
    kaggle = Path(sys.executable).parent / "kaggle"
    if not kaggle.exists():
        raise RuntimeError("kaggle CLI not found next to the interpreter; pip install kaggle")
    for remote_path in KAGGLE_CSI_BENCH_FILES:
        with tempfile.TemporaryDirectory() as scratch:
            subprocess.run(
                [str(kaggle), "datasets", "download", "guozhenjennzhu/csi-bench",
                 "-f", remote_path, "-p", scratch],
                check=True, capture_output=True, text=True,
            )
            produced = list(Path(scratch).iterdir())
            if not produced:
                raise RuntimeError(f"kaggle returned no file for {remote_path}")
            payload = produced[0]
            target = original / remote_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if payload.suffix == ".zip" and not remote_path.endswith(".zip"):
                with zipfile.ZipFile(payload) as handle:
                    names = handle.namelist()
                    with handle.open(names[0]) as source, target.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
            else:
                shutil.move(str(payload), target)
        log(f"fetched {remote_path}")
    return "BreathingDetection sessions for both labels (empty and sleep) plus the official label mapping."


def fetch_wallhack(original: Path) -> str:
    item = load_download_manifest("wallhack18k")[0]
    archive = download(item["url"], CACHE / item["name"], item.get("bytes"))
    members = list(WALLHACK_CLASS_FILES.values())
    written = selective_extract(archive, members, original)
    truncated = [path for path in written if truncate_csv(path, 1000)]
    note = (f"One CSV per class from the official Zenodo archive ({', '.join(Path(m).name for m in members)}). "
            "Classes: no presence, walking, arm waving.")
    if truncated:
        note += " Files truncated to the first 1000 packets to keep the sample small."
    return note


def fetch_nist(original: Path) -> str:
    import io
    import tarfile

    item = load_download_manifest("nist-breathesmart")[0]
    archive = download(item["url"], CACHE / item["name"], item.get("bytes"))
    extracted = 0
    with zipfile.ZipFile(archive) as handle:
        for member_name in NIST_PATTERNS:
            chosen = next((info for info in handle.infolist() if info.filename == member_name), None)
            if chosen is None:
                raise RuntimeError(f"missing NIST bundle {member_name}")
            payload = io.BytesIO(handle.read(chosen))
            base = original / Path(chosen.filename).parent
            with tarfile.open(fileobj=payload, mode="r:gz") as bundle:
                for member in bundle.getmembers():
                    if not member.isfile() or ".." in Path(member.name).parts:
                        continue
                    target = base / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    with target.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    extracted += 1
    return (f"All nine FrameRate3 breathing patterns from the official NIST release "
            f"({', '.join(Path(name).stem for name in NIST_PATTERNS)}).")


def fetch_figshare_csi_har(original: Path) -> str:
    files = fetch_json("https://api.figshare.com/v2/articles/14386892/files")
    archives = [item for item in files if item["name"].lower().endswith(".zip")]
    if not archives:
        raise RuntimeError("no zip archive listed by figshare API")
    item = archives[0]
    archive = download(item["download_url"], CACHE / item["name"], item.get("size"))
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    data_member = f"{FIGSHARE_SESSION}/data.csv"
    label_member = f"{FIGSHARE_SESSION}/label.csv"
    if data_member not in names or label_member not in names:
        raise RuntimeError(f"expected session files missing: {FIGSHARE_SESSION}")
    written = selective_extract(archive, [data_member, label_member], original)
    rows = None
    for path in written:
        if path.name == "data.csv" and truncate_csv(path, FIGSHARE_MAX_ROWS):
            rows = FIGSHARE_MAX_ROWS
    if rows:
        for path in written:
            if path.name == "label.csv":
                truncate_csv(path, rows)
    return (f"Session {FIGSHARE_SESSION} with all seven HAR labels from the official Figshare release "
            f"(standing, walking, sitting, lying, get up/down, no person).")


class _RemoteFile:
    """Read-only file object over an HTTP resource that supports Range requests."""

    def __init__(self, url: str, size: int, block: int = 8 * 1024 * 1024):
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self.cache: Dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _fetch_block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * self.block
            end = min(start + self.block, self.size) - 1
            request = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "wisensehub-sample-fetcher"})
            with urllib.request.urlopen(request) as response:
                if len(self.cache) > 8:
                    self.cache.clear()
                self.cache[index] = response.read()
        return self.cache[index]

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.pos
        out = bytearray()
        while count > 0 and self.pos < self.size:
            index, offset = divmod(self.pos, self.block)
            chunk = self._fetch_block(index)[offset:offset + count]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            count -= len(chunk)
        return bytes(out)


def fetch_operanet(original: Path, max_rows: int = 600) -> str:
    """Pull the smallest WiFi CSI MAT out of the 36 GB collection zip via HTTP ranges."""
    from scipy.io import loadmat, savemat

    sys.path.insert(0, str(ROOT / "src"))
    from wifi_datahub.adapters.official_profiles import _column_mapping

    articles = fetch_json("https://api.figshare.com/v2/collections/5551209/articles?page_size=100")
    wifi = next((a for a in articles if a["title"].startswith("wificsi")), None)
    if wifi is None:
        raise RuntimeError("no wificsi article found in the OPERAnet collection")
    detail = fetch_json(f"https://api.figshare.com/v2/articles/{wifi['id']}")
    item = detail["files"][0]
    remote = zipfile.ZipFile(_RemoteFile(item["download_url"], item["size"]))  # type: ignore[arg-type]
    mats = sorted((info for info in remote.infolist() if info.filename.lower().endswith(".mat")),
                  key=lambda info: info.file_size)
    if not mats:
        raise RuntimeError("no MAT members inside the OPERAnet wificsi zip")
    chosen = mats[0]
    cached = CACHE / "operanet" / Path(chosen.filename).name
    if not cached.exists() or cached.stat().st_size != chosen.file_size:
        cached.parent.mkdir(parents=True, exist_ok=True)
        log(f"extracting {chosen.filename} ({chosen.file_size} bytes) from remote zip")
        with remote.open(chosen) as source, cached.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
    columns = _column_mapping(loadmat(cached, simplify_cells=True))
    if not columns:
        raise RuntimeError("OPERAnet MAT table columns not found in sample file")
    truncated = {name: values[:max_rows] for name, values in columns.items()}
    target = original / "WiFi" / Path(chosen.filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    savemat(target, truncated, do_compression=True)
    return (f"WiFi CSI recording {Path(chosen.filename).name} from the official Figshare collection, "
            f"truncated to the first {max_rows} packets to keep the sample small.")


FETCHERS = {
    "csi-bench": fetch_csi_bench,
    "wallhack18k": fetch_wallhack,
    "nist-breathesmart": fetch_nist,
    "figshare-csi-har": fetch_figshare_csi_har,
    "operanet": fetch_operanet,
}


# --- mirroring ------------------------------------------------------------

def mirror_and_zip(dataset_id: str) -> Dict[str, int]:
    original = DATA / dataset_id / "original"
    sample_dir = SITE_SAMPLES / dataset_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    shutil.copytree(original, sample_dir)
    archive_path = SITE_SAMPLES / f"{dataset_id}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(sample_dir.rglob("*")):
            if path.is_file():
                handle.write(path, Path(dataset_id) / path.relative_to(sample_dir))
    total = sum(path.stat().st_size for path in sample_dir.rglob("*") if path.is_file())
    return {"files": sum(1 for p in sample_dir.rglob("*") if p.is_file()),
            "bytes": total, "zip_bytes": archive_path.stat().st_size}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(FETCHERS),
                        help="dataset ids to fetch (default: all pilot datasets)")
    parser.add_argument("--keep-original", action="store_true",
                        help="do not wipe data/<id>/original before fetching")
    args = parser.parse_args(argv)
    requested = args.datasets or list(FETCHERS)
    report: Dict[str, dict] = {}
    report_path = SITE_SAMPLES / "fetch-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    for dataset_id in requested:
        if dataset_id not in FETCHERS:
            log(f"unknown dataset {dataset_id}; choices: {', '.join(FETCHERS)}")
            continue
        original = DATA / dataset_id / "original"
        try:
            if not args.keep_original and original.exists():
                shutil.rmtree(original)
            original.mkdir(parents=True, exist_ok=True)
            note = FETCHERS[dataset_id](original)
            stats = mirror_and_zip(dataset_id)
            report[dataset_id] = {"status": "ok", "note": note, **stats}
            log(f"{dataset_id}: sample ready ({stats['files']} files, {stats['bytes']} bytes)")
        except Exception as exc:  # noqa: BLE001 - skip-on-failure by design
            report[dataset_id] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
            log(f"{dataset_id}: SKIPPED ({exc})")
    SITE_SAMPLES.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
