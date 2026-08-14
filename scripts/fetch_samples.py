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
import gzip
import hashlib
import http.client
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .sample_sources_xrf55 import fetch_xrf55
except ImportError:  # Direct execution: python scripts/fetch_samples.py
    from sample_sources_xrf55 import fetch_xrf55

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "_cache"
SITE_SAMPLES = ROOT / "site" / "samples"

KAGGLE_CSI_BENCH_FILES = [
    "BreathingDetection/metadata/label_mapping.json",
    "FallDetection/metadata/label_mapping.json",
    "Localization/metadata/label_mapping.json",
    "MotionSourceRecognition/metadata/label_mapping.json",
    "Multitask/HumanActivityRecognition/metadata/label_mapping.json",
    "Multitask/HumanIdentification/metadata/label_mapping.json",
    "Multitask/ProximityRecognition/metadata/label_mapping.json",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_empty/Dpl_HC_near_Fridge/device_Hex_1cd6be1df30d/Diff_hard/session_613001__freq208_5G.h5",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_sleep/Dpl_Set1/device_Hex_1cd6be198931/Diff_easy/2022_03_08/session_24588001__freq208_5G.h5",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_sleep/Dpl_Set2/device_Hex_1cd6be198931/Diff_hard/2022_03_12/session_74249002__freq208_5G.h5",
    "BreathingDetection/sub_Human/user_U06/env_E10/act_sleep/Dpl_Set3/device_Hex_1cd6be198931/Diff_medium/2022_03_17/session_101876002__freq208_5G.h5",
    "FallDetection/sub_Human/user_U07/act_Fall/env_E21/device_HP/session_2168000__freq232.h5",
    "FallDetection/sub_Human/user_U08/act_Nonfall/env_E22/device_ESP32/session_10000__freq64.h5",
    "Localization/sub_Human/env_E01/Diff_easy/Dpl_case1/act_singleuser_walking/user_U01/motionsrc_001/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_1000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_easy/Dpl_case1/act_singleuser_walking/user_U01/motionsrc_010/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_5000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_easy/Dpl_case1/act_singleuser_walking/user_U01/motionsrc_100/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_10000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_hard/Dpl_case1/act_singleuser_walking/user_U01/motionsrc_010/device_GoveePlug-2G_AmazonPlug-2G_WyzePlug-2G/session_17000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_hard/Dpl_case1/act_twouser_walking/user_UM04/motionsrc_101/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_21000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_hard/Dpl_case1/act_twouser_walking/user_UM04/motionsrc_110/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_29000__freq168.h5",
    "Localization/sub_Human/env_E01/Diff_medium/Dpl_case1/act_singleuser_walking/user_U01/motionsrc_001/device_GoveePlug-2G_AmazonPlug-2G_WyzePlug-2G/session_37000__freq168.h5",
    "Localization/sub_Human/env_E03/Diff_hard/Dpl_case1/act_twouser_walking/user_UM01/motionsrc_011/device_EchoShow8-5G_AppleHomePod-5G_EchoSpot-5G/session_10000__freq168.h5",
    "Localization/sub_Human/env_E07/Diff_easy/Dpl_case1/act_empty/motionsrc_000/device_EchoDot3rd-5G_EchoDot3rd-5G_EchoDot3rd-5G/session_1000__freq2496.h5",
    "MotionSourceRecognition/sub_Fan/user_F01/act_Test_1/env_E11/device_HP/session_708000__freq232.h5",
    "MotionSourceRecognition/sub_Human/user_U03/act_Human/env_E19/device_HP/session_1753000__freq232.h5",
    "MotionSourceRecognition/sub_IRobot/user_IR01/act_Test_1/env_E11/device_HP/session_768000__freq232.h5",
    "MotionSourceRecognition/sub_Pet/user_P01/act_Max_Test_1/env_E20/device_HP/session_990000__freq232.h5",
    "Multitask/sub_Human_h5/user_U01/act_jumping/env_E01/device_AmazonPlug/session_16000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_running/env_E01/device_AmazonPlug/session_199000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_seated-breathing/env_E01/device_AmazonPlug/session_241000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_walking-case1/env_E01/device_AmazonPlug/session_265000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_walking-case3/env_E01/device_AmazonPlug/session_313000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_walking-case4/env_E01/device_AmazonPlug/session_337000__freq56.h5",
    "Multitask/sub_Human_h5/user_U01/act_wavinghand/env_E01/device_AmazonPlug/session_361000__freq56.h5",
    "Multitask/sub_Human_h5/user_U02/act_jumping/env_E01/device_AmazonPlug/session_394000__freq56.h5",
    "Multitask/sub_Human_h5/user_U03/act_jumping/env_E02/device_AmazonPlug/session_742000__freq56.h5",
    "Multitask/sub_Human_h5/user_U04/act_jumping/env_E03/device_AmazonPlug/session_898000__freq56.h5",
    "Multitask/sub_Human_h5/user_U05/act_jumping/env_E04/device_AmazonPlug/session_1192000__freq56.h5",
    "Multitask/sub_Human_h5/user_U06/act_jumping/env_E06/device_AmazonPlug/session_1765000__freq56.h5",
    "Multitask/sub_Human_h5/user_U22/act_jumping/env_E27/device_AmazonPlug/session_2101000__freq56.h5",
]

WALLHACK_CLASS_FILES = {
    "0": "wallhack1.8k/LOS/BQ/b1.csv",
    "1": "wallhack1.8k/LOS/BQ/w1.csv",
    "2": "wallhack1.8k/LOS/BQ/ww1.csv",
}

FIGSHARE_SESSIONS = ("room_1/1", "room_2/1", "room_3/5")
FIGSHARE_MAX_ROWS = 5200
NIST_PATTERNS = [f"Table 8/FrameRate3/BreathingPattern{index}.tar.gz" for index in range(1, 10)]
ARIL_ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1SCxUHbl6rNWM3kT0c-D4s_kyAero9_-o&export=download&confirm=t"
)
ARIL_ARCHIVE_BYTES = 210_789_354
ARIL_TRAIN_MEMBER = "data/train_data_split_amp.mat"
CSIDA_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/gyr6c4nbsc/files/"
    "d0344aa0-f8ee-4357-9e9a-e4e2519699a4/file_downloaded"
)
CSIDA_ARCHIVE_BYTES = 5_298_186_429
CSIDA_ARCHIVE_SHA256 = "e9ea189f747d7562ee2d363bfdf383ed2cfe116cda468523ed055b34f4724e38"
CSIDA_LABEL_ARRAYS = (
    "csi_label_act", "csi_label_env", "csi_label_loc", "csi_label_user",
)
EHUNAM_URL = "https://ndownloader.figshare.com/files/52814369"
EHUNAM_BYTES = 77_732_935_390
EXPOSING_URL = "https://zenodo.org/api/records/7732595/files/S1.zip/content"
EXPOSING_BYTES = 11_071_365_703
GLASGOW_ACTIVITY_URL = "https://researchdata.gla.ac.uk/1283/1/Location1_Z1Z2Z3.zip"
GLASGOW_ACTIVITY_BYTES = 404_443_572
GLASGOW_ACTIVITY_ARCHIVES = [
    (1, GLASGOW_ACTIVITY_URL, GLASGOW_ACTIVITY_BYTES),
    (2, "https://researchdata.gla.ac.uk/1283/2/Location2_Z1Z2Z3.zip", 668_430_519),
    (3, "https://researchdata.gla.ac.uk/1283/3/Location3_Z1Z2Z3.zip", 780_130_696),
]
GLASGOW_MULTIUSER_URL = (
    "https://researchdata.gla.ac.uk/1151/1/"
    "Data_Set_for_5G-Enabled_Contactless_Multi-User_Presence_andActivity_Detection_"
    "for_Independent_Assisted_Living.7z"
)
GLASGOW_MULTIUSER_BYTES = 843_889_798
WIRELESS_HAR_URL = "https://ndownloader.figshare.com/files/36582708"
WIRELESS_HAR_BYTES = 8_554_140_385
PRESENCE_URL = "https://zenodo.org/api/records/3676058/files/260-4.csi.json.gz/content"
PRESENCE_ANNOTATIONS_URL = "https://zenodo.org/api/records/3676058/files/annotations.csv/content"
MMFI_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1ExV3AQeHstQ3Z1VBFOC0z1BDtTD5nZ1B&export=download&confirm=t"
)
MMFI_BYTES = 22_301_071_119
NTU_HAR_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1DszE7byFzlpyI9gZvmVn51fTr8L1iZaI&export=download&confirm=t"
)
NTU_HAR_BYTES = 4_740_430_217
NTU_ID_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1IKTg5M7vDdZPnt6649i2Z3jiR4Fivsy1&export=download&confirm=t"
)
NTU_ID_BYTES = 2_817_627_776
UT_HAR_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1fEiI3nAoOsddR5qcJQXqz4ocM3aMAcwz&export=download&confirm=t"
)
UT_HAR_BYTES = 383_128_602
WIDAR_BVP_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=14vp4D8W0X2bDLpXnpP-U_VT9PIGkVf_4&export=download&confirm=t"
)
WIDAR_BVP_BYTES = 1_143_833_206
WIFI_TAD_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1gy0ppFtypVTtgBfrFzdMJUbXTb1MbPSK&export=download&confirm=t"
)
WIFI_TAD_BYTES = 225_363_686
OPERANET_WIFI1_URL = "https://ndownloader.figshare.com/files/30689729"
OPERANET_WIFI1_BYTES = 36_490_626_012
OPERANET_ACTIVITY_EXPERIMENTS = {
    # Table 2 of the OPERAnet data descriptor.  These five recordings are the
    # smallest same-receiver set whose released packet annotations cover every
    # HAR state, including background and the steady noactivity intervals.
    "019": ("background",),
    "002": ("walk", "noactivity"),
    "003": ("sit", "stand"),
    "004": ("liedown", "standfromlie"),
    "005": ("bodyrotate",),
}

# These official files are publicly fetchable, but the releases do not grant
# WiSenseHub permission to redistribute their raw measurements.  We generate
# real derived previews locally and give the user an exact official-source
# fetch command instead of silently repackaging the bytes into site/samples.
OFFICIAL_FETCH_ONLY = {"aril", "ntu-fi", "ut-har", "wallhack18k", "wifi-tad", "wiar"}


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


def write_official_manifest(original: Path, payload: dict) -> None:
    (original / "official-sample-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "kind": "official-mini-sample", **payload}, indent=2) + "\n",
        encoding="utf-8",
    )


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
    return ("CSI-Bench mini sample: one recording per official label across all seven tasks, plus easy/medium/hard difficulty coverage where those settings exist, and each task label_mapping.json.")


def fetch_wallhack(original: Path) -> str:
    item = load_download_manifest("wallhack18k")[0]
    archive = download(item["url"], CACHE / item["name"], item.get("bytes"))
    members = [
        f"wallhack1.8k/{path}/{antenna}/{prefix}{index}.csv"
        for path in ("LOS", "NLOS")
        for antenna in ("BQ", "PIFA")
        for prefix, indices in (("b", (1,)), ("w", range(1, 6)), ("ww", range(1, 6)))
        for index in indices
    ]
    written = selective_extract(archive, members, original)
    truncated = [path for path in written if truncate_csv(path, 1000)]
    note = ("All 44 raw CSV combinations from the official Zenodo archive: three activities "
            "across LOS/NLOS, BQ/PIFA, and rooms 1–5 where released.")
    if truncated:
        note += " Each file is shortened to its first 1000 authentic packets."
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
    selected_members: List[str] = []
    for session in FIGSHARE_SESSIONS:
        required = [f"{session}/data.csv", f"{session}/label.csv"]
        optional = f"{session}/label_boxes.csv"
        if not all(member in names for member in required):
            raise RuntimeError(f"expected session files missing: {session}")
        selected_members.extend(required)
        if optional in names:
            selected_members.append(optional)
    written = selective_extract(archive, selected_members, original)
    session_rows: Dict[Path, int] = {}
    for path in written:
        if path.name == "data.csv" and truncate_csv(path, FIGSHARE_MAX_ROWS):
            session_rows[path.parent] = FIGSHARE_MAX_ROWS
    for path in written:
        if path.name in {"label.csv", "label_boxes.csv"} and path.parent in session_rows:
            truncate_csv(path, session_rows[path.parent])
    write_official_manifest(original, {
        "dataset_id": "figshare-csi-har",
        "source": "https://figshare.com/articles/dataset/14386892",
        "selected_members": selected_members,
        "room_coverage": ["room_1", "room_2", "room_3"],
        "labels": ["standing", "walking", "sitting", "lying", "get up", "get down", "no person"],
        "license": "CC BY 4.0",
    })
    return ("Authentic sessions from all three rooms, each retaining real labels; together they "
            "cover all seven HAR labels. Licensed CC BY 4.0.")


def select_aril_representatives(activity_label, location_label) -> List[int]:
    """Pick one real sample per location while covering all six gestures."""
    import numpy as np

    activity = np.asarray(activity_label).reshape(-1)
    location = np.asarray(location_label).reshape(-1)
    if activity.size != location.size:
        raise ValueError("ARIL activity and location label counts differ")

    selected: List[int] = []
    for location_id in range(16):
        desired_gesture = location_id % 6
        matches = np.flatnonzero((location == location_id) & (activity == desired_gesture))
        if not matches.size:
            matches = np.flatnonzero(location == location_id)
        if not matches.size:
            raise ValueError(f"ARIL source has no sample for location {location_id}")
        selected.append(int(matches[0]))

    if set(activity[selected].tolist()) != set(range(6)):
        raise ValueError("ARIL representative subset does not cover all six gestures")
    return selected


def fetch_aril(original: Path) -> str:
    """Create a small authentic subset from the official ARIL data archive."""
    import numpy as np
    from scipy.io import loadmat, savemat

    extractor = shutil.which("bsdtar")
    if extractor is None:
        raise RuntimeError("ARIL RAR extraction requires bsdtar/libarchive")

    archive = download(ARIL_ARCHIVE_URL, CACHE / "aril" / "data.rar", ARIL_ARCHIVE_BYTES)
    with tempfile.TemporaryDirectory(prefix="aril-official-") as scratch_name:
        scratch = Path(scratch_name)
        completed = subprocess.run(
            [extractor, "-xf", str(archive), "-C", str(scratch), ARIL_TRAIN_MEMBER],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"could not extract official ARIL train split: {message}")
        source = scratch / ARIL_TRAIN_MEMBER
        payload = loadmat(source, simplify_cells=True)

    data = np.asarray(payload["train_data"])
    activity = np.asarray(payload["train_activity_label"]).reshape(-1)
    location = np.asarray(payload["train_location_label"]).reshape(-1)
    selected = select_aril_representatives(activity, location)
    target = original / "train_data_split_amp.mat"
    savemat(target, {
        "train_data": data[selected],
        "train_activity_label": activity[selected, None],
        "train_location_label": location[selected, None],
    }, do_compression=True)
    source_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "aril",
        "kind": "official-mini-sample",
        "source": "https://github.com/geekfeiw/ARIL",
        "source_archive": ARIL_ARCHIVE_URL,
        "source_archive_sha256": source_digest,
        "source_member": ARIL_TRAIN_MEMBER,
        "source_sample_count": int(data.shape[0]),
        "selected_source_indices": selected,
        "selected_sample_count": len(selected),
        "gesture_labels": sorted(np.unique(activity[selected]).astype(int).tolist()),
        "location_labels": sorted(np.unique(location[selected]).astype(int).tolist()),
        "license_note": "The official repository does not state redistribution terms; confirm them before publishing this subset.",
    }
    (original / "official-sample-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return (
        "Authentic processed CSI from the official ARIL archive: one real sample per location "
        "with all six gestures represented (16 samples total). Confirm the official repository "
        "terms before redistributing this subset."
    )


class _RemoteFile(io.RawIOBase):
    """Read-only file object over an HTTP resource that supports Range requests."""

    def __init__(self, url: str, size: int, block: int = 8 * 1024 * 1024):
        super().__init__()
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self.cache: Dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
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
            expected = end - start + 1
            payload = bytearray()
            for attempt in range(8):
                request_start = start + len(payload)
                request = urllib.request.Request(
                    self.url,
                    headers={
                        "Range": f"bytes={request_start}-{end}",
                        "User-Agent": "wisensehub-sample-fetcher",
                    },
                )
                try:
                    with urllib.request.urlopen(request) as response:
                        chunk = response.read()
                except http.client.IncompleteRead as exc:
                    chunk = exc.partial
                except urllib.error.HTTPError as exc:
                    if exc.code not in {408, 429, 500, 502, 503, 504}:
                        raise
                    chunk = b""
                    log(
                        f"retrying transient HTTP {exc.code} for range "
                        f"{request_start}-{end} (attempt {attempt + 2}/8)"
                    )
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
                except urllib.error.URLError as exc:
                    chunk = b""
                    log(
                        f"retrying HTTP range {request_start}-{end} after {exc.reason} "
                        f"(attempt {attempt + 2}/8)"
                    )
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
                if chunk:
                    payload.extend(chunk)
                if len(payload) == expected:
                    break
                if len(payload) > expected:
                    raise RuntimeError(
                        f"range server returned too many bytes for {request_start}-{end}: {len(payload)}/{expected}"
                    )
                log(f"retrying incomplete HTTP range {request_start}-{end} (attempt {attempt + 2}/8)")
            if len(payload) != expected:
                raise RuntimeError(f"incomplete HTTP range {start}-{end}: {len(payload)}/{expected} bytes")
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[index] = bytes(payload)
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


def select_csida_representatives(labels: Dict[str, object]) -> List[int]:
    """Greedily cover every released CSIDA label value with few real clips."""
    import numpy as np

    values = {name: np.asarray(value).reshape(-1) for name, value in labels.items()}
    sizes = {value.size for value in values.values()}
    if not values or len(sizes) != 1:
        raise ValueError("CSIDA label arrays must be present and have matching lengths")

    uncovered = {name: set(np.unique(value).tolist()) for name, value in values.items()}
    selected: List[int] = []
    sample_count = next(iter(sizes))
    while any(uncovered.values()):
        best_index = -1
        best_score = 0
        for index in range(sample_count):
            if index in selected:
                continue
            score = sum(values[name][index].item() in remaining for name, remaining in uncovered.items())
            if score > best_score:
                best_index, best_score = index, score
        if best_index < 0:
            raise ValueError("CSIDA representative selection could not cover all labels")
        selected.append(best_index)
        for name, remaining in uncovered.items():
            remaining.discard(values[name][best_index].item())
    return selected


def fetch_csida(original: Path) -> str:
    """Read only representative chunks from the official 4.93 GB CSIDA ZIP."""
    import numpy as np
    import zarr

    remote = zipfile.ZipFile(_RemoteFile(CSIDA_ARCHIVE_URL, CSIDA_ARCHIVE_BYTES))  # type: ignore[arg-type]
    with tempfile.TemporaryDirectory(prefix="csida-official-") as scratch_name:
        scratch = Path(scratch_name)
        members = [".zgroup", "csi_data_amp/.zarray"]
        for name in CSIDA_LABEL_ARRAYS:
            members.extend((f"{name}/.zarray", f"{name}/0"))
        selective_extract_from_handle(remote, members, scratch)

        labels = {
            name: np.asarray(zarr.open(str(scratch / name), mode="r")).reshape(-1)
            for name in CSIDA_LABEL_ARRAYS
        }
        selected = select_csida_representatives(labels)

        # csi_data_amp uses two samples per chunk: [2, 1800, 3, 114].
        chunk_members = sorted({f"csi_data_amp/{index // 2}.0.0.0" for index in selected})
        selective_extract_from_handle(remote, chunk_members, scratch)
        source_amplitude = zarr.open(str(scratch / "csi_data_amp"), mode="r")
        amplitude = np.stack([np.asarray(source_amplitude[index], dtype=np.float32) for index in selected])

    zarr.save(str(original / "csi_data_amp"), amplitude)
    for name, values in labels.items():
        zarr.save(str(original / name), values[selected])

    manifest = {
        "schema_version": "1.0",
        "dataset_id": "csida",
        "kind": "official-mini-sample",
        "source": "https://data.mendeley.com/datasets/gyr6c4nbsc/1",
        "source_archive": CSIDA_ARCHIVE_URL,
        "source_archive_sha256": CSIDA_ARCHIVE_SHA256,
        "source_axis_order": ["sample", "time", "rx_link", "subcarrier"],
        "source_shape": [2844, 1800, 3, 114],
        "selected_source_indices": selected,
        "selected_sample_count": len(selected),
        "label_coverage": {
            name: sorted(np.unique(values[selected]).astype(int).tolist())
            for name, values in labels.items()
        },
        "license": "CC BY 4.0",
        "attribution": "CSIDA-1, Zhang et al., Mendeley Data, DOI: 10.17632/gyr6c4nbsc.1",
    }
    (original / "official-sample-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return (
        f"{len(selected)} authentic processed CSIDA clips from the official Mendeley release, "
        "covering all six gestures, both rooms, all released position IDs, and all five users. "
        "Licensed CC BY 4.0."
    )


def selective_extract_from_handle(
    archive: zipfile.ZipFile, members: List[str], destination: Path,
) -> List[Path]:
    """Extract exact safe members from an already-open local or remote ZIP."""
    written: List[Path] = []
    for member in members:
        if Path(member).is_absolute() or ".." in Path(member).parts:
            raise ValueError(f"unsafe archive member path: {member}")
        info = archive.getinfo(member)
        target = destination / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == info.file_size:
            written.append(target)
            continue
        with archive.open(info) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        written.append(target)
    return written


def fetch_ehunam(original: Path) -> str:
    """Extract authentic recordings covering every released EHUNAM label."""
    remote_source = _RemoteFile(EHUNAM_URL, EHUNAM_BYTES)
    remote = zipfile.ZipFile(remote_source)  # type: ignore[arg-type]
    mats = sorted(
        (info for info in remote.infolist() if info.filename.lower().endswith(".mat")),
        key=lambda info: info.file_size,
    )
    def member_features(info: zipfile.ZipInfo) -> set[str]:
        parts = Path(info.filename).stem.split("_")
        if len(parts) < 9:
            return set()
        campaign, _collection_set, _receiver, application, people, activity, machines, status, _sequence = parts[:9]
        features = {
            f"campaign:{campaign}", f"application:{application}",
            f"occupancy:{0 if people == '#' else len(people)}",
        }
        features.update(f"identity:{person}" for person in people if person != "#")
        if activity != "#":
            features.add(f"activity-combination:{activity}")
            features.update(f"activity:{code}" for code in activity)
        if machines != "#":
            features.add(f"machine-combination:{machines}")
            features.update(f"machine:{machine}" for machine in machines)
        if status != "#":
            features.add(f"machine-state:{status}")
        return features

    features_by_name = {info.filename: member_features(info) for info in mats}
    universe = set().union(*features_by_name.values())
    chosen: list[zipfile.ZipInfo] = []
    uncovered = set(universe)
    while uncovered:
        best = max(
            mats,
            key=lambda info: (len(features_by_name[info.filename] & uncovered), -info.file_size),
        )
        covered = features_by_name[best.filename] & uncovered
        if not covered:
            raise RuntimeError(f"EHUNAM coverage selection stalled with {sorted(uncovered)}")
        chosen.append(best)
        uncovered -= covered
    # Figshare stores this archive with ZIP method 9 (Deflate64), which Python's
    # zipfile can index but not decompress. Decode only each chosen compressed
    # member after locating it with HTTP range reads.
    try:
        from inflate64 import Inflater
    except ImportError as exc:
        raise RuntimeError("EHUNAM extraction requires the data extra: pip install -e '.[data]'") from exc
    for info in chosen:
        remote_source.seek(info.header_offset)
        header = remote_source.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"invalid EHUNAM local ZIP header for {info.filename}")
        name_length = int.from_bytes(header[26:28], "little")
        extra_length = int.from_bytes(header[28:30], "little")
        remote_source.seek(info.header_offset + 30 + name_length + extra_length)
        compressed = remote_source.read(info.compress_size)
        decoded = Inflater().inflate(compressed)
        if len(decoded) != info.file_size or zlib.crc32(decoded) != info.CRC:
            raise RuntimeError(f"EHUNAM integrity check failed for {info.filename}")
        target = original / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(decoded)
    write_official_manifest(original, {
        "dataset_id": "ehunam",
        "source": "https://doi.org/10.6084/m9.figshare.28541225",
        "source_archive": EHUNAM_URL,
        "selected_members": [info.filename for info in chosen],
        "label_coverage": sorted(universe),
        "license": "CC BY 4.0",
        "attribution": "EHUNAM, Diaz-Sanchez et al., Figshare dataset 28541225",
    })
    return (f"{len(chosen)} authentic EHUNAM recordings selected from the official "
            f"release to cover all {len(universe)} published label values. Licensed CC BY 4.0.")


def fetch_exposing_csi(original: Path, max_packets: int = 600) -> str:
    """Extract one real 160 MHz AX-CSI capture for every activity label."""
    from scipy.io import loadmat, savemat

    remote = zipfile.ZipFile(_RemoteFile(EXPOSING_URL, EXPOSING_BYTES))  # type: ignore[arg-type]
    chosen: Dict[str, zipfile.ZipInfo] = {}
    for code in "ABCDEFGHIJKL":
        member = f"S1/S1a_{code}.mat"
        try:
            chosen[code] = remote.getinfo(member)
        except KeyError as exc:
            raise RuntimeError(f"Exposing the CSI archive has no fixed-receiver capture {member}") from exc
    remote.close()

    def cache_member(info: zipfile.ZipInfo) -> Path:
        cached = CACHE / "exposing-csi" / Path(info.filename).name
        if cached.exists() and cached.stat().st_size == info.file_size:
            log(f"cache hit: {cached.relative_to(ROOT)}")
            return cached
        cached.parent.mkdir(parents=True, exist_ok=True)
        partial = cached.with_name(f"{cached.name}.download")
        log(f"extracting {info.filename} ({info.file_size} bytes) from remote zip")
        # Use one range-backed archive handle per worker.  This keeps the
        # official 300 MB members independent and avoids serializing twelve
        # long downloads through one shared seek pointer.
        with zipfile.ZipFile(
            _RemoteFile(EXPOSING_URL, EXPOSING_BYTES)  # type: ignore[arg-type]
        ) as archive:
            remote_info = archive.getinfo(info.filename)
            with archive.open(remote_info) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
        if partial.stat().st_size != info.file_size:
            raise RuntimeError(
                f"incomplete Exposing the CSI member {info.filename}: "
                f"{partial.stat().st_size} of {info.file_size} bytes"
            )
        partial.replace(cached)
        return cached

    # Three concurrent transfers cut the wait substantially while keeping
    # memory bounded; MAT parsing remains sequential below.
    with ThreadPoolExecutor(max_workers=3) as pool:
        cached_by_code = dict(zip(chosen, pool.map(cache_member, chosen.values())))

    selected_packets: Dict[str, int] = {}
    source_packets: Dict[str, int] = {}
    for code, info in chosen.items():
        cached = cached_by_code[code]
        payload = loadmat(cached, simplify_cells=True)
        csi_key = next((key for key in ("csi", "csi_buff") if key in payload), None)
        if csi_key is None:
            raise RuntimeError(f"{info.filename} does not contain csi or csi_buff")
        csi = payload[csi_key]
        packet_count = min(max_packets, int(csi.shape[0]))
        target = original / info.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        savemat(target, {csi_key: csi[:packet_count]}, do_compression=True)
        selected_packets[code] = packet_count
        source_packets[code] = int(csi.shape[0])
    write_official_manifest(original, {
        "dataset_id": "exposing-csi",
        "source": "https://zenodo.org/records/7732595",
        "source_archive": EXPOSING_URL,
        "selected_members": {code: info.filename for code, info in chosen.items()},
        "source_packets": source_packets,
        "selected_packets": selected_packets,
        "receiver_setting": "S1a fixed for every activity",
        "license": "CC BY-SA 4.0",
    })
    return (f"Twelve authentic AX-CSI captures from the same S1a receiver, one for every "
            f"activity A-L, each shortened to its first {max_packets} packets. Licensed CC BY-SA 4.0.")


def fetch_mmfi(original: Path, frames_per_activity: int = 30) -> str:
    """Range-extract a real WiFi sequence for every MM-Fi activity A01-A27."""
    remote = zipfile.ZipFile(_RemoteFile(MMFI_URL, MMFI_BYTES))  # type: ignore[arg-type]
    selected: Dict[str, List[zipfile.ZipInfo]] = {}
    for index in range(1, 28):
        activity = f"A{index:02d}"
        names = [f"E01/S01/{activity}/wifi-csi/frame{frame:03d}.mat" for frame in range(1, frames_per_activity + 1)]
        try:
            selected[activity] = [remote.getinfo(name) for name in names]
        except KeyError as exc:
            raise RuntimeError(f"MM-Fi coherent S01 sequence is missing {exc.args[0]}") from exc
    members = [item.filename for activity in sorted(selected) for item in selected[activity]]
    selective_extract_from_handle(remote, members, original)
    write_official_manifest(original, {
        "dataset_id": "mm-fi",
        "source": "https://ntu-aiot-lab.github.io/mm-fi",
        "source_archive": MMFI_URL,
        "selected_members": {key: [item.filename for item in values] for key, values in selected.items()},
        "labels": sorted(selected),
        "frames_per_activity": frames_per_activity,
        "license": "CC BY-NC 4.0",
        "redistribution": "noncommercial redistribution with attribution",
    })
    return (f"Authentic {frames_per_activity}-frame WiFi sequences for all 27 MM-Fi "
            "activities, fetched directly from the public official archive. Licensed CC BY-NC 4.0.")


def fetch_ntu_fi(original: Path) -> str:
    """Fetch one official SenseFi clip for all HAR and Human-ID labels."""
    har_members = {
        "box": "NTU-Fi_HAR/train_amp/box/box143.mat",
        "circle": "NTU-Fi_HAR/train_amp/circle/circle59.mat",
        "clean": "NTU-Fi_HAR/train_amp/clean/clean88.mat",
        "fall": "NTU-Fi_HAR/test_amp/fall/fall168.mat",
        "run": "NTU-Fi_HAR/train_amp/run/run32.mat",
        "walk": "NTU-Fi_HAR/train_amp/walk/walk101.mat",
    }
    identity_members = {
        "001": "NTU-Fi-HumanID/test_amp/001/a0.mat",
        "002": "NTU-Fi-HumanID/test_amp/002/b12.mat",
        "003": "NTU-Fi-HumanID/train_amp/003/b16.mat",
        "004": "NTU-Fi-HumanID/test_amp/004/c8.mat",
        "005": "NTU-Fi-HumanID/test_amp/005/a0.mat",
        "006": "NTU-Fi-HumanID/test_amp/006/c0.mat",
        "007": "NTU-Fi-HumanID/train_amp/007/b19.mat",
        "008": "NTU-Fi-HumanID/test_amp/008/b6.mat",
        "009": "NTU-Fi-HumanID/test_amp/009/c12.mat",
        "010": "NTU-Fi-HumanID/train_amp/010/a13.mat",
        "011": "NTU-Fi-HumanID/test_amp/011/c8.mat",
        "012": "NTU-Fi-HumanID/test_amp/012/c1.mat",
        "013": "NTU-Fi-HumanID/train_amp/013/c16.mat",
        "015": "NTU-Fi-HumanID/train_amp/015/c18.mat",
    }
    har = zipfile.ZipFile(_RemoteFile(NTU_HAR_URL, NTU_HAR_BYTES))  # type: ignore[arg-type]
    identity = zipfile.ZipFile(_RemoteFile(NTU_ID_URL, NTU_ID_BYTES))  # type: ignore[arg-type]
    selective_extract_from_handle(har, list(har_members.values()), original)
    selective_extract_from_handle(identity, list(identity_members.values()), original)
    write_official_manifest(original, {
        "dataset_id": "ntu-fi",
        "source": "https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark",
        "source_archives": [NTU_HAR_URL, NTU_ID_URL],
        "selected_members": {"Activity": har_members, "Identity": identity_members},
        "license": "Dataset license not stated in the SenseFi release",
        "redistribution": "official-source fetch only",
    })
    return ("Twenty authentic NTU-Fi clips: all six HAR activities and all fourteen "
            "released identity folders, fetched from the public official links. Raw files remain local.")


def fetch_ut_har(original: Path) -> str:
    """Fetch the official validation split and retain one real row per class."""
    import numpy as np

    remote = zipfile.ZipFile(_RemoteFile(UT_HAR_URL, UT_HAR_BYTES))  # type: ignore[arg-type]
    data_member = "UT_HAR/data/X_val.csv"
    label_member = "UT_HAR/label/y_val.csv"
    cache_root = CACHE / "ut-har"
    extracted = selective_extract_from_handle(remote, [data_member, label_member], cache_root)
    data_path = next(path for path in extracted if path.name.startswith("X_"))
    label_path = next(path for path in extracted if path.name.startswith("y_"))
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False).reshape(-1)
    selected_indices = [int(np.flatnonzero(labels == value)[0]) for value in range(7)]
    selected_data = np.asarray(data[selected_indices], dtype=np.float32)
    selected_labels = np.asarray(labels[selected_indices], dtype=np.int64)
    target_data = original / data_member
    target_label = original / label_member
    target_data.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    with target_data.open("wb") as handle:
        np.save(handle, selected_data)
    with target_label.open("wb") as handle:
        np.save(handle, selected_labels)
    write_official_manifest(original, {
        "dataset_id": "ut-har",
        "source": "https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark",
        "source_archive": UT_HAR_URL,
        "source_members": [data_member, label_member],
        "selected_source_rows": selected_indices,
        "labels": ["lie down", "fall", "walk", "pickup", "run", "sit down", "stand up"],
        "license": "Dataset license not stated",
        "redistribution": "official-source fetch only",
    })
    return ("Seven authentic UT-HAR validation clips, one for every class, fetched from "
            "the public official SenseFi link. Raw files remain local.")


def fetch_widar3(original: Path) -> str:
    """Range-extract one authentic BVP recording for every Widar gesture."""
    members = [
        "Widardata/train/1-Push&Pull/user2-1-5-4-8-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/2-Sweep/user16-2-3-5-4-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/3-Clap/user3-3-2-4-10-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/4-Slide/user3-4-2-4-7-1-1e-07-100-20-100000-L0.csv",
        "Widardata/test/5-Draw-N(H)/user17-7-2-4-1-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/6-Draw-O(H)/user17-5-2-4-5-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/7-Draw-Rectangle(H)/user3-6-2-2-8-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/8-Draw-Triangle(H)/user2-5-2-2-3-1-1e-07-100-20-100000-L0.csv",
        "Widardata/test/9-Draw-Zigzag(H)/user17-6-2-2-5-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/10-Draw-Zigzag(V)/user3-5-5-4-7-1-1e-07-100-20-100000-L0.csv",
        "Widardata/test/11-Draw-N(V)/user2-6-4-2-13-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/12-Draw-O(V)/user2-4-3-2-3-1-1e-07-100-20-100000-L0(1).csv",
        "Widardata/train/13-Draw-1/user2-1-2-1-2-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/14-Draw-2/user2-2-1-2-5-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/15-Draw-3/user2-3-5-1-2-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/16-Draw-4/user1-4-2-4-2-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/17-Draw-5/user1-5-2-2-1-1-1e-07-100-20-100000-L0.csv",
        "Widardata/test/18-Draw-6/user1-6-4-4-2-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/19-Draw-7/user1-7-2-4-3-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/20-Draw-8/user2-8-2-4-6-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/21-Draw-9/user1-9-2-2-10-1-1e-07-100-20-100000-L0.csv",
        "Widardata/train/22-Draw-10/user1-10-5-4-5-1-1e-07-100-20-100000-L0.csv",
    ]
    remote = zipfile.ZipFile(_RemoteFile(WIDAR_BVP_URL, WIDAR_BVP_BYTES))  # type: ignore[arg-type]
    selective_extract_from_handle(remote, members, original)
    labels = [re.sub(r"^\d+-", "", Path(member).parent.name) for member in members]
    write_official_manifest(original, {
        "dataset_id": "widar3",
        "source": "https://ieee-dataport.org/open-access/widar-30-zero-effort-cross-domain-gesture-recognition-wi-fi",
        "source_archive": WIDAR_BVP_URL,
        "selected_members": dict(zip(labels, members)),
        "representation": "processed body-velocity profile (not raw CSI)",
        "license": "CC BY 4.0",
    })
    return "Twenty-two authentic Widar BVP clips, one for every released gesture. Licensed CC BY 4.0."


def fetch_wifi_tad(original: Path) -> str:
    """Extract two authentic continuous sequences covering all seven labels."""
    members = [
        "dataset/annotations/Class Index_Detection.txt",
        "dataset/annotations/val_Annotation_ours.csv",
        "dataset/annotations/val_video_info.csv",
        "dataset/annotations/test_Annotation_ours.csv",
        "dataset/annotations/test_video_info.csv",
        "dataset/annotations/gt.json",
        "dataset/smartwifi/validation_npy/5_sjz_0.npy",
        "dataset/smartwifi/validation_npy/1.dat.npy",
    ]
    remote = zipfile.ZipFile(_RemoteFile(WIFI_TAD_URL, WIFI_TAD_BYTES))  # type: ignore[arg-type]
    selective_extract_from_handle(remote, members, original)
    write_official_manifest(original, {
        "dataset_id": "wifi-tad",
        "source": "https://github.com/aiotgroup/WiFiTAD",
        "source_archive": WIFI_TAD_URL,
        "selected_members": members,
        "labels": ["run", "walk", "jump", "wave", "bend", "stand", "sit"],
        "sample_rate_hz": 100,
        "representation": "processed 30-channel temporal features",
        "license": "Dataset license not stated",
        "redistribution": "official-source fetch only",
    })
    return ("Two authentic WiFiTAD continuous feature sequences whose official interval "
            "annotations cover all seven activities. Raw files stay local because the data license is unstated.")


def fetch_wiar(original: Path) -> str:
    """Fetch one authentic Intel 5300 recording per WiAR activity."""
    trials = {activity: 1 for activity in range(1, 17)}
    trials.update({4: 2, 11: 3})
    members = []
    for activity, trial in trials.items():
        filename = f"csi_a{activity}_{trial}.dat"
        relative = Path("distance_factor_activity_data") / "1_alldata" / filename
        url = f"https://raw.githubusercontent.com/linteresa/WiAR/master/{relative.as_posix()}"
        download(url, original / relative)
        members.append(relative.as_posix())
    write_official_manifest(original, {
        "dataset_id": "wiar",
        "source": "https://github.com/linteresa/WiAR",
        "selected_members": members,
        "label_coverage": list(range(1, 17)),
        "setting": {"distance_m": 1},
        "nominal_sample_rate_hz": 30,
        "license": "Dataset license not stated",
        "redistribution": "official-source fetch only",
    })
    return ("Sixteen authentic WiAR Intel 5300 captures, one for every activity. Raw files "
            "stay local because the official repository does not state a dataset license.")


def fetch_glasgow_activity(original: Path) -> str:
    """Select one real CSV for every joint activity/location class deposited."""
    selected: Dict[str, tuple[zipfile.ZipFile, zipfile.ZipInfo]] = {}
    remotes: List[tuple[int, str, int, zipfile.ZipFile]] = []
    activity_pattern = re.compile(
        r"^(EmptyRoom|NoActivity|Sitting|Standing|Leaning|WalkingRxTx|WalkingTxRx)",
        re.IGNORECASE,
    )
    for location, url, size in GLASGOW_ACTIVITY_ARCHIVES:
        remote = zipfile.ZipFile(_RemoteFile(url, size))  # type: ignore[arg-type]
        remotes.append((location, url, size, remote))
        for info in remote.infolist():
            if not info.filename.lower().endswith(".csv") or info.file_size <= 0:
                continue
            stem = re.sub(r"_\d+$", "", Path(info.filename).stem)
            activity_match = activity_pattern.match(stem)
            if not activity_match:
                continue
            raw_activity = activity_match.group(1)
            activity = {
                "emptyroom": "EmptyRoom", "noactivity": "NoActivity", "sitting": "Sitting",
                "standing": "Standing", "leaning": "Leaning",
                "walkingrxtx": "WalkingRxTx", "walkingtxrx": "WalkingTxRx",
            }[raw_activity.lower()]
            location_match = re.search(r"L(\d+)", stem, re.IGNORECASE)
            zone_match = re.search(r"Z(\d+)", stem, re.IGNORECASE)
            source_location = int(location_match.group(1)) if location_match else location
            suffix = f"L{source_location}" + (f"Z{zone_match.group(1)}" if zone_match else "")
            # Empty-room is one activity class without a subject zone.  The
            # release repeats it once per location archive; keep one authentic
            # representative instead of inflating the 37 semantic joint
            # classes to 39 by treating those duplicate room baselines as new
            # labels.
            key = "EmptyRoom/global" if activity == "EmptyRoom" else f"{activity}/{suffix}"
            current = selected.get(key)
            if current is None or info.file_size < current[1].file_size:
                selected[key] = (remote, info)
    if len(selected) != 37:
        raise RuntimeError(f"expected 37 joint classes deposited by Glasgow, found {len(selected)}: {sorted(selected)}")
    for remote, info in selected.values():
        selective_extract_from_handle(remote, [info.filename], original)
    missing_upstream = [
        "Sitting/L1Z1", "Sitting/L1Z2", "Sitting/L1Z3",
        "NoActivity/L1Z2", "NoActivity/L1Z3", "Standing/L1Z1",
    ]
    write_official_manifest(original, {
        "dataset_id": "glasgow-activity-localization",
        "source": "https://researchdata.gla.ac.uk/1283/",
        "source_archives": [url for _location, url, _size in GLASGOW_ACTIVITY_ARCHIVES],
        "selected_members": {key: info.filename for key, (_remote, info) in selected.items()},
        "available_joint_classes": len(selected),
        "documented_joint_classes": 43,
        "missing_from_official_archives": missing_upstream,
        "license": "CC BY 4.0",
    })
    return ("Thirty-seven authentic recordings covering every activity/location class "
            "actually present in the three official archives. Six documented Location-1 "
            "classes are absent upstream and are reported, not fabricated. Licensed CC BY 4.0.")


def fetch_glasgow_multiuser(original: Path) -> str:
    """Extract one authentic CSV for every occupancy/activity combination."""
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("Glasgow multi-user extraction requires: pip install -e '.[data]'") from exc
    source = download(
        GLASGOW_MULTIUSER_URL,
        CACHE / "glasgow-multiuser" / "official-release.7z",
        GLASGOW_MULTIUSER_BYTES,
    )
    with py7zr.SevenZipFile(source, "r") as archive:
        infos = [item for item in archive.list() if not item.is_directory and item.filename.lower().endswith(".csv")]
        groups: Dict[str, list] = {}
        for info in infos:
            parent = Path(info.filename).parent.name
            groups.setdefault(parent, []).append(info)
        chosen = {name: min(items, key=lambda item: item.uncompressed) for name, items in groups.items()}
        if len(chosen) != 16:
            raise RuntimeError(f"expected 16 Glasgow multi-user settings, found {len(chosen)}")
        archive.extract(path=original, targets=[item.filename for item in chosen.values()])
    write_official_manifest(original, {
        "dataset_id": "glasgow-multiuser",
        "source": "https://researchdata.gla.ac.uk/1151/",
        "source_archive": GLASGOW_MULTIUSER_URL,
        "selected_members": {name: item.filename for name, item in chosen.items()},
        "license": "CC BY 4.0",
    })
    return "Sixteen authentic recordings, one for every released occupancy/activity combination in the official Glasgow release. Licensed CC BY 4.0."


def fetch_wifi_presence(original: Path, max_records: int = 300) -> str:
    """Stream real packets from each annotated state in official recording 260-4."""
    annotation_path = download(PRESENCE_ANNOTATIONS_URL, original / "annotations.csv")
    intervals = {
        "Approach": (1532363110.154, 1532363138.677),
        "Enter": (1532363138.721, 1532363144.104),
        "Mobile": (1532363144.145, 1532363157.083),
        "Stationary": (1532363157.091, 1532363459.722),
        "Exit": (1532364682.393, 1532364699.398),
        "Departure": (1532364699.412, 1532364710.484),
        "Gone": (1532364710.494, 1532365080.253),
    }
    targets = {
        label: original / "segments" / label.lower() / "260-4.csi.json.gz"
        for label in intervals
    }
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    sinks = {label: gzip.open(target, "wb") for label, target in targets.items()}
    counts = {label: 0 for label in intervals}
    request = urllib.request.Request(PRESENCE_URL, headers={"User-Agent": "wisensehub-sample-fetcher"})
    try:
        with urllib.request.urlopen(request) as response, gzip.GzipFile(fileobj=response, mode="rb") as source:
            for line in source:
                if not line.strip():
                    continue
                timestamp_match = re.search(rb'"t"\s*:\s*([0-9.]+)', line[:160])
                if not timestamp_match:
                    continue
                timestamp = float(timestamp_match.group(1))
                for label, (start, end) in intervals.items():
                    if counts[label] < max_records and start <= timestamp <= end:
                        sinks[label].write(line)
                        counts[label] += 1
                        break
                if all(count >= max_records for count in counts.values()):
                    break
                if timestamp > max(end for _start, end in intervals.values()):
                    break
    finally:
        for sink in sinks.values():
            sink.close()
    missing = {label: count for label, count in counts.items() if count == 0}
    if missing:
        raise RuntimeError(f"official presence/movement stream had no packets for annotated states: {missing}")
    write_official_manifest(original, {
        "dataset_id": "wifi-presence-movement",
        "source": "https://zenodo.org/records/3676058",
        "source_file": "260-4.csi.json.gz",
        "annotation_file": annotation_path.name,
        "selected_intervals": {label: {"start": start, "end": end, "records": counts[label]}
                               for label, (start, end) in intervals.items()},
        "license": "CC BY 4.0",
    })
    return ("Authentic interval-aligned CSI from all seven official states—Gone, Approach, "
            "Enter, Mobile, Stationary, Exit, and Departure—with the official annotations. "
            "Licensed CC BY 4.0.")


def fetch_wireless_har(original: Path) -> str:
    """Extract every Room 2 activity plus the release's room summaries.

    Rooms 1 and 3 are stored as very large continuous participant tables,
    while Room 2 provides the official per-activity clips needed for a compact,
    complete label demo. Do not pretend those layouts are interchangeable.
    """
    remote = zipfile.ZipFile(_RemoteFile(WIRELESS_HAR_URL, WIRELESS_HAR_BYTES))  # type: ignore[arg-type]
    groups: Dict[str, List[zipfile.ZipInfo]] = {}
    summaries: List[zipfile.ZipInfo] = []
    for info in remote.infolist():
        parts = Path(info.filename).parts
        if "WiFi_CSI" not in parts:
            continue
        wifi_index = parts.index("WiFi_CSI")
        if len(parts) <= wifi_index + 2:
            continue
        room = parts[wifi_index + 1]
        child = parts[wifi_index + 2]
        if room not in {"Room_1", "Room_2", "Room_3"}:
            continue
        if child == "activity_distribution.csv":
            summaries.append(info)
        elif room == "Room_2" and info.filename.lower().endswith(".mat") and len(parts) > wifi_index + 3:
            groups.setdefault(parts[wifi_index + 2].lower(), []).append(info)
    chosen = {name: min(items, key=lambda item: item.file_size) for name, items in groups.items()}
    expected = {"kneel", "liedown", "pickup", "sit", "sitrotate", "stand", "standrotate", "walk"}
    if set(chosen) != expected:
        raise RuntimeError(f"unexpected Wireless HAR Room 2 activity set: {sorted(chosen)}")
    if {Path(info.filename).parent.name for info in summaries} != {"Room_1", "Room_3"}:
        raise RuntimeError("Wireless HAR archive is missing a Room 1 or Room 3 activity summary")
    selected = [info.filename for info in chosen.values()] + [info.filename for info in summaries]
    selective_extract_from_handle(remote, selected, original)
    write_official_manifest(original, {
        "dataset_id": "wireless-har-wifi-uwb",
        "source": "https://figshare.com/articles/dataset/20444538",
        "source_archive": WIRELESS_HAR_URL,
        "sample_scope": "WiFi CSI only; full release also contains UWB CIR",
        "selected_members": {f"Room_2/{name}": info.filename for name, info in chosen.items()},
        "room_summaries": [info.filename for info in summaries],
        "license": "CC BY 4.0",
    })
    return ("Eight authentic Room 2 WiFi CSI recordings, one per released activity label, plus "
            "the official Room 1 and Room 3 activity summaries. The compact sample is WiFi-only; "
            "the full release also contains UWB CIR. Licensed CC BY 4.0.")


def _kaggle_download_file(dataset: str, remote_path: str, target: Path) -> None:
    kaggle = ROOT.parent / ".venv" / "bin" / "kaggle"
    if not kaggle.exists():
        raise RuntimeError("configured Kaggle CLI not found at ../.venv/bin/kaggle")
    with tempfile.TemporaryDirectory(prefix="wisensehub-kaggle-") as scratch_name:
        scratch = Path(scratch_name)
        subprocess.run(
            [str(kaggle), "datasets", "download", dataset, "-f", remote_path, "-p", str(scratch)],
            check=True, capture_output=True, text=True,
        )
        payloads = [path for path in scratch.iterdir() if path.is_file()]
        if not payloads:
            raise RuntimeError(f"Kaggle returned no payload for {remote_path}")
        payload = payloads[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if payload.suffix.lower() == ".zip":
            with zipfile.ZipFile(payload) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                selected = next((item for item in members if item.filename == remote_path), members[0])
                with archive.open(selected) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        else:
            shutil.move(str(payload), target)


def fetch_wimans(original: Path) -> str:
    """Download real CSI covering every WiMANS label and setting marginal."""
    _kaggle_download_file("shuokanghuang/wimans", "annotation.csv", original / "annotation.csv")
    members = []
    for index in range(1, 10):
        remote_path = f"wifi_csi/amp/act_1_{index}.npy"
        _kaggle_download_file("shuokanghuang/wimans", remote_path, original / remote_path)
        members.append(remote_path)
    for label in ("act_67_10", "act_79_1", "act_10_1", "act_16_1", "act_201_1"):
        remote_path = f"wifi_csi/amp/{label}.npy"
        _kaggle_download_file("shuokanghuang/wimans", remote_path, original / remote_path)
        members.append(remote_path)
    write_official_manifest(original, {
        "dataset_id": "wimans",
        "source": "https://www.kaggle.com/datasets/shuokanghuang/wimans",
        "selected_members": members,
        "labels": ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"],
        "setting_coverage": {
            "environment": ["classroom", "meeting_room", "empty_room"],
            "band_ghz": ["2.4", "5"],
            "occupancy": [0, 1, 2, 3, 4, 5],
            "location": ["a", "b", "c", "d", "e"],
        },
        "license": "CC BY-NC-SA 4.0",
    })
    return ("Fourteen authentic WiMANS CSI amplitude clips: all nine activities plus five "
            "clips selected to cover all environments, bands, occupancy counts, and locations, "
            "joined to the official annotation table. Licensed CC BY-NC-SA 4.0.")


def fetch_xrf_v2(original: Path) -> str:
    """Fetch seven authentic continuous WiFi sequences covering all 34 raw labels."""
    dataset = "anonymous20251/xrfv2dataset"
    stems = ("0_3_3", "3_2_6", "6_1_1", "3_3_14", "0_2_20", "3_1_13", "15_1_7")
    members = [f"XRFV2_processed/WWADL/wifi/{stem}.h5" for stem in stems]
    metadata_members = [
        "XRFV2/XRFV2/wifi_annotations.json", "XRFV2/XRFV2/train.csv",
        "XRFV2/XRFV2/test.csv", "XRFV2/XRFV2/info.json",
    ]
    for remote_path in [*members, *metadata_members]:
        _kaggle_download_file(dataset, remote_path, original / remote_path)
    import h5py
    import numpy as np
    label_ids: set[int] = set()
    shapes = {}
    for member in members:
        with h5py.File(original / member, "r") as handle:
            label_ids.update(int(value) for value in np.asarray(handle["label"])[:, 1])
            shapes[member] = list(handle["amp"].shape)
    if label_ids != set(range(34)):
        raise RuntimeError(f"XRF V2 representative sequences cover {sorted(label_ids)}, expected 0..33")
    write_official_manifest(original, {
        "dataset_id": "xrf-v2",
        "source": "https://github.com/airslab2020/XRFV2",
        "source_dataset": "https://www.kaggle.com/datasets/anonymous20251/xrfv2dataset",
        "selected_members": members, "metadata_members": metadata_members,
        "source_shapes": shapes, "label_coverage": list(range(34)),
        "selection_coverage": {
            "subjects": [0, 3, 6, 15],
            "scenes": {"1": "dining room", "2": "study room", "3": "bedroom"},
            "raw_activity_labels": 34,
        },
        "sample_rate_hz": 50, "license": "MIT",
    })
    return ("Seven authentic XRFV2 continuous WiFi sequences whose official interval tables "
            "cover all 34 raw action labels. The derived benchmark merges five walking "
            "destinations into one class and therefore has 30 classes. Licensed MIT.")


def fetch_operanet(original: Path, max_rows: int = 3000) -> str:
    """Extract one contiguous, authentic CSI excerpt for every HAR label."""
    from scipy.io import savemat

    remote = None
    cache_root = CACHE / "operanet"
    cache_root.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, dict] = {}
    source_files: Dict[str, str] = {}

    for experiment, expected_labels in OPERANET_ACTIVITY_EXPERIMENTS.items():
        source_name = f"wificsi1_exp{experiment}.mat"
        cached = cache_root / source_name
        if cached.exists():
            log(f"cache hit: {cached.relative_to(ROOT)}")
        else:
            if remote is None:
                remote = zipfile.ZipFile(  # type: ignore[arg-type]
                    _RemoteFile(OPERANET_WIFI1_URL, OPERANET_WIFI1_BYTES)
                )
            info = remote.getinfo(source_name)
            log(f"extracting {source_name} ({info.file_size} bytes) from official OPERAnet archive")
            with remote.open(info) as source, cached.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
        source_files[experiment] = source_name
        runs = (
            {"background": [(0, _operanet_v73_row_count(cached))]}
            if experiment == "019"
            else _operanet_v73_label_runs(cached, "activity")
        )
        for label in expected_labels:
            candidates = runs.get(label, [])
            if not candidates:
                raise RuntimeError(f"{source_name} has no released {label!r} packet interval")
            start, stop = max(candidates, key=lambda pair: pair[1] - pair[0])
            stop = min(stop, start + max_rows)
            columns = _operanet_v73_csi_columns(
                cached,
                max_rows=stop - start,
                start_row=start,
                include_metadata=experiment != "019",
            )
            if "timestamp" in columns:
                columns["timestamp"] = _operanet_elapsed_milliseconds(columns["timestamp"])
            if experiment == "019":
                # The empty-room file stores one MATLAB char object per packet,
                # unlike the compact string vectors used by experiments 2–5.
                # Table 2 is the authoritative annotation for the whole file.
                import numpy as np
                columns["activity"] = np.full(stop - start, "background", dtype="U16")
                columns["exp_no"] = np.full(stop - start, "exp_019", dtype="U16")
                columns["room_no"] = np.full(stop - start, "1", dtype="U4")
            target = original / "WiFi" / "Activity" / label / (
                f"{Path(source_name).stem}__rows_{start}_{stop}.mat"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            savemat(target, {"wifi_csi": columns}, do_compression=True)
            selected[label] = {
                "source_file": source_name,
                "source_rows": [start, stop],
                "packets": stop - start,
                "person_id": sorted(set(map(str, columns.get("person_id", [])))),
                "room_no": sorted(set(map(str, columns.get("room_no", [])))),
            }

    manifest = {
        "schema_version": "1.0",
        "dataset_id": "operanet",
        "kind": "official-mini-sample",
        "source": "https://doi.org/10.6084/m9.figshare.c.5551209",
        "source_archive": OPERANET_WIFI1_URL,
        "source_files": source_files,
        "selected_labels": selected,
        "label_coverage": sorted(selected),
        "selected_csi_columns": 270,
        "license": "CC0",
        "attribution": "OPERAnet, Bocus et al., Figshare collection 5551209",
    }
    (original / "official-sample-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return ("Eight authentic, contiguous OPERAnet CSI excerpts from one receiver, covering "
            "background, noactivity, and all six released HAR activities. Figshare marks the data CC0.")


def _operanet_v73_table_cells(handle) -> tuple[List[str], object, object]:
    """Locate the released MATLAB table's variable-name and value cells."""
    import h5py
    import numpy as np

    def decode_char(reference) -> str:
        value = np.asarray(handle[reference]).reshape(-1)
        if value.dtype.kind not in "ui":
            return ""
        return "".join(chr(int(item)) for item in value if int(item))

    # MATLAB records the table's value and name cells in the small MCOS
    # reference vector.  Looking there directly is crucial for background
    # recordings, whose #refs# tree contains more than half a million tiny
    # per-row char datasets and would take minutes to traverse recursively.
    cells = []
    if "#subsystem#/MCOS" in handle:
        mcos_refs = np.asarray(handle["#subsystem#/MCOS"]).reshape(-1)
        for reference in mcos_refs:
            obj = handle[reference]
            if (
                isinstance(obj, h5py.Dataset)
                and obj.dtype.kind == "O"
                and 270 <= obj.size <= 400
            ):
                cells.append(obj)
    else:
        # Focused unit fixtures use the same cell contract without MATLAB's
        # outer MCOS wrapper.  They are tiny, so a normal traversal is safe.
        handle.visititems(
            lambda _name, obj: cells.append(obj)
            if isinstance(obj, h5py.Dataset)
            and obj.attrs.get("MATLAB_class") == np.bytes_(b"cell")
            else None
        )
    names = None
    names_cell = None
    for cell in cells:
        if cell.dtype.kind != "O":
            continue
        try:
            decoded = [decode_char(ref) for ref in np.asarray(cell).reshape(-1)]
        except (KeyError, TypeError, ValueError):
            continue
        if sum(bool(re.fullmatch(r"tx\d+rx\d+_sub\d+", item)) for item in decoded) == 270:
            names, names_cell = decoded, cell
            break
    if names is None or names_cell is None:
        raise RuntimeError("OPERAnet MATLAB table variable names were not found")

    csi_index = next(
        index for index, name in enumerate(names)
        if re.fullmatch(r"tx\d+rx\d+_sub\d+", name)
    )
    values_cell = None
    for cell in cells:
        if cell.name == names_cell.name or cell.size != names_cell.size:
            continue
        refs = np.asarray(cell).reshape(-1)
        candidate = handle[refs[csi_index]]
        if candidate.dtype.names == ("real", "imag"):
            values_cell = cell
            break
    if values_cell is None:
        raise RuntimeError("OPERAnet MATLAB table CSI values were not found")
    return names, names_cell, values_cell


def _operanet_v73_row_count(source: Path) -> int:
    """Read the packet count from the first released CSI table column."""
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        names, _names_cell, values_cell = _operanet_v73_table_cells(handle)
        index = next(
            value for value, name in enumerate(names)
            if re.fullmatch(r"tx\d+rx\d+_sub\d+", name)
        )
        reference = np.asarray(values_cell).reshape(-1)[index]
        return int(max(handle[reference].shape))


def _operanet_decode_mcos_strings(handle, reference, start_row: int = 0, stop_row: int | None = None):
    """Decode one MATLAB MCOS string column, optionally selecting a row range."""
    import numpy as np

    header = np.asarray(handle[reference]).reshape(-1)
    if header.size < 5 or int(header[0]) != 0xDD000000:
        raise ValueError("OPERAnet metadata column is not an MCOS string vector")
    instance = int(header[4])
    mcos_refs = np.asarray(handle["#subsystem#/MCOS"]).reshape(-1)
    serialized = np.asarray(handle[mcos_refs[instance]]).reshape(-1)
    count = int(serialized[2])
    stop = count if stop_row is None else min(count, int(stop_row))
    start = max(0, int(start_row))
    if stop < start:
        raise ValueError(f"invalid OPERAnet row range {start}:{stop}")
    lengths = serialized[4:4 + count].astype(np.int64, copy=False)
    offsets = np.empty(count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    packed = serialized[4 + count:].astype("<u8", copy=False).tobytes()
    characters = np.frombuffer(packed, dtype="<u2")
    base = int(offsets[start])
    selected = characters[base:int(offsets[stop])]
    result = np.empty(stop - start, dtype="U128")
    for output_index, row in enumerate(range(start, stop)):
        left = int(offsets[row]) - base
        right = int(offsets[row + 1]) - base
        result[output_index] = selected[left:right].tobytes().decode("utf-16le")
    return result


def _operanet_v73_csi_columns(
    source: Path, max_rows: int, start_row: int = 0, include_metadata: bool = False,
) -> Dict[str, object]:
    """Read CSI columns directly from an official MATLAB 7.3 table.

    OPERAnet stores the table as MATLAB's HDF5/MCOS representation. Reading the
    referenced column arrays directly avoids loading the 287 MB recording and,
    importantly, preserves the released complex measurements without inventing
    metadata or signal values.
    """
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        names, _names_cell, values_cell = _operanet_v73_table_cells(handle)

        csi_indices = [
            index for index, name in enumerate(names)
            if re.fullmatch(r"tx\d+rx\d+_sub\d+", name)
        ]
        refs = np.asarray(values_cell).reshape(-1)
        stop_row = start_row + max_rows
        columns: Dict[str, object] = {}
        for index in csi_indices:
            dataset = handle[refs[index]]
            if dataset.ndim != 2 or 1 not in dataset.shape:
                raise ValueError(f"unexpected OPERAnet CSI column shape {dataset.shape}")
            if dataset.shape[0] == 1:
                raw = np.asarray(dataset[:, start_row:stop_row]).reshape(-1)
            else:
                raw = np.asarray(dataset[start_row:stop_row, :]).reshape(-1)
            columns[names[index]] = raw["real"] + 1j * raw["imag"]
        if include_metadata:
            for name in ("timestamp", "activity", "exp_no", "person_id", "room_no"):
                if name not in names:
                    continue
                index = names.index(name)
                columns[name] = _operanet_decode_mcos_strings(
                    handle, refs[index], start_row=start_row, stop_row=stop_row
                )
        return columns


def _operanet_v73_label_runs(source: Path, column: str) -> Dict[str, List[tuple[int, int]]]:
    """Return all contiguous row intervals for one released string label column."""
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        names, _names_cell, values_cell = _operanet_v73_table_cells(handle)
        if column not in names:
            raise ValueError(f"OPERAnet table has no {column!r} column")
        refs = np.asarray(values_cell).reshape(-1)
        labels = _operanet_decode_mcos_strings(handle, refs[names.index(column)])
    runs: Dict[str, List[tuple[int, int]]] = {}
    if labels.size == 0:
        return runs
    start = 0
    current = str(labels[0])
    for index in range(1, labels.size):
        label = str(labels[index])
        if label == current:
            continue
        runs.setdefault(current, []).append((start, index))
        start, current = index, label
    runs.setdefault(current, []).append((start, int(labels.size)))
    return runs


def _operanet_elapsed_milliseconds(values: object):
    """Convert released HH:MM:SS.mmm strings to elapsed milliseconds."""
    import numpy as np

    result = []
    day_offset = 0.0
    previous = None
    for raw in np.asarray(values).reshape(-1):
        hours, minutes, seconds = str(raw).split(":")
        current = (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000.0
        if previous is not None and current + day_offset < previous:
            day_offset += 24 * 60 * 60 * 1000.0
        current += day_offset
        result.append(current)
        previous = current
    elapsed = np.asarray(result, dtype=np.float64)
    return elapsed - elapsed[0] if elapsed.size else elapsed


FETCHERS = {
    "aril": fetch_aril,
    "csida": fetch_csida,
    "csi-bench": fetch_csi_bench,
    "ehunam": fetch_ehunam,
    "exposing-csi": fetch_exposing_csi,
    "mm-fi": fetch_mmfi,
    "ntu-fi": fetch_ntu_fi,
    "ut-har": fetch_ut_har,
    "widar3": fetch_widar3,
    "wifi-tad": fetch_wifi_tad,
    "wiar": fetch_wiar,
    "glasgow-activity-localization": fetch_glasgow_activity,
    "glasgow-multiuser": fetch_glasgow_multiuser,
    "wallhack18k": fetch_wallhack,
    "nist-breathesmart": fetch_nist,
    "figshare-csi-har": fetch_figshare_csi_har,
    "operanet": fetch_operanet,
    "wifi-presence-movement": fetch_wifi_presence,
    "wimans": fetch_wimans,
    "xrf55": fetch_xrf55,
    "xrf-v2": fetch_xrf_v2,
    "wireless-har-wifi-uwb": fetch_wireless_har,
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
            if dataset_id in OFFICIAL_FETCH_ONLY:
                sample_dir = SITE_SAMPLES / dataset_id
                archive_path = SITE_SAMPLES / f"{dataset_id}.zip"
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
                if archive_path.exists():
                    archive_path.unlink()
                stats = {"files": sum(1 for path in original.rglob("*") if path.is_file()),
                         "bytes": sum(path.stat().st_size for path in original.rglob("*") if path.is_file())}
                report[dataset_id] = {
                    "status": "ok", "kind": "official-fetch-preview", "delivery": "official-fetch",
                    "note": note, **stats,
                }
                log(f"{dataset_id}: real local sample ready; raw redistribution disabled")
            else:
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
