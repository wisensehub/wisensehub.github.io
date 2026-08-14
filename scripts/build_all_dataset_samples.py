#!/usr/bin/env python3
"""Build real mini-samples and preprocessing plans for the website.

Only authentic released measurements are published by default. Deterministic
adapter fixtures remain available behind ``--allow-fixtures`` for local adapter
tests, but the public website build never substitutes them for real data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wifi_datahub.catalog import load_datasets  # noqa: E402
from wifi_datahub.prepare import prepare_dataset  # noqa: E402


DATA = ROOT / "data"
SITE_SAMPLES = ROOT / "site" / "samples"
REPORT_PATH = SITE_SAMPLES / "fetch-report.json"
PLAN_PATH = ROOT / "catalog" / "sample-plans.json"
OFFICIAL_SAMPLE_IDS = {
    "aril", "csi-bench", "csida", "ehunam", "exposing-csi",
    "figshare-csi-har", "glasgow-activity-localization", "glasgow-multiuser", "nist-breathesmart",
    "mm-fi", "ntu-fi", "operanet", "ut-har", "wallhack18k", "wiar", "widar3",
    "wifi-presence-movement", "wifi-tad", "wimans", "wireless-har-wifi-uwb",
    "xrf-v2", "xrf55",
}

# These public sources permit local use/fetching, but their release does not
# give WiSenseHub a clear right to mirror the raw subset.  Their real previews
# and standardized outputs are built locally; the website gives the exact
# official-source fetch command and never creates a downloadable raw archive.
LOCAL_ONLY_SAMPLE_IDS = {"aril", "ntu-fi", "ut-har", "wallhack18k", "wifi-tad", "wiar"}

REAL_SAMPLE_BLOCKERS = {
    "signfi": "The SignFi owner requires a Terms of Use application. WiSenseHub cannot accept that agreement or republish the data for you.",
    "wifi-80mhz": "The CC BY 4.0 IEEE DataPort release requires a signed-in subscription. Supply a lawfully downloaded source file before WiSenseHub can build its real previews.",
    "wipe-fall": "The CC BY 4.0 University of Glasgow release is supplied only through its Request Data form. After the owner sends it, keep the six official folders under data/wipe-fall/original/; WiSenseHub will prepare the real 51-subcarrier files locally.",
}


def signal(time: int, channels: int, *, offset: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, time, dtype=np.float32)[:, None]
    c = np.linspace(0, np.pi, channels, dtype=np.float32)[None, :]
    return (1.5 + np.sin(t + c) + 0.25 * np.cos(0.35 * t - c) + offset).astype(np.float32)


def save_mat(path: Path, payload: dict) -> None:
    from scipy.io import savemat
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload, do_compression=True)


def fixture_aril(root: Path) -> None:
    base = signal(192, 52)
    # One compact clip for every location, with all six gestures represented.
    # This lets the website demonstrate both ARIL targets instead of showing a
    # single anonymous heatmap.
    data = np.stack([
        (base + 0.25 * np.sin((index + 1) * np.linspace(0, np.pi, 192))[:, None]).T
        for index in range(16)
    ])
    save_mat(root / "train_data_split_amp.mat", {
        "train_data": data,
        "train_activity_label": (np.arange(16, dtype=np.int16) % 6)[:, None],
        "train_location_label": np.arange(16, dtype=np.int16)[:, None],
    })


def fixture_csida(root: Path) -> None:
    import zarr
    shape = (6, 1800, 3, 114)
    base = signal(1800, 114)
    clips = []
    for gesture in range(6):
        receivers = np.stack([
            base + 0.12 * rx + 0.22 * np.sin(
                (gesture + 1) * np.linspace(0, 2 * np.pi, 1800, dtype=np.float32)
            )[:, None]
            for rx in range(3)
        ], axis=1)
        clips.append(receivers)
    zarr.save(str(root / "csi_data_amp"), np.stack(clips).astype(np.float32))
    zarr.save(str(root / "csi_data_pha"), np.zeros(shape, dtype=np.float32))
    for name, values in {
        "csi_label_act": [0, 1, 2, 3, 4, 5], "csi_label_env": [0, 1, 0, 1, 0, 1],
        "csi_label_loc": [0, 1, 2, 0, 1, 2], "csi_label_user": [0, 1, 2, 3, 4, 0],
    }.items():
        zarr.save(str(root / name), np.asarray(values))


def fixture_ehunam(root: Path) -> None:
    value = signal(80, 64).astype(np.complex64) * np.exp(1j * 0.1)
    save_mat(root / "MC1_01A_1_HAR_e_J_#_#_01.mat", {
        "CSI": value, "BW": [[20]], "Subcarriers": [[64]],
        "Environment": "Office", "Timestamp": np.full((80, 1), 10),
    })


def fixture_exposing(root: Path) -> None:
    value = signal(24, 2048).astype(np.complex64) * np.exp(1j * 0.2)
    save_mat(root / "subject_01" / "S1_A.mat", {"csi_buff": value})


def fixture_glasgow_activity(root: Path) -> None:
    path = root / "Zone_1" / "Walking" / "walk_zone1.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, signal(160, 52).T, delimiter=",")


def fixture_glasgow_multiuser(root: Path) -> None:
    path = root / "1_Subject_Sitting" / "1_Sitting_01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, signal(160, 52).T, delimiter=",")


def fixture_mmfi(root: Path) -> None:
    folder = root / "E01" / "S01" / "A01" / "wifi-csi"
    for index in range(24):
        frame = signal(1, 270, offset=index / 40).reshape(3, 3, 30)
        save_mat(folder / f"frame{index + 1:03d}.mat", {"CSIamp": frame})


def fixture_ntu(root: Path) -> None:
    value = signal(500, 342).T.reshape(3, 114, 500)
    save_mat(root / "train_amp" / "walk" / "sample_001.mat", {"CSIamp": value})


def fixture_operanet(root: Path) -> None:
    table = {
        f"tx{tx}rx{rx}_sub{sub}": signal(80, 1, offset=(tx + rx + sub) / 100).reshape(-1).astype(np.complex64)
        for tx in range(1, 4) for rx in range(1, 4) for sub in range(1, 31)
    }
    table.update(
        timestamp=np.arange(80) * 10,
        activity=np.asarray(["walk"] * 40 + ["sit"] * 40),
        person_id=np.asarray(["One"] * 80), room_no=np.asarray(["1"] * 80),
    )
    save_mat(root / "WiFi" / "exp001.mat", {"wificsi": table})


def fixture_signfi(root: Path) -> None:
    base = signal(200, 90).reshape(200, 30, 3)
    csi = np.stack([base, base + 0.4], axis=-1).astype(np.complex64) * np.exp(1j * 0.15)
    save_mat(root / "dataset_lab_276_dl.mat", {"csid_lab": csi, "label_lab": [[1], [2]]})


def fixture_ut_har(root: Path) -> None:
    # UT-HAR is a signed processed feature release; keep negative values so the
    # preview exercises that semantic contract instead of amplitude-only data.
    base = signal(250, 90) - 1.5
    path = root / "processed.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=np.stack([base, base + 0.3]), label=np.asarray([0, 1]))


def fixture_wiar(root: Path) -> None:
    path = root / "subject_01" / "walk" / "csi_a1_1.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = (30 * (1 * 1 * 8 * 2 + 3) + 7) // 8
    records = []
    for index in range(80):
        header = bytearray(20)
        header[0:4] = int(1_000_000 + index * 10_000).to_bytes(4, "little")
        header[8], header[9] = 1, 1
        header[16:18] = expected.to_bytes(2, "little")
        # A deterministic non-zero payload gives the real Intel 5300 parser an
        # informative synthetic shape demo instead of an all-black image.
        payload = bytes(((index * 17 + byte * 29 + 11) % 256) for byte in range(expected))
        body = bytes(header) + payload
        records.append(struct.pack(">H", len(body) + 1) + bytes([187]) + body)
    path.write_bytes(b"".join(records))


def fixture_widar(root: Path) -> None:
    path = root / "Widardata" / "train" / "user1" / "gesture1" / "sample.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, signal(22, 400), delimiter=",")


def fixture_wifi_80(root: Path) -> None:
    value = signal(160, 256).astype(np.complex64) * np.exp(1j * 0.25)
    save_mat(root / "subject_01" / "day_01" / "AR1a_W.mat", {"csi_buff": value})


def fixture_presence(root: Path) -> None:
    path = root / "room_01" / "G19-10.csi.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(80):
            packet = [[{"r": float(1 + np.sin(index / 8 + sub / 10 + link)), "i": float(link / 5)}
                       for link in range(3)] for sub in range(30)]
            handle.write(json.dumps({"t": 100.0 + index / 20, "csi": packet}) + "\n")


def fixture_wifi_tad(root: Path) -> None:
    source = root / "smartwifi" / "validation_npy" / "video_01.npy"
    source.parent.mkdir(parents=True, exist_ok=True)
    np.save(source, signal(180, 60).T)
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "val_video_info.csv").write_text(
        "name,fps,sample_fps,count,sample_count\nvideo_01,30,10,540,180\n", encoding="utf-8")
    (annotations / "val_Annotation_ours.csv").write_text(
        "name,x,class,start,end\nvideo_01,x,2,60,240\n", encoding="utf-8")


def fixture_wimans(root: Path) -> None:
    path = root / "wifi_csi" / "amp" / "walk_1_1.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, signal(160, 270).reshape(160, 3, 3, 30))


def fixture_wipe(root: Path) -> None:
    path = root / "subject_01" / "low_risk" / "trial_01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, signal(160, 52), delimiter=",")


def fixture_wireless(root: Path) -> None:
    path = root / "Wireless_sensing_human_activity_recognition" / "WiFi_CSI" / "Room_1" / "walking.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, signal(160, 52), delimiter=",")


def fixture_xrf_v2(root: Path) -> None:
    import h5py
    path = root / "train" / "subject_01" / "walk.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("amp", data=signal(160, 270).reshape(160, 3, 3, 30))
        handle.create_dataset("label", data=np.asarray(1, dtype=np.int16))


def fixture_xrf55(root: Path) -> None:
    path = root / "Raw_dataset" / "WiFi" / "walk_1_1.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, signal(160, 270).reshape(160, 3, 3, 30))


FIXTURES: dict[str, Callable[[Path], None]] = {
    "aril": fixture_aril,
    "csida": fixture_csida,
    "ehunam": fixture_ehunam,
    "exposing-csi": fixture_exposing,
    "glasgow-activity-localization": fixture_glasgow_activity,
    "glasgow-multiuser": fixture_glasgow_multiuser,
    "mm-fi": fixture_mmfi,
    "ntu-fi": fixture_ntu,
    "operanet": fixture_operanet,
    "signfi": fixture_signfi,
    "ut-har": fixture_ut_har,
    "wiar": fixture_wiar,
    "widar3": fixture_widar,
    "wifi-80mhz": fixture_wifi_80,
    "wifi-presence-movement": fixture_presence,
    "wifi-tad": fixture_wifi_tad,
    "wimans": fixture_wimans,
    "wipe-fall": fixture_wipe,
    "wireless-har-wifi-uwb": fixture_wireless,
    "xrf-v2": fixture_xrf_v2,
    "xrf55": fixture_xrf55,
}


def mirror_and_zip(dataset_id: str) -> dict[str, int]:
    original = DATA / dataset_id / "original"
    sample_dir = SITE_SAMPLES / dataset_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    shutil.copytree(original, sample_dir)
    archive_path = SITE_SAMPLES / f"{dataset_id}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sample_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(dataset_id) / path.relative_to(sample_dir))
    files = [path for path in sample_dir.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "zip_bytes": archive_path.stat().st_size,
    }


def remove_public_fixture(dataset_id: str) -> None:
    """Remove a previously published fixture without touching local test data."""
    sample_dir = SITE_SAMPLES / dataset_id
    archive = SITE_SAMPLES / f"{dataset_id}.zip"
    preview_dir = ROOT / "site" / "assets" / "previews" / dataset_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    if archive.exists():
        archive.unlink()
    if preview_dir.exists():
        shutil.rmtree(preview_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset IDs (default: all)")
    parser.add_argument("--force", action="store_true", help="Rebuild generated fixtures and outputs")
    parser.add_argument(
        "--allow-fixtures", action="store_true",
        help="Build synthetic adapter fixtures for local testing (never use for the public site)",
    )
    args = parser.parse_args()

    catalog = {item["id"]: item for item in load_datasets()}
    requested = args.datasets or sorted(catalog)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    plans: dict[str, dict] = {}

    for dataset_id in requested:
        dataset = catalog[dataset_id]
        original = DATA / dataset_id / "original"
        fixture_marker = original / ".wisensehub-fixture.json"
        official_ready = (
            dataset_id in OFFICIAL_SAMPLE_IDS
            and report.get(dataset_id, {}).get("status") == "ok"
            and original.exists()
            and not fixture_marker.exists()
        )
        fetch_info = report.get(dataset_id, {})
        delivery = "official-fetch" if dataset_id in LOCAL_ONLY_SAMPLE_IDS else fetch_info.get("delivery", "hosted-sample")
        kind = fetch_info.get("kind", "official-mini-sample") if official_ready else "generated-adapter-fixture"
        if not official_ready:
            # Access/terms-gated datasets must never fall back to generated
            # website signals, even when fixture generation is explicitly
            # enabled for adapter development.
            if not args.allow_fixtures or dataset_id in REAL_SAMPLE_BLOCKERS:
                remove_public_fixture(dataset_id)
                reason = REAL_SAMPLE_BLOCKERS.get(dataset_id, (
                    "A redistributable real mini-sample is not hosted yet. "
                    "Open the official release to access the recorded data."
                ))
                blocker_kind = {
                    "wifi-80mhz": "account-gated-source",
                    "signfi": "terms-gated-source",
                    "wipe-fall": "request-only-source",
                }.get(dataset_id, "real-sample-unavailable")
                report[dataset_id] = {
                    "status": "skipped",
                    "kind": blocker_kind,
                    "reason": reason,
                    "source_url": dataset["original"].get("download_page")
                    or dataset["original"]["landing_page"],
                    "license": dataset["original"].get("license"),
                    "redistribution": dataset["original"].get("redistribution"),
                }
                plans[dataset_id] = {
                    "sample_kind": blocker_kind,
                    "sample_note": reason,
                    "official_source": report[dataset_id]["source_url"],
                }
                print(f"{dataset_id}: no hosted real sample; synthetic demo removed")
                continue
            fixture = FIXTURES.get(dataset_id)
            if fixture is None:
                raise RuntimeError(f"no fixture builder registered for {dataset_id}")
            marker = fixture_marker
            if args.force and original.exists() and marker.exists():
                shutil.rmtree(original)
            if not marker.exists():
                original.mkdir(parents=True, exist_ok=True)
                fixture(original)
                marker.write_text(json.dumps({
                    "dataset_id": dataset_id,
                    "kind": kind,
                    "note": "Deterministic adapter fixture; not an official measurement.",
                }, indent=2) + "\n", encoding="utf-8")
        summary = prepare_dataset(dataset_id, DATA, force=args.force, setting="random")
        if delivery == "official-fetch":
            sample_dir = SITE_SAMPLES / dataset_id
            archive_path = SITE_SAMPLES / f"{dataset_id}.zip"
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            if archive_path.exists():
                archive_path.unlink()
            files = [path for path in original.rglob("*") if path.is_file()]
            stats = {"files": len(files), "bytes": sum(path.stat().st_size for path in files)}
        else:
            stats = mirror_and_zip(dataset_id)
        note = report.get(dataset_id, {}).get("note") if official_ready else (
            "Generated adapter fixture matching the documented source layout. "
            "It exercises preprocessing and visualization but is not official recorded data."
        )
        report[dataset_id] = {
            "status": "ok", "kind": kind, "delivery": delivery, "note": note,
            "source_url": dataset["original"]["landing_page"],
            "license": dataset["original"].get("license"),
            "redistribution": dataset["original"].get("redistribution"),
            **stats,
        }
        view = summary.get("view_options") or {}
        split = summary.get("split") or {}
        ready = int(summary.get("converted") or 0) + int(summary.get("skipped") or 0)
        plans[dataset_id] = {
            "sample_kind": kind,
            "sample_note": note,
            "profile": view.get("profile"),
            "target_rate_hz": view.get("target_rate_hz"),
            "duration_s": view.get("duration_s"),
            "target_length": view.get("target_length"),
            "interpolation": view.get("interpolation"),
            "layout": view.get("layout"),
            "window_mode": view.get("window_mode"),
            "window_policy": summary.get("window_policy"),
            "full_source_preserved": bool((summary.get("window_policy") or {}).get("full_source_preserved")),
            "split_setting": split.get("setting"),
            "split_provenance": split.get("provenance"),
            "processed": ready,
            "converted": summary.get("converted"),
            "skipped": summary.get("skipped"),
            "failed": summary.get("failed"),
            "command": (
                f"wisensehub prepare {dataset_id} --data-root data --setting random"
                if view.get("profile") == "task-auto"
                else f"wisensehub prepare {dataset_id} --data-root data --setting random --profile {view.get('profile')}"
            ),
            "outputs": {
                "native": f"data/{dataset_id}/standardized/",
                "views": f"data/{dataset_id}/standardized/views/",
                "manifest": f"data/{dataset_id}/prepare-manifest.json",
                "split": f"data/{dataset_id}/splits/random.json",
            },
        }
        print(f"{dataset_id}: {kind}, {ready} ready, {summary['failed']} failed")

    SITE_SAMPLES.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    existing_plans = {}
    if PLAN_PATH.exists():
        existing_plans = json.loads(PLAN_PATH.read_text(encoding="utf-8")).get("datasets", {})
    existing_plans.update(plans)
    PLAN_PATH.write_text(json.dumps({"schema_version": "1.0", "datasets": existing_plans}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
