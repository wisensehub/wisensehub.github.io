from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import validate_catalog
from .adapters.aril import convert_aril_mat
from .adapters.xrf_v2 import convert_xrf_v2_h5
from .download import download_dataset
from .quality import write_quality_report
from .prepare import prepare_dataset, registered_datasets
from .registry import load_adapter_registry, load_split_registry
from .standardize import standardize_csv
from .views import ViewOptions, TASK_PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wisensehub", description="WiFi sensing dataset catalog and standardizer")
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="Catalog operations")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_commands.add_parser("validate", help="Validate all dataset metadata")
    standardize = commands.add_parser("standardize", help="Convert canonical CSV input to WiSenseHub NPZ")
    standardize.add_argument("--input", type=Path, required=True)
    standardize.add_argument("--output", type=Path, required=True)
    standardize.add_argument("--dataset-id", required=True)
    standardize.add_argument("--sample-rate", type=float, default=100.0)
    standardize.add_argument("--duration", type=float)
    download = commands.add_parser("download", help="Download an official original release from a verified manifest")
    download.add_argument("dataset_id")
    download.add_argument("--output-root", type=Path, default=Path("data"))
    aril = commands.add_parser("convert-aril", help="Convert official ARIL MAT files")
    aril.add_argument("--input", type=Path, required=True)
    aril.add_argument("--output", type=Path, required=True)
    aril.add_argument("--split", choices=["train", "test"], required=True)
    quality = commands.add_parser("quality", help="Generate a JSON quality report for standardized NPZ")
    quality.add_argument("--input", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    xrf = commands.add_parser("convert-xrf-v2", help="Convert an official XRF V2 WiFi HDF5 sequence")
    xrf.add_argument("--input", type=Path, required=True)
    xrf.add_argument("--output", type=Path, required=True)
    xrf.add_argument("--receivers", type=int, nargs="*", choices=[0, 1, 2])
    settings = commands.add_parser("settings", help="List conversion and split settings for a dataset")
    settings.add_argument("dataset_id", choices=registered_datasets())
    prepare = commands.add_parser("prepare", help="Auto-discover and convert a dataset under data/<id>/original")
    prepare.add_argument("dataset_id", choices=registered_datasets())
    prepare.add_argument("--data-root", type=Path, default=Path("data"))
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--setting", help="Split setting; run 'wisensehub settings <id>' to list choices")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    prepare.add_argument("--holdout", nargs="+", help="Group values assigned to test for a cross-group setting")
    prepare.add_argument("--profile", choices=sorted(TASK_PROFILES),
                         help="Task-family view profile (default: catalog standardization.profile)")
    prepare.add_argument("--target-rate", type=float, help="Generate a derived view at this sample rate in Hz")
    prepare.add_argument("--duration", type=float, help="Generate a derived view with this duration in seconds")
    prepare.add_argument("--target-length", type=int, help="Generate a derived view with this exact time length")
    prepare.add_argument("--interpolation", choices=["none", "nearest", "linear"], default="linear")
    prepare.add_argument("--layout", choices=["canonical", "flat", "link-subcarrier"], default="canonical",
                         help="Derived-view tensor layout; canonical keeps named signal axes, flat/link-subcarrier flattens non-time axes")
    prepare.add_argument("--links", type=int, help="Legacy datasets: choose combined link count (default: keep native)")
    prepare.add_argument("--subcarriers", type=int, help="Choose subcarrier count (default: keep native)")
    prepare.add_argument("--tx-links", type=int, help="Choose Tx link count when the dataset has a tx_link axis (default: keep native)")
    prepare.add_argument("--rx-links", type=int, help="Choose Rx link count when the dataset has an rx_link axis (default: keep native)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "catalog":
        entries, errors = validate_catalog()
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"Validated {len(entries)} dataset entries.")
        return 0
    if args.command == "standardize":
        sidecar = standardize_csv(args.input, args.output, args.dataset_id, args.sample_rate, args.duration)
        print(f"Wrote {args.output} and {sidecar}")
        return 0
    if args.command == "download":
        files = download_dataset(args.dataset_id, args.output_root)
        print(f"Downloaded {len(files)} file(s) for {args.dataset_id}")
        return 0
    if args.command == "convert-aril":
        sidecar = convert_aril_mat(args.input, args.output, args.split)
        print(f"Wrote {args.output} and {sidecar}")
        return 0
    if args.command == "quality":
        write_quality_report(args.input, args.output)
        print(f"Wrote {args.output}")
        return 0
    if args.command == "convert-xrf-v2":
        sidecar = convert_xrf_v2_h5(args.input, args.output, args.receivers)
        print(f"Wrote {args.output} and {sidecar}")
        return 0
    if args.command == "settings":
        adapter = load_adapter_registry()[args.dataset_id]
        config = load_split_registry()[args.dataset_id]
        print(f"{args.dataset_id}: {adapter['handler']} adapter ({adapter['implementation']}); default setting={config['default']}")
        print(f"official reference: {adapter['official_reference']}")
        print(f"recognized source layouts: {', '.join(adapter['patterns'])}")
        for item in config["settings"]:
            group = f", group_by={item['group_by']}" if item.get("group_by") else ""
            print(f"- {item['id']}: {item['kind']} ({item['provenance']}{group})")
        return 0
    if args.command == "prepare":
        try:
            summary = prepare_dataset(
                args.dataset_id, args.data_root, args.limit, args.force,
                args.setting, args.seed, args.ratios, args.holdout,
                ViewOptions(
                    profile=args.profile,
                    target_rate_hz=args.target_rate,
                    duration_s=args.duration,
                    target_length=args.target_length,
                    interpolation=args.interpolation,
                    layout=args.layout,
                    links=args.links,
                    subcarriers=args.subcarriers,
                    tx_links=args.tx_links,
                    rx_links=args.rx_links,
                ),
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            return 1
        counts = summary["split"]["partition_counts"]
        print(f"Prepared {args.dataset_id}: {summary['converted']} converted, {summary['skipped']} skipped")
        view = summary.get("view_options") or {}
        if view:
            print(
                f"View profile={view.get('profile')} rate={view.get('target_rate_hz')} Hz "
                f"duration={view.get('duration_s')} s length={view.get('target_length')}"
            )
        window = summary.get("window_policy") or {}
        if window:
            print(
                f"Windows mode={window.get('mode')} length={window.get('selected_window_length')} "
                f"full_source_preserved={window.get('full_source_preserved')}"
            )
        print(f"Split {summary['split']['setting']} ({summary['split']['provenance']}): {counts}")
        return 0
    return 2
