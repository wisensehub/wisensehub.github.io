# WiSenseHub

[Live data hub](https://wisensehub.github.io/) ·
[GitHub repository](https://github.com/wisensehub/wisensehub.github.io)

WiSenseHub is an open, task-oriented data hub for WiFi sensing datasets. It
indexes original releases, documents experimental settings, and provides a
reproducible pipeline for converting heterogeneous CSI recordings into a
common representation.

> **Agent-ready.** Install the open WiSenseHub Agent Skill in Codex, Claude
> Code, Cursor, or another compatible agent, then prepare datasets with
> plain-language requests.

## What this repository provides

- A machine-readable catalog organized by sensing task, hardware, environment,
  scenario, and subject count.
- Links and checksums for original dataset releases without silently
  redistributing restricted data.
- A canonical CSI schema with explicit units, timestamps, sampling rates,
  missing-data masks, labels, and provenance.
- Reproducible conversion commands and a small synthetic example.
- A static website deployable directly to GitHub Pages.
- A public quickstart tutorial plus a companion Jupyter notebook for
  interactive inspection.
- A repository-owned agent skill for guided dataset preparation and validation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m wifi_datahub catalog validate
python scripts/build_site_data.py
python -m unittest discover -s tests
```

## Agent-ready workflow

Paste this one line into Codex, Claude Code, Cursor, or another Agent
Skills-compatible coding agent:

```text
Install the WiSenseHub skill by running npx skills add wisensehub/wisensehub.github.io --skill wisensehub -g, then help me prepare WiFi sensing data.
```

After installation, ask for the outcome you need:

- “Download the CSI-Bench sample, standardize it, and show me what changed.”
- “Prepare an activity-recognition split and summarize the labels.”
- “Inspect my CSI files, fix format problems, and visualize one sample.”

The agent handles setup, processing, and verification, then explains the result
in plain language.

## Unified dataset workflow

Place an official release under the dataset's `original/` directory:

```text
data/<dataset-id>/
├── original/       # files downloaded from the official source
├── standardized/   # generated NPZ tensors and JSON sidecars
├── reports/        # generated quality reports
├── splits/         # generated evaluation protocol manifests
└── prepare-manifest.json
```

List the dataset's supported protocols, then run one command:

```bash
python -m wifi_datahub settings widar3
python -m wifi_datahub prepare widar3 --setting cross_subject --holdout 3 --data-root data
python -m wifi_datahub prepare wallhack18k --setting cross_device --holdout pifa --data-root data
python -m wifi_datahub prepare xrf-v2 --setting official --data-root data
```

Use `--limit 5` for a quick trial and `--force` to rebuild existing outputs.
The manifest records every converted, skipped, or failed source file.

When a model needs fixed-size tensors, ask `prepare` to derive an additional
standard view while preserving the native standardized file:

```bash
python -m wifi_datahub prepare ut-har \
  --setting official \
  --data-root data \
  --target-length 128 \
  --layout link-subcarrier

python -m wifi_datahub prepare wifi-presence-movement \
  --setting random \
  --data-root data \
  --target-rate 100 \
  --duration 4 \
  --interpolation linear
```

Derived views are written to `data/<dataset-id>/standardized/views/`. Supported
view options are `--target-rate`, `--duration`, `--target-length`,
`--interpolation {none,nearest,linear}`, `--layout {canonical,flat,link-subcarrier}`,
`--links`, `--subcarriers`, `--tx-links`, and `--rx-links`. The native NPZ remains in `standardized/`; the
manifest records both paths.

All 25 catalog entries have a dataset-aware converter. Twenty-one adapters
follow an official loader or preprocessing reference; four follow a documented
release schema where no public preprocessing implementation was available.
Run `wisensehub settings <dataset-id>` to see the exact recognized layout and
reference. Keep the downloaded directory tree intact: extract/copy the complete
release into `original/` and run `prepare`—manual file renaming is not required.
See the [adapter matrix](docs/adapters.md) and [split settings](docs/splits.md).
The website also provides one concrete source-layout and conversion walkthrough
for every catalog dataset; these examples are maintained in
`catalog/examples.json` and validated with the rest of the catalog.
See [known limitations](docs/known-limitations.md) for the verification
maturity model and dataset licensing boundaries.

Run the standardization demonstration:

```bash
python examples/make_synthetic_input.py
python -m wifi_datahub standardize \
  --input examples/data/synthetic_csi.csv \
  --output examples/data/standardized/synthetic_csi.npz \
  --dataset-id synthetic-demo \
  --sample-rate 100 \
  --duration 4
```

Download an original release whose official files have a verified manifest:

```bash
python -m wifi_datahub download wallhack18k --output-root data
```

Restricted or account-gated datasets intentionally link to their official
landing pages instead of attempting to bypass access conditions.

Convert an official ARIL processed MAT split and inspect output quality:

```bash
python -m wifi_datahub convert-aril \
  --input data/aril/original/data/train_data_split_amp.mat \
  --output data/aril/standardized/train.npz \
  --split train
python -m wifi_datahub quality \
  --input data/aril/standardized/train.npz \
  --output data/aril/standardized/train.quality.json
```

Convert one official XRF V2 WiFi HDF5 sequence:

```bash
pip install -e ".[data]"
python -m wifi_datahub convert-xrf-v2 \
  --input data/xrf-v2/original/sequence.h5 \
  --output data/xrf-v2/standardized/sequence.npz
```

Outputs are compressed NumPy `.npz` tensors plus JSON provenance sidecars,
quality reports, and split manifests. Complex CSI, timestamps, relative power,
and labels are retained when the official source supplies enough information;
amplitude-only releases are not presented as calibrated complex CSI. See
[the standard](docs/standard.md) for the exact contract.

## Tutorial and notebook

The GitHub Pages tutorial is designed as the stable public walkthrough:

- Website: `site/tutorial.html`
- Embedded notebook viewer: `site/notebook.html`
- Notebook: `site/notebooks/quickstart.ipynb`

The web page is best for reviewers and first-time users because it loads
without a Python runtime. The notebook is best for a live demo: it generates
the synthetic CSI fixture, converts it to WiSenseHub `.npz`, inspects one
amplitude trace, and shows how the same command pattern applies to official
dataset releases. The embedded viewer renders the notebook inside GitHub Pages;
live browser execution can be added later with JupyterLite or Binder.

## Repository layout

```text
catalog/                 Dataset and task metadata
schemas/                 JSON schemas
src/wifi_datahub/        Validation and standardization library
src/wifi_datahub/adapters/
                         Dataset-specific conversion adapters
scripts/                 Catalog and website build utilities
site/                    GitHub Pages website
examples/                Reproducible small example
tests/                   Unit tests
docs/                    Standard, curation, and licensing decisions
```

## Scope and terminology

The hub distinguishes clip-level human activity recognition (HAR) from
continuous temporal activity detection/localization (TAD/TAL). Spatial
localization and temporal localization are separate tasks.

## Data and licensing

Code in this repository is MIT licensed. Dataset ownership and licenses remain
with their original creators. A catalog entry is not permission to redistribute
the underlying data. When a license disallows derivatives or redistribution,
WiSenseHub publishes metadata and conversion recipes only.
