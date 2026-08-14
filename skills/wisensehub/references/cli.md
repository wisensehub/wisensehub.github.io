# CLI and output reference

## Installation extras

`pip install -e ".[data]"` installs WiSenseHub in editable mode plus runtime
readers such as HDF5 support. `.[dev]` installs development and test tools; it is
not enough by itself when a source adapter needs the data readers.

## Prepare controls

`--setting` selects how converted samples are assigned to train, validation,
and test partitions. It does not change CSI values. Always discover valid values
with `wisensehub settings <dataset-id>`.

- Official settings reproduce a published partition and need the official
  metadata or split files.
- Cross-group settings hold out subjects, rooms, devices, or other groups.
- Random settings create a seeded hub-defined split and are appropriate for
  small demonstrations, not claims of official benchmark comparability.

`--profile vital-sign` uses 10 Hz and a maximum 30-second / 300-step window.
`--profile general-sensing` uses 100 Hz and a maximum 3-second / 300-step window.
CSI-Bench defaults to `task-auto`: BreathingDetection uses 10 Hz and other tasks
use 100 Hz. Its window length is selected independently for each task.
ARIL uses the 100 Hz general sensing rule for gesture and location. Its 192
source steps remain intact because they are shorter than the 300-step maximum.

By default, WiSenseHub measures the shortest resampled recording in the
dataset and uses the smaller of that length and the profile maximum. Longer
recordings are separated into consecutive, non-overlapping windows. The final
remainder is padded and marked invalid in `valid_mask`; no source tail is
dropped.

Optional view overrides:

- `--target-rate HZ`: output time sampling rate.
- `--duration SECONDS`: clip duration.
- `--target-length STEPS`: exact number of time steps.
- `--interpolation {none,nearest,linear}`: resampling method.
- `--layout canonical`: preserve canonical signal axes.
- `--layout flat`: flatten non-time signal axes.
- `--layout link-subcarrier`: expose link/subcarrier features for convenient
  model input.
- `--links N`: choose the combined link axis used by legacy datasets.
- `--subcarriers N`: center-select or zero-pad subcarriers.
- `--tx-links N` and `--rx-links N`: select or zero-pad explicit Tx/Rx axes.
- `--seed N`: seed for hub-generated random partitions.
- `--ratios TRAIN VAL TEST`: custom random partition ratios.
- `--holdout VALUE...`: test group values for supported cross-group settings.
- `--limit N`: convert only the first N discovered inputs for a trial.
- `--force`: rebuild generated outputs that already exist.

Prefer a named profile without overrides for reproducibility. Record every
override in the final report.

An explicit duration or target length overrides automatic shortest-recording
selection, but full-coverage segmentation still applies.

## Output contract

For `data/<dataset-id>/`:

```text
original/                 untouched extracted source release
standardized/             native standardized NPZ and JSON sidecars
standardized/views/       optional fixed-profile or custom views
reports/                  quality reports
splits/<setting>.json     train/validation/test membership
prepare-manifest.json     run options and per-source status
```

NPZ data may include CSI, amplitude, phase, timestamps, labels, metadata, and a
mask. The mask marks which values are valid after padding or resampling; invalid
or padded values should not contribute to model loss or summary statistics.

CSI-Bench and ARIL use `[time, subcarrier, tx_link, rx_link]`, or
`[sample, time, subcarrier, tx_link, rx_link]` for batched/segmented files. Its
`subcarrier_mask`, `tx_link_mask`, and `rx_link_mask` distinguish preserved
source positions from zero padding. By default, all three native dimensions
are preserved. ARIL's processed release maps to 52 subcarriers, one Tx, and
one Rx, while preserving both gesture and location labels.

A split member may look like `standardized/views/clip.npz::2`. The suffix means
“window 2 inside this NPZ”; open the path before `::`, then index the arrays with
the number after it.
