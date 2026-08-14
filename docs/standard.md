# WiSenseHub Standard v1

## Design principles

1. Preserve original complex CSI whenever it exists.
2. Never claim absolute physical power when calibration is unavailable.
3. Make every interpolation, crop, pad, and unit conversion reproducible.
4. Keep continuous TAD/TAL streams continuous; fixed windows are derived views.
5. Record missing packets rather than hiding them.

## Canonical signal arrays

A standardized `.npz` output contains a primary signal tensor, `valid_mask`,
and `packet_index`. For CSI releases the primary tensor is `amplitude`; other
arrays are included when supported by the source:

| Array | Shape | Unit | Description |
|---|---|---|---|
| `timestamp_s` | `[T]` | seconds | Monotonic time, starting at zero |
| `csi_real` | `[T, S, Tx, Rx]` | source-native | Real CSI component |
| `csi_imag` | `[T, S, Tx, Rx]` | source-native | Imaginary CSI component |
| `amplitude` | `[T, S, Tx, Rx]` | linear | `sqrt(real² + imag²)` |
| `power_db_rel` | `[T, S, Tx, Rx]` | dB relative | `10 log10(power / reference_power)` |
| `valid_mask` | `[T]` | boolean | True when supported by an observed packet |
| `packet_index` | `[T]` | index | Original or generated packet position |

The shape shown to users is always per sample: `[T,S,Tx,Rx]`. A file may store
several samples internally, but the sample index is not part of the signal
schema. Processed task representations such as Widar3 BVP retain their
documented axes rather than being mislabeled as raw CSI.

`S` is subcarrier. `Tx` and `Rx` keep the source antenna/device layout. When a
release exposes streams but no trustworthy factorization, WiSenseHub records
the evidence and preserves them as one Tx by all source streams.

## Native files and derived views

Dataset adapters first write a native standardized file under
`data/<dataset-id>/standardized/`. Native files preserve the official release as
closely as possible after axis naming, dtype normalization, and provenance
capture.

By default, `wisensehub prepare` also writes a **task-family derived view** under
`standardized/views/`, selected from the dataset catalog
`standardization.profile`:

| Profile | Rate | Duration | Channel policy | Typical tasks |
|---|---|---|---|---|
| `general-sensing` | 100 Hz | flexible, up to 3 s | keep native `S`, `Tx`, `Rx` | HAR, occupancy, fall, motion |
| `vital-sign` | 10 Hz | flexible, up to 30 s | keep native `S`, `Tx`, `Rx` | breathing / vital signs |

```bash
wisensehub prepare <dataset-id> --profile vital-sign
wisensehub prepare <dataset-id> \
  --target-rate 100 \
  --duration 3 \
  --interpolation linear \
  --layout link-subcarrier
```

Derived views keep a `derived_from` pointer to the native NPZ. Supported
policies are:

| Option | Purpose |
|---|---|
| `--profile` | Apply a task-family rate/duration policy |
| `--target-rate` | Resample a timestamped or rate-known sequence to a fixed Hz |
| `--duration` | Crop/pad/resample to a fixed time interval |
| `--target-length` | Force an exact number of time steps when seconds are unknown |
| `--interpolation` | Choose `none`, `nearest`, or `linear` |
| `--layout canonical` | Keep `[T,S,Tx,Rx]` per sample |
| `--layout link-subcarrier` | Flatten `S × Tx × Rx` to `[T,F]` for a model view |
| `--subcarriers`, `--tx-links`, `--rx-links` | Optional crop/pad of signal dims (off by default) |

## Time and sampling

- General sensing: 100 Hz, with task-length windows up to 3 seconds.
- Vital sign: 10 Hz, with task-length windows up to 30 seconds.
- Subcarrier, Tx, and Rx counts stay native unless their explicit options are set.
- Interpolation is performed independently on real and imaginary components.
- Samples outside the observed range are zero padded and marked invalid.
- When the source does not report a rate, the task profile rate is assumed and
  recorded in the sidecar.

## Power units

Commodity CSI is usually uncalibrated. WiSenseHub therefore uses relative dB
with the median valid power of each sample as the 0 dB reference. Absolute dBm
is only permitted when the source supplies a documented calibration equation
and calibration metadata.

## Labels

Clip tasks store one or more labels in the sidecar. TAD/TAL stores a `segments`
array with `start_seconds`, `end_seconds`, and canonical `label`, plus the
original source label.

## Provenance

Every standardized file has a JSON sidecar recording dataset ID, source file,
source checksum when known, adapter version, target rate, duration policy,
power reference, transformations, and creation timestamp.
