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
| `csi_real` | `[T, L, S]` | source-native | Real CSI component |
| `csi_imag` | `[T, L, S]` | source-native | Imaginary CSI component |
| `amplitude` | `[T, L, S]` | linear | `sqrt(real² + imag²)` |
| `power_db_rel` | `[T, L, S]` | dB relative | `10 log10(power / reference_power)` |
| `valid_mask` | `[T]` | boolean | True when supported by an observed packet |
| `packet_index` | `[T]` | index | Original or generated packet position |

Clip collections may add a leading sample axis, producing `[N,T,L,S]` and a
mask of shape `[N,T]`. Processed task representations such as Widar3 BVP retain
their documented axes rather than being mislabeled as raw CSI.

`L` is the flattened transmit-receive link dimension and `S` is subcarrier.
Original antenna dimensions remain in the JSON sidecar.

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
| `general-sensing` | 100 Hz | 3 s (`T=300`) | keep native `L` and `S` | HAR, occupancy, fall, motion |
| `vital-sign` | 10 Hz | 60 s (`T=600`) | keep native `L` and `S` | breathing / vital signs |

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
| `--layout canonical` | Keep `[T,L,S]` or `[N,T,L,S]` |
| `--layout link-subcarrier` | Flatten link/subcarrier axes to `[T,F]` or `[N,T,F]` |
| `--links`, `--subcarriers` | Optional force/pad of channel dims (off by default) |

## Time and sampling

- General sensing: 3 seconds at 100 Hz (`T=300`).
- Vital sign: 60 seconds at 10 Hz (`T=600`).
- Link and subcarrier counts stay native unless `--links` / `--subcarriers` are set.
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
