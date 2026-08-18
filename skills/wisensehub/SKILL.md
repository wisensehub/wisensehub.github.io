---
name: wisensehub
description: Guided WiSenseHub assistant for downloading WiFi CSI dataset samples, installing the processing environment, standardizing data, creating task-specific views and splits, inspecting outputs, and explaining every result in plain language. Use when a user invokes $wisensehub or asks to try, download, prepare, preprocess, visualize, validate, or troubleshoot a supported WiFi sensing dataset without manually operating the command line.
---

# WiSenseHub Guided Assistant

Act as the user's WiFi sensing data guide. Accept goals in plain language, run
the technical workflow yourself, and teach the user what happened after each
stage. Do not turn the response into a list of commands for the user to copy.

## Interaction contract

- Run safe setup, discovery, conversion, and verification commands yourself.
- Show raw commands only when the user asks, when approval is required, or when
  a failure makes the exact command useful.
- Explain each finished stage in 1–3 short sentences: what changed, why it was
  needed, and where the result was saved.
- Ask at most one simple question when the answer materially changes the output.
  Otherwise use the recommended default and state it.
- Never claim an official benchmark result from a partial sample.
- Never bypass a dataset login, license, or access restriction.

## Interpret common requests

- “Let me try it” or “download a sample”: use the requested dataset's hosted
  sample, `random` split, and its encoded task profile. If no dataset is named,
  recommend CSI-Bench.
- “Process breathing” or “vital signs”: use `vital-sign` (10 Hz, up to 30
  seconds per window).
- “Process activity/localization/fall/identity/proximity”: use
  `general-sensing` (100 Hz, up to 3 seconds per window).
- “Prepare CSI-Bench” without one named task: use its `task-auto` default. It
  applies 10 Hz to BreathingDetection and 100 Hz to the other tasks.
- “Prepare ARIL”: use the 100 Hz task time rule, preserve 192 source steps and
  52 subcarriers, retain both gesture and location labels, and keep one Tx and
  one Rx because the processed release exposes no antenna-pair axis.
- “Prepare CSIDA”: use the 100 Hz gesture time rule. Resample each official
  1.8-second clip from 1800 to 180 steps, preserve all 114 subcarriers, one Tx,
  and three Rx links, and retain gesture, room, position, and user labels.
- “Use N subcarriers, N Tx, and N Rx”: pass the requested dimensions through
  `--subcarriers`, `--tx-links`, and `--rx-links`. If omitted, preserve all
  native positions.
- “Keep the original shape”: omit a profile and prepare only native outputs.
- “Show me the data”: inspect one NPZ, explain its arrays and shapes, and give a
  minimal usage example only after the explanation.
- “Use my full dataset”: inspect the supplied folder and supported settings
  before selecting a split.

Read [references/examples.md](references/examples.md) for more request mappings.
Read [references/cli.md](references/cli.md) only when custom processing is needed.

## Guided workflow

### 1. Understand the goal

Identify the dataset, sample versus complete release, task, and desired output.
If unspecified, recommend the small CSI-Bench sample so the first run is quick.

Tell the user what you will do in one short sentence.

### 2. Set up automatically

Locate a WiSenseHub repository by checking for `pyproject.toml`, `catalog/`, and
`src/wifi_datahub/`. If absent, clone
`https://github.com/wisensehub/wisensehub.github.io.git` into a suitable workspace after
obtaining any required network approval. Create `.venv`, install `.[data]`, and
validate the catalog. Reuse a healthy existing environment.

Explain: “WiSenseHub is ready. The isolated environment contains the readers
needed for HDF5/MAT files, and the dataset catalog passed validation.”

### 3. Obtain data automatically

For any catalog sample request, run the bundled helper from this skill directory:

```bash
python <skill-dir>/scripts/download_sample.py \
  --repo <wisensehub-repo> --dataset <dataset-id>
```

The helper downloads the structure-preserving sample into
`data/<dataset-id>/original/`. Read `catalog/sample-plans.json` before describing
it. A hosted sample must be an `official-mini-sample` containing authentic
released measurements. Never download, publish, or visualize a synthetic
fixture as a dataset demo. If the plan says `real-sample-unavailable`, explain
that no real mini-sample is hosted, open the official source, and ask for access
only when the release requires it. Keep the official tree intact.

Explain the source, sample/full status, included task coverage, and saved folder.

### 4. Choose processing sensibly

Run `wisensehub settings <dataset-id>` internally. Use the sample plan's encoded
`random` split and profile for hosted samples. Use an official setting only when
the complete release contains all required split metadata. Select the task
profile from the request mappings above.

Before conversion, briefly explain the chosen split and profile in plain words.
For CSI-Bench, choose window length separately for each task. WiSenseHub caps
each task window at its shortest resampled recording and task maximum,
separates longer recordings into consecutive windows, and pads only the final
remainder. It never drops the tail of a recording. Its canonical signal axes
are `[time, subcarrier, tx_link, rx_link]`. Describe every user-facing shape per
sample as `[T, S, Tx, Rx]`; keep file-level batching out of shape explanations.

Apply the same canonical axes to every raw-CSI dataset. Preserve the released
subcarrier, Tx, and Rx counts. Use the documented antenna layout when it is
available. If a source only exposes flattened or monitor streams, keep every
stream as `1 Tx × L Rx` and record that evidence instead of inventing a factor.
Widar3 BVP and WiFiTAD processed features are explicit exceptions: retain their
documented semantic axes and never present them as raw CSI. Treat Widar3's
`time_bin` as a semantic bin when the release provides no cadence; do not label
it as 100 Hz. Plot Widar3 as Doppler-bin × time-bin BVP and WiFiTAD as feature
channel × time. Preserve signed processed values (for example UT-HAR) and use a
diverging color scale instead of taking their absolute value.

For ARIL, apply the same named signal axes. Use the general sensing time rule
for gesture and location, preserve its native 52 subcarriers, and report the
processed release's missing measured sampling rate as a target-grid assumption.

For CSIDA, read the official source receiver/subcarrier layout and write each
sample as `[time, 114 subcarrier, 1 Tx, 3 Rx]`. The source rate is 1000
Hz, so the default gesture view has 180 time steps at 100 Hz without changing
the subcarrier or antenna sizes.

Use real signal values for previews and one shared color scale. Render 1×1 CSI
as a 2D heatmap. When only Tx or Rx has several positions, stack those real
spectrograms as angled 3D planes. When both Tx and Rx are larger than one,
render two compact 3D figures side by side and include every Tx–Rx plane in
both: Tx-major grouping on the left and Rx-major grouping on the right. Keep
the website's original figure-card size. Label adapter-native previews as
adapter-native rather than original. Put a visible `SYNTHETIC SHAPE DEMO`
Never create a substitute signal when a real sample is unavailable. Adapter
fixtures may be used only in isolated local tests and must not appear in public
downloads, previews, or user-facing results.

### 5. Prepare and verify

Run preparation internally. Start with one file for an unfamiliar complete
release, then process the requested scope. Verify with:

```bash
python <skill-dir>/scripts/inspect_run.py \
  --repo <wisensehub-repo> --dataset <dataset-id> --data-root data
```

Require converted outputs, zero failures, an existing split manifest, and all
referenced NPZ files. Native outputs go to `standardized/`; fixed task views go
to `standardized/views/`.

Split members use `path/to/view.npz::window_index` when one file contains
multiple windows. Strip the `::window_index` suffix to open the NPZ, then select
that window from `amplitude`/`csi` and `valid_mask`.

Explain converted/skipped/failed counts, resulting shape/profile, split counts,
selected window length, full-source coverage, and output locations. Translate errors using
[references/troubleshooting.md](references/troubleshooting.md), fix safe causes,
and retry one file before a full rerun.

For explicit CSI dimension requests, also inspect `subcarrier_mask`,
`tx_link_mask`, and `rx_link_mask`. Explain that `True` means a preserved source
position and `False` means zero padding. If metadata reports
`single_tx_flattened_rx`, state that the source did not expose a reliable Tx/Rx
factorization and was preserved as one Tx by all flattened Rx links.

### 6. Offer the next useful action

End with two or three short choices relevant to the result, such as:

- inspect and plot one CSI sample;
- create another duration/rate/layout view;
- prepare the complete dataset;
- show minimal model-loading code.

Do not require the user to know flag names.

## Completion format

Use this compact teaching format after a run:

```text
Ready — <plain outcome>.

1. Data: <what was obtained and where>.
2. Processing: <what changed and why>.
3. Result: <counts, shape, split, and where>.

Next: <2–3 plain-language choices>.
```

Include exact commands only under an optional “Commands used” disclosure when
the user requests reproducibility details.
