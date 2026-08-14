# Natural-language examples

## First-time sample

User: “Download a sample and prepare it for me.”

Action: set up WiSenseHub, download the hosted CSI-Bench sample, use a random
split and CSI-Bench's task-auto time profile, verify all outputs, then explain
each stage. Do not ask the user to open a terminal.

## Any catalog dataset

User: “Prepare the UT-HAR sample for me.”

Action: read its catalog entry and `catalog/sample-plans.json`. If the plan is
`real-sample-unavailable`, do not generate or download a substitute signal.
Explain that no real mini-sample is hosted, direct the user to the official
release, and offer to prepare an authorized local copy after they obtain it.

## Breathing

User: “Prepare the breathing sample for vital-sign research.”

Action: obtain the sample if missing, select the vital-sign profile, explain
that it creates full-coverage windows of up to 30 seconds at 10 Hz, then verify
the views and report the selected window length.

## Custom shape

User: “I need 5-second activity clips at 50 Hz.”

Action: use the activity/general-sensing workflow with custom duration and
rate. Explain that each full window has 250 time steps, longer recordings create
multiple windows, and the last remainder uses `valid_mask`. Record both overrides.

User: “Give me CSI-Bench fall data with 64 subcarriers, 1 Tx, and 2 Rx.”

Action: select the fall/general-sensing 100 Hz time profile and request
`--subcarriers 64 --tx-links 1 --rx-links 2`. Verify the output axes are
`[time, subcarrier, tx_link, rx_link]` (plus `sample` when segmented), then
explain each dimension mask and any source antenna-mapping assumption.

User: “Prepare the ARIL gesture sample with the same standard axes.”

Action: use the 100 Hz task time rule and preserve each sample as
`[time, subcarrier, tx_link, rx_link] = [192, 52, 1, 1]`. Verify gesture and
location labels, all three dimension masks, and the recorded time-grid
assumption. Explain that the hosted ARIL package contains authentic processed
measurements selected from the official release.

## Inspect outputs

User: “What did I get?”

Action: read the preparation manifest and one NPZ. Explain native versus derived
views, array names, mask meaning, shapes, labels, and split membership before
showing any code. If a split member has a `::window_index` suffix, explain and
handle it rather than treating the complete string as a file path.

## Full dataset

User: “Prepare my complete CSI-Bench download.”

Action: locate the supplied folder, confirm official split files are present,
and then use `official_id`. If they are missing, explain why the official split
cannot be created and offer a random exploratory split without presenting it as
an official benchmark.

## Troubleshooting

User: “It says 36 conversions failed.”

Action: read `prepare-manifest.json`, group failures by exact error, explain the
dominant cause in plain language, apply a safe fix, retry one source file, and
continue only if that test succeeds.
