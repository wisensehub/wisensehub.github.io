# ARIL preparation

Download the processed amplitude archive linked by the official ARIL
repository and place both split files here:

```text
data/aril/original/
├── train_data_split_amp.mat
└── test_data_split_amp.mat
```

Run:

```bash
wisensehub prepare aril
```

The adapter expects `train_data`/`test_data`, activity labels, and location
labels inside the MAT files. Each sample uses `[time, subcarrier, tx_link,
rx_link]`, with 192 time steps, 52 native subcarriers, one Tx, and one Rx.
Because the processed release does not provide trustworthy packet timestamps,
the native output preserves packet index. The task view uses the 100 Hz task
grid and records that rate as an assumption in its JSON metadata.

For a small authentic local sample, run `python scripts/fetch_samples.py aril`.
It downloads the official archive, selects one real recording per location
while covering all six gestures, and records the original sample indices in
`official-sample-manifest.json`. The upstream repository does not state
redistribution terms, so confirm them before publishing the generated subset.
