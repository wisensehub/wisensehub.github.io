# Troubleshooting

## `official split file not found`

The selected official setting depends on metadata not present in the partial
download. For a representative sample, select a supported `random` setting. For
benchmark reproduction, download the complete release including its split
files and keep their official relative paths.

## One or more conversions failed

Inspect `data/<dataset-id>/prepare-manifest.json`. Group records with
`status: failed` by `error`, fix the repeated cause, and retry with `--limit 1`.
Do not repeatedly rerun the entire dataset without reading the first error.

## HDF5 or MAT reader missing

Install runtime data dependencies from the repository root:

```bash
.venv/bin/python -m pip install -e ".[data]"
```

Then retry one file. Avoid changing the adapter merely because an optional
reader is absent.

## No source files discovered

Run `wisensehub settings <dataset-id>` and compare its recognized layouts with
the directory under `data/<dataset-id>/original/`. Check whether the archive
introduced an extra top-level directory; keeping that official level is usually
valid. Confirm the dataset ID and file extensions before moving anything.

## Output did not change

Preparation skips existing outputs by default. First inspect the manifest and
confirm that the new options are correct. Then rerun the exact dataset with
`--force`. Derived views are under `standardized/views/`, while native files
remain under `standardized/` by design.

## Partial sample and official claims

A structure-preserving sample can validate conversion, schemas, views, and
example code. It cannot validate dataset-wide statistics or reproduce an
official benchmark split when required subjects, environments, or split files
are absent. State this limitation in the run report.
