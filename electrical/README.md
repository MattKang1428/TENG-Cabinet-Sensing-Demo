# Electrical recognition demo

This example applies the fitted Fig. 3 electrical classifier to 27 real
feature records. It runs on a CPU without acquisition hardware.

From the repository root:

```text
python -m pip install -r requirements.txt
python electrical/demo.py
```

The program writes `electrical/output/predictions.csv` and
`electrical/output/verification.json`. A successful example reports
`samples: 27` and `matches_expected_predictions: true`; all predicted labels
and gate decisions match `expected_output/example_predictions.csv`.
Use `--output-dir <directory>` to choose another output location.

## Files

| Location | Purpose |
| --- | --- |
| `demo.py` | CSV input, prediction, output and example verification |
| `code/electrical_inference.py` | Two-stage classification logic |
| `models/inference_bundle.pkl` | Both fitted scalers and LDA classifiers, feature order and label map |
| `demo_data/features_example.csv` | 27 feature records with source IDs and reference labels |
| `expected_output/example_predictions.csv` | Reference predictions for these 27 rows |

## Model and example

The first stage applies its own StandardScaler and binary linear discriminant
analysis (LDA). Windows classified as active pass through a second, separately
fitted StandardScaler and eight-class LDA. Both stages receive the original
eight features.

Final labels 1-8 denote signal channels. Label 0 denotes an assigned
synchronous other-channel window. In the source dataset, reference labels
come from detected peaks; label 0 is assigned to the other seven channels at
the same event time.

The example contains the first three source-table records for each reference
label 0-8, selected by label and row order.

## Predict a feature CSV

```text
python electrical/demo.py --input-csv my_features.csv --output-dir my_predictions
```

Provide the following eight numeric, finite feature columns. Their column
order in the CSV is arbitrary; the program selects them by name. Optional
`sample_id` and `label` (or `reference_label`) columns are copied into the
output. Without a sample ID, rows are numbered automatically. Custom inputs
produce predictions without comparing them to the bundled example output.

| Column | Meaning | Unit |
| --- | --- | --- |
| `max` | Maximum voltage | V |
| `mean` | Mean voltage | V |
| `peak_ratio` | Positive-peak contribution to positive-plus-negative magnitude | dimensionless |
| `pvi_seconds` | Positive/negative-extremum interval for a bipolar window | s |
| `rise_time_seconds` | Last 10%-of-positive-peak crossing to peak | s |
| `crest_factor` | Maximum absolute voltage divided by RMS | dimensionless |
| `fall_half_delay_seconds` | Peak to half-height crossing relative to the initial baseline | s |
| `derivative_std` | Standard deviation of the time derivative | V/s |

Features must follow the same definitions as the supplied records, derived
from 2,000-sample voltage windows at 1,000 Hz. The demo input is a feature
table, not a raw waveform file; do not standardize the features beforehand.

## Environment

Use the dependencies pinned in the repository-root [requirements.txt](../requirements.txt).
