# TENG-Cabinet-Sensing

Electrical and optical contact-recognition demos for Fig. 3 of *TENG based data center safety system*.

## System requirements

- Python 3.13.5
- Windows 11; CPU
- Dependencies listed in `requirements.txt`

## Installation

Download the repository using **Code > Download ZIP** and extract it. Open a terminal in the project directory and install the dependencies:

```text
python -m pip install -r requirements.txt
```

## Demo

Run both branches:

```text
python run_demo.py
```

Run one branch:

```text
python run_demo.py --branch electrical
python run_demo.py --branch optical
```

| Branch | Input | Model | Output |
| --- | --- | --- | --- |
| Electrical | 27 feature records | Two-stage LDA | Predicted labels and verification report |
| Optical | 9 images | Eight-feature LDA | Predicted labels, feature vectors and verification report |

Each demo compares its results with the files in `expected_output/`. Generated files are saved in the corresponding branch's `output/` directory.

A successful run reports `matches_expected_predictions: true` for all 27 electrical records and `PASS` for all nine optical images. The combined demo took 12.3 seconds in the tested environment.

## Instructions for use

See the [electrical instructions](electrical/README.md) for the feature-table format and prediction command, and the [optical instructions](optical/README.md) for image prediction and feature extraction.

## File structure

```text
TENG-Cabinet-Sensing/
├── README.md
├── requirements.txt
├── run_demo.py
├── electrical/
│   ├── README.md
│   ├── demo.py
│   ├── code/
│   ├── models/
│   ├── demo_data/
│   └── expected_output/
└── optical/
    ├── README.md
    ├── demo.py
    ├── code/
    ├── models/
    ├── demo_data/
    └── expected_output/
```

`code/` contains the inference and feature-processing functions, `models/` contains trained models, `demo_data/` contains the example inputs, and `expected_output/` contains the reference predictions.
