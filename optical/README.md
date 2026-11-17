# Optical recognition demo

This demo applies the optical classifier used for Fig. 3 to supplied episode images. It extracts one blue-intensity score from each of eight calibrated regions, then applies the saved StandardScaler and Linear Discriminant Analysis (LDA) model. The model file contains the fitted parameters, ROI coordinates and decision thresholds.

## Run

Install the repository-level `requirements.txt`. From the repository root:

```text
python optical/demo.py
```

The program loads the nine images in `demo_data/images/`, checks their hashes and compares every predicted class and eight-score feature vector with `expected_output/predictions.json`. A successful run reports **PASS** and writes `optical/output/predictions.json`. All required inputs are included.

To predict an image with the same calibrated camera geometry:

```text
python optical/demo.py --image path/to/image.jpg
```

This prints the predicted class and scores, and saves `optical/output/prediction.json`. Both commands accept `--output-dir PATH`. Packaged inputs are located relative to `demo.py`; the current working directory does not affect them.

## Supplied examples and expected result

The demo uses the earliest evaluation image from each reference class (0-8). The saved predictions match the reference labels for all nine images.

Classes 1-8 identify the eight optical regions. Class 0 is the automatic below-labeling-threshold category. `demo_data/manifest.csv` records the label and SHA-256 for each supplied image. All supplied examples use the LDA decision branch.

## Files and dependencies

- `demo.py`: runs the example or predicts one image.
- `code/predict_8d.py`: loads the fitted model and applies its decision logic.
- `code/deployment_features.py`: image reading, ROI extraction and blue-score calculations.
- `models/model_8d.pkl`: the original fitted model, with its ROI and threshold configuration.
- `demo_data/`: Nine images and their manifest.
- `expected_output/predictions.json`: saved predictions and features for comparison.

Validated with Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1 and OpenCV 4.12.0.88.
