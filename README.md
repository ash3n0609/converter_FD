# Fault Prediction for Buck-Boost Converter

This repository contains an end-to-end Machine Learning pipeline to detect and classify faults in a DC-DC buck-boost converter.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Run the training pipeline:**
   ```bash
   python pipeline.py
   ```
   This will load data, train the models (Random Forest, XGBoost, SVM, PyTorch MLP), evaluate them, output metrics, and save the best model weights into the `saved_models` directory.

2. **Run inference:**
   ```bash
   python inference.py
   ```
   This script demonstrates how to load the trained models and predict on a new data sample.

## Swapping Synthetic Data with Real Simulink Data

Currently, the pipeline generates synthetic data for testing. Once your real Simulink data is ready:

1. Export your Simulink data to a CSV file. The pipeline expects tabular data where each row represents a time window (e.g., extracted features from that window) and has the following columns (plus a label column):
   - `Vin`, `Vout`, `IL`, `duty_cycle`, `switch_temp`, `fault_label`
2. Place the CSV file in a known location.
3. Open `config.py` and update `USE_SYNTHETIC_DATA = False`.
4. Update `DATA_PATH` in `config.py` to point to your CSV file.
5. If your column names differ, update `FEATURE_COLS` and `LABEL_COL` in `config.py`.

The pipeline will automatically load your CSV and train the models on your real data.
