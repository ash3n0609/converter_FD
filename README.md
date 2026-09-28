# Fault Prediction & Diagnostics for DC-DC Buck-Boost Converter

This repository contains an end-to-end Machine Learning and Hybrid Ensemble pipeline to detect, classify, and diagnose faults in a DC-DC buck-boost converter. It supports training on real Simulink multi-unit dataset exports, model serialisation, ONNX edge export for microcontrollers (ESP32-S3 / TFLite), and an interactive web dashboard for real-time file diagnostics.

---

## 🌟 Key Features & Updates

* **Real Simulink Dataset Integration (`dataset_v5`)**: 
  * Automatically aggregates 36 raw Simulink simulation files (**2,072,416 data samples**) across 3 independent converter test units (`unit1`, `unit2`, `unit3`).
  * Extracts **41,440 time-series feature windows** (Window Size = 50 samples) containing 8 statistical domain features (`V_mean`, `V_std`, `V_rms`, `V_p2p`, `I_mean`, `I_std`, `I_rms`, `I_p2p`).
  * Cross-Unit Group Split prevents temporal data leakage by evaluating generalization on unseen units.

* **6-Class Fault Taxonomy**:
  1. `0: Healthy` (Nominal steady-state operation)
  2. `1: ESR Mild` (Early output capacitor degradation)
  3. `2: ESR Severe` (Advanced capacitor ESR degradation with high ripple)
  4. `3: Switch Deg` (MOSFET conduction degradation & voltage drop)
  5. `4: Gate Deg` (Gate driver waveform distortion)
  6. `5: Sensor Fault` (Voltage feedback measurement anomaly / noise spike)

* **Baseline & Advanced Hybrid Architectures**:
  * **Baseline Models**: Majority Trivial, Random Forest, XGBoost, SVM (RBF), Quadratic Discriminant Analysis (QDA), Quadratic SVM, PyTorch MLP (with Batch Normalization), Same-Board Oracle.
  * **Hybrid Model 1 (Stacking Ensemble)**: Combines QDA (high Healthy recall), Quadratic SVM (high Sensor Fault recall), Random Forest, and XGBoost with a Logistic Regression Meta-Learner.
  * **Hybrid Model 2 (Hierarchical Cascade)**: Two-stage routing separating component degradation from signal anomalies.

* **Embedded & Microcontroller Deployment**:
  * Automatic export of trained deep learning models to **ONNX** (`saved_models/PyTorch_MLP.onnx`) for TFLite / ESP32-S3 edge deployment.

* **Interactive Web Dashboard (`app.py`)**:
  * Sleek glassmorphism dark-mode interface built with Flask and Chart.js.
  * **Drag & Drop CSV Dataset Upload**: Upload raw Simulink files (`Time / s`, `Load Current`, `Load Voltage`) for instant converter health scoring and fault window breakdown.
  * **Converter Health Index**: Computes dynamic 0–100% health score.
  * **Time-Series Classification Timeline**: Visualizes window-by-window health state transitions.
  * **Manual Parameter Tester**: Interactive sliders for single-window feature diagnosis.

---

## 📁 Repository Structure

```text
converter_FD/
├── app.py                  # Flask web server for the interactive dashboard
├── templates/
│   └── index.html          # Web UI dashboard template (Glassmorphism & Chart.js)
├── build_dataset.py        # Concatenates raw Simulink CSV files from dataset_v5/
├── config.py               # Central configuration (Paths, Hyperparameters, Fault Classes)
├── data_loader.py          # Data ingestion, window feature extraction, and group-splitting
├── models.py               # Baseline models, PyTorch MLP, and Hybrid Ensemble architectures
├── pipeline.py             # End-to-end training and evaluation script
├── inference.py            # CLI script for single-sample & batch CSV inference
├── requirements.txt        # Python package dependencies
├── dataset_v5/             # Folder containing raw Simulink CSV files (36 files)
├── saved_models/           # Serialized trained model weights (.joblib, .pt, .onnx)
└── plots/                  # Generated confusion matrices & feature importance plots
```

---

## 🚀 Setup Instructions

1. **Create and activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ How to Run

### 1. Build Dataset from Raw Simulink CSV Files
If you have raw simulation files in `dataset_v5/` (with columns `Time / s`, `Load Current`, `Load Voltage`), combine them into the master training dataset:
```bash
python build_dataset.py
```
*(This generates `data/simulink_export.csv` containing all 2.07M simulation rows).*

### 2. Run the Full Model Training Pipeline
Train all baselines and hybrid models, output evaluation metrics, generate plots, and save trained weights:
```bash
python pipeline.py
```
* **Saved Outputs:** Model binaries saved in `saved_models/` and evaluation charts saved in `plots/`.

### 3. Launch the Interactive Web Dashboard
Start the web application to inspect and test your models via your browser:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

* **CSV Upload:** Drag & drop any `.csv` dataset file containing `Time / s`, `Load Current`, `Load Voltage` to view real-time health index, fault distribution, and window timeline graphs.

### 4. Run Command-Line Inference Demo
To test predictions via CLI on sample feature vectors or full CSV files:
```bash
python inference.py
```

---

## 📊 Dataset Schema & Custom Data Setup

If you wish to test custom Simulink data:
1. Ensure your CSV export contains columns named `Time / s` (or `time`), `Load Current` (or `current`), and `Load Voltage` (or `voltage`).
2. Place the CSV file in `data/simulink_export.csv` or upload it directly through the web UI at `http://127.0.0.1:5000`.
3. In `config.py`, toggle `USE_SYNTHETIC_DATA = False` to train on real data, or `USE_SYNTHETIC_DATA = True` for quick synthetic benchmarking.
