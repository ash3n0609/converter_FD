# Fault Prediction & Diagnostics for DC-DC Buck-Boost Converter
## Primary Model: Hybrid 1 - Stacking Ensemble Architecture

This repository contains an end-to-end Machine Learning pipeline and interactive diagnostics dashboard for fault detection, classification, and prognostic health monitoring in DC-DC buck-boost converters.

The project is built around **Hybrid 1: Stacking Ensemble**, fusing four complementary estimators:
* **Linear Support Vector Classifier (Maximum-Margin Hyperplane)**: Fast primal coordinate descent providing robust separation for steady DC voltage drops (Switch Degradation vs Nominal) invariant to board-level parasitic ripple.
* **Multiclass Logistic Regression**: Provides stable, calibrated linear boundaries adhering to the underlying linear circuit equations.
* **Regularized Random Forest (RF)**: Constrained tree depth (`max_depth=6`, `min_samples_leaf=15`) prevents memorizing board-specific micro-ripples while capturing non-linear capacitor ESR and gate switching anomalies.
* **Extreme Gradient Boosting (XGBoost)**: Shallow boosted trees (`max_depth=3`, `min_child_weight=10`) providing residual corrections without ripple overfitting.

---

## 🌟 Key Features

* **Real Simulink Dataset Integration (`dataset_v5`)**: 
  * Aggregates 36 raw Simulink simulation runs (**3,109,444 data points**) across 3 independent converter units (`unit1`, `unit2`, `unit3`) and two distinct load resistances ($2.5\,\Omega$ and $5.0\,\Omega$).
  * **Cold-Start Transient Filtering**: Automatically trims the initial $8\%$ power-up capacitor charging transient ($0\,\text{V} \rightarrow 5\,\text{V}$), eliminating cold-start label noise.
  * **Cross-Unit Group Split**: Trains on `unit1` and `unit2`, and evaluates generalization on the unseen `unit3` test set (**83.70% accuracy, 83.22% Macro F1**).

* **13 Domain-Engineered Power Electronics Features**:
  1. `V_mean`: Mean output voltage
  2. `V_std`: Output voltage standard deviation
  3. `V_rms`: Root-mean-square output voltage
  4. `V_p2p`: Peak-to-peak output voltage ripple
  5. `I_mean`: Mean load current
  6. `I_std`: Load current standard deviation
  7. `I_rms`: Root-mean-square load current
  8. `I_p2p`: Peak-to-peak load current ripple
  9. `V_ripple_ratio`: Normalized voltage ripple ($\frac{V_{\text{p2p}}}{|V_{\text{mean}}| + \epsilon}$) — *Direct indicator of capacitor ESR degradation.*
  10. `I_ripple_ratio`: Normalized current ripple ($\frac{I_{\text{p2p}}}{|I_{\text{mean}}| + \epsilon}$) — *Sensitive to inductor and switch anomalies.*
  11. `R_apparent`: Apparent load impedance ($\frac{V_{\text{mean}}}{|I_{\text{mean}}| + \epsilon}$) — *Decouples operating load changes ($2.5\,\Omega$ vs $5.0\,\Omega$) from physical faults.*
  12. `V_crest_factor`: Voltage waveform crest factor ($\frac{V_{\text{p2p}}}{2\sqrt{2} \cdot V_{\text{rms}} + \epsilon}$) — *Measures harmonic distortion from gate driver switching.*
  13. `P_ratio`: Conduction efficiency ratio ($\frac{V_{\text{mean}} \cdot I_{\text{mean}}}{V_{\text{rms}} \cdot I_{\text{rms}} + \epsilon}$) — *Captures switch conduction degradation independently of unit-to-unit parasitic ripple.*

* **5-Class Fault Taxonomy**:
  1. `0: Healthy` (Nominal steady-state regulation)
  2. `1: ESR Mild` (Early output capacitor degradation)
  3. `2: ESR Severe` (Advanced capacitor ESR degradation with excessive ripple)
  4. `3: Switch Deg` (MOSFET conduction degradation & voltage drop)
  5. `4: Gate Deg` (Gate driver waveform distortion)

* **Interactive Web Dashboard (`app.py`)**:
  * Minimal warm-beige UI powered by Flask and Chart.js.
  * **Drag & Drop CSV Upload**: Instant Converter Health Score (0–100%), fault window breakdown table, and time-series state transition timeline graph.
  * **Manual Parameter Tester**: Interactive sliders to test single feature windows in real time.

---

## 📁 Repository Structure

```text
converter_FD/
├── app.py                  # Flask web server for the interactive dashboard
├── templates/
│   └── index.html          # Web UI dashboard template (Glassmorphism & Chart.js)
├── build_dataset.py        # Concatenates raw Simulink CSV files from dataset_v5/
├── config.py               # Central configuration (Feature definitions, taxonomy, paths)
├── data_loader.py          # Data ingestion, cold-start filtering, invariant feature extraction
├── models.py               # Hybrid 1: Stacking Ensemble definition & base estimator builders
├── pipeline.py             # End-to-end training, cross-unit evaluation, and model serialization
├── inference.py            # CLI script for single-sample & batch CSV inference
├── requirements.txt        # Python package dependencies
├── dataset_v5/             # Folder containing raw Simulink CSV files (36 files)
├── saved_models/           # Serialized Stacking Ensemble weights & StandardScaler (.joblib)
└── plots/                  # Generated confusion matrix plots
```

---

## 🚀 Setup Instructions

1. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🛠️ How to Run

### 1. Train and Evaluate the Hybrid 1 Stacking Ensemble
Train the ensemble with 5-fold cross-validation on `unit1` and `unit2`, and evaluate cross-unit generalization on `unit3`:
```powershell
python pipeline.py
```
* **Outputs**: Serializes `saved_models/Hybrid_1_Stacking_Ensemble.joblib` and generates `plots/Hybrid_1_Stacking_Ensemble_confusion_matrix.png`.

### 2. Launch the Web Interface
Start the Flask web dashboard:
```powershell
python app.py
```
Open your browser at:
👉 **`http://127.0.0.1:5000`**

### 3. Run Command-Line Inference Demo
To test predictions on single feature vectors or entire raw Simulink CSV files via CLI:
```powershell
python inference.py
```
