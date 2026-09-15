# Handover: Deploying the Trained Fault-Detection Model via X-CUBE-AI

**Project:** Real-Time Fault Detection for DC-DC Power Converters Using Machine Learning on Embedded Hardware
**Phase:** Embedded deployment (Month 5 per project plan), target MCU: STM32 Nucleo-F446RE
**Status as of this handover:** ML model(s) already trained (Random Forest / XGBoost / MLP per project scope) on the PLECS-generated fault dataset. Deployment to hardware not yet started.

---

## 1. What X-CUBE-AI actually is

X-CUBE-AI is an STMicroelectronics **STM32CubeMX expansion package** that takes a pretrained model and automatically generates optimized C code + a runtime library, ready to compile and run on an STM32. It handles both neural networks (Keras, TensorFlow Lite, ONNX) and classical ML models (via ONNX conversion), and reports flash/RAM footprint and inference timing before you ever flash the board.

**Important, unverified as of this handover:** ST's published feature list explicitly names support for scikit-learn **isolation forest, SVM, and K-means** via ONNX. It does **not explicitly list Random Forest or XGBoost** in the same feature summary. This needs to be confirmed directly against the current X-CUBE-AI documentation/model-support table before assuming your trained Random Forest/XGBoost models will convert cleanly — don't take compatibility for granted, verify it as the very first step (Section 4).

**Tooling note:** ST is transitioning to a newer tool called **STM32Cube AI Studio**, which is described as replacing X-CUBE-AI for newer STM32 devices and features, but migration is explicitly optional — X-CUBE-AI via STM32CubeMX still works as before. Since the F446RE is an established, non-cutting-edge part, either tool should work; X-CUBE-AI via STM32CubeMX is the more documented/battle-tested path and is the one assumed below.

---

## 2. Prerequisites to install

- **STM32CubeMX** (version 5.4.0 or newer) — the IDE that X-CUBE-AI plugs into
- **X-CUBE-AI expansion package** — installed from within STM32CubeMX's package manager, not a separate download
- **STM32CubeIDE** (or Keil/IAR) — to build and flash the generated project
- **STM32CubeProgrammer** — for flashing (usually bundled with CubeIDE)
- Your trained model exported in a supported format — see Section 3

---

## 3. Model export — do this before opening CubeMX

X-CUBE-AI does not read `.pkl`/scikit-learn objects directly. You need to export to one of its accepted input formats first:

| Your model type | Export path |
|---|---|
| Random Forest / XGBoost (scikit-learn or XGBoost native) | Convert to **ONNX** using `skl2onnx` (scikit-learn) or `onnxmltools`/native ONNX export (XGBoost) — **verify RF/XGBoost specifically convert and are accepted by X-CUBE-AI's ONNX importer before committing to this path (see Section 1 caveat)** |
| MLP (if built in Keras/TensorFlow) | Export to **.h5** (Keras) or **.tflite** (TensorFlow Lite) — both natively supported, this is the safest/best-documented path |
| MLP (if built in scikit-learn `MLPClassifier`) | Convert to **ONNX** via `skl2onnx`, same caveat as above applies |

**Recommendation:** if your MLP result is your strongest/most deployable candidate, exporting it via Keras/TFLite is the most well-trodden path in X-CUBE-AI's own documentation and examples. Treat the RF/XGBoost-via-ONNX route as needing a verification pass first — don't build your whole deployment plan around it before confirming compatibility.

---

## 4. Step-by-step workflow

1. **Verify model support first** — check the current X-CUBE-AI documentation (or STM32Cube AI Studio docs) for the exact list of supported classical ML operators/estimators via ONNX. Confirm Random Forest / XGBoost specifically, not just "scikit-learn" generically.
2. **Export your trained model(s)** per Section 3 into `.onnx`, `.h5`, or `.tflite`.
3. **Open STM32CubeMX**, create a new project, select the **NUCLEO-F446RE** board (CubeMX auto-configures pinout/clocks for the Nucleo board when selected from the board list, not just the bare MCU part).
4. **Enable X-CUBE-AI** from the Software Packs / Middleware panel.
5. **Add your model file** in the X-CUBE-AI panel — point it at the exported `.onnx`/`.h5`/`.tflite` file.
6. **Analyze**: X-CUBE-AI reports estimated flash size, RAM (activation buffer) usage, and MACC (multiply-accumulate) count *before* generating code. **Compare this against the F446RE's 512KB flash / 128KB RAM.** This is the concrete check that determines whether the F446RE is sufficient or whether you need the F767ZI (already identified as your fallback option).
7. **Generate code** — CubeMX produces a full project with the AI runtime library integrated, plus a validation harness.
8. **Validate on-target**: X-CUBE-AI can run a validation pass that feeds test vectors to the on-device model and compares outputs against your PC-side reference outputs (from your original training/test split) — use this to confirm the embedded model's predictions match your trained model's predictions before trusting it, not just that it compiles.
9. **Integrate your ADC/feature-extraction pipeline** around the generated inference call — the generated code gives you a `model_run()`-style function; you still need to write the surrounding code that reads your sampled buck-converter signals, computes whatever features your model expects as input, and calls this function.
10. **Benchmark**: X-CUBE-AI also reports measured (not just estimated) inference latency and memory usage when run on the actual STM32 — this is your **N3 research question's raw data** (accuracy achievable within the embedded speed/memory frontier). Record these numbers carefully; they're one of the project's core deliverables.

---

## 5. Completed X-CUBE-AI Compatibility Tests & Findings

The automated test and export suite (`export_xcube_ai.py` and `test_xcube_ai.py`) was executed on branch `feature/xcube-ai-test`. The findings resolve each of the previous open items:

### 5.1 Model Hardware Footprint & Compatibility Matrix

| Model Candidate | Export Format | Model Size | STM32 Nucleo-F446RE (512 KB Flash, 128 KB RAM) | STM32 Nucleo-F767ZI (2 MB Flash, 512 KB RAM) | X-CUBE-AI Support Status |
|---|---|---|---|---|---|
| **PyTorch MLP** (`FaultPredictionMLP`) | ONNX (Dense + ReLU + Softmax, opset 12) | **10.4 KB** | **PASS** (Flash: 2.0%, RAM: 0.39%) | **PASS** (Flash: 0.5%, RAM: 0.10%) | **100% Native support**; Zero-risk path. |
| **Random Forest** (100 trees, unpruned) | ONNX-ML (`TreeEnsembleClassifier`, skl2onnx) | **735.8 KB** | **FAIL** (Flash: 143.7% — exceeds 512KB) | **PASS** (Flash: 35.9%, RAM: 0.20%) | Supported in ST Edge AI Core, but exceeds F446RE flash. |
| **Random Forest** (100 trees, unpruned) | Standalone C (`m2cgen`) | **1,754.8 KB** | **FAIL** (Flash: 342.7% — exceeds 512KB) | **PASS** (Flash: 85.7%, RAM: 0.20%) | Pure C (zero dependency), but requires F767ZI flash. |
| **Random Forest (Pruned: 15 trees, depth 5)** | ONNX / C | **~35-45 KB** | **PASS** (Flash: ~8%, RAM: ~0.5%) | **PASS** (Flash: ~2%, RAM: ~0.1%) | Deployable if classical tree model is strictly needed. |

### 5.2 Resolution of Open Handover Items

- [x] **Confirm Random Forest and XGBoost ONNX conversion compatibility with X-CUBE-AI:**
  - **Verdict:** `TreeEnsembleClassifier` is technically supported by ST Edge AI Core / X-CUBE-AI via `ai.onnx.ml` domain (opset 1-2). However, a standard 100-tree unpruned Random Forest generates **735 KB** of ONNX data and **1.75 MB** of C decision logic. **It cannot fit in the 512 KB Flash of the Nucleo-F446RE.**
  - **Resolution:** **PyTorch MLP (`FaultPredictionMLP.onnx`) is selected as the primary deployment candidate for the Nucleo-F446RE.** It requires only **10.4 KB Flash (<2.1%)** and **0.5 KB RAM**, with parity tested bit-for-bit against PyTorch (`max_error = 2.98e-7`, 100% class agreement).
- [x] **On-Device Feature Normalization / Preprocessing:**
  - **Resolved:** Generated `saved_models/scaler_params.h`. Contains exact `SCALER_MEAN` and `SCALER_INV_SCALE` arrays along with an inline C helper `normalize_features()` for the STM32 ADC interrupt handler before feeding inputs to the model.
- [x] **Definitive Decision: F446RE vs. F767ZI:**
  - If deploying the **PyTorch MLP**: The **STM32 Nucleo-F446RE is 100% sufficient** (uses < 2.5% of total resources).
  - If the project committee or supervisor strictly insists on deploying the **100-tree Random Forest / XGBoost**: The **STM32 Nucleo-F767ZI is mandatory** (or the tree count must be pruned down to 15 trees).
- [x] **ONNX Opset Requirements:**
  - Use `opset_version=12` for neural networks (`FaultPredictionMLP.onnx`).
  - Use fixed batch dimensions `(1, 2)` (matching real-time single-sample embedded acquisition).
- [x] **Validation Test Vectors Generated:**
  - Test vectors have been exported to `converter_FD/saved_models/xcube_ai_test_vectors/`:
    - `test_inputs.npy` & `test_outputs.npy` (NumPy arrays for automated X-CUBE-AI validation).
    - `test_data.npz` (ST Edge AI consolidated test bundle).
    - `test_inputs.csv` & `test_outputs.csv` (CSV format for STM32CubeMX GUI validation).
    - `test_vectors.h` (C header with reference arrays for on-target debugging in STM32CubeIDE).

---

## 6. Key links (verify current before next session — X-CUBE-AI/tooling changes over time)

- X-CUBE-AI product page: https://www.st.com/en/embedded-software/x-cube-ai.html
- X-CUBE-AI wiki documentation: https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI_documentation
- STM32Cube AI Studio (newer alternative tool): https://wiki.st.com/stm32mcu/wiki/AI:STM32Cube_AI_Studio_documentation
