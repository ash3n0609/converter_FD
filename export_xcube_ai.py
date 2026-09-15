import os
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import numpy as np
import torch
import torch.nn as nn
import config
from models import FaultPredictionMLP
from data_loader import load_and_preprocess_data

def export_pytorch_to_onnx(pt_model_path, onnx_model_path, input_dim=2, num_classes=5, with_softmax=True):
    """
    Exports the trained PyTorch MLP model to an ONNX model optimized for STM32Cube.AI.
    Uses model.eval() so Dropout layers are removed.
    """
    print(f"Loading PyTorch weights from {pt_model_path}...")
    base_model = FaultPredictionMLP(input_dim, num_classes)
    base_model.load_state_dict(torch.load(pt_model_path, map_location='cpu'))
    base_model.eval()

    class DeployableMLP(nn.Module):
        def __init__(self, model, add_softmax=True):
            super(DeployableMLP, self).__init__()
            # Extract only active evaluation layers (strip Dropout)
            self.linear1 = model.network[0]
            self.relu1 = model.network[1]
            self.linear2 = model.network[3]
            self.relu2 = model.network[4]
            self.linear3 = model.network[6]
            self.add_softmax = add_softmax
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.relu1(self.linear1(x))
            x = self.relu2(self.linear2(x))
            x = self.linear3(x)
            if self.add_softmax:
                x = self.softmax(x)
            return x

    deploy_model = DeployableMLP(base_model, add_softmax=with_softmax)
    deploy_model.eval()

    # Dummy input with shape (1, input_dim) representing 1 sensor reading sample
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    print(f"Exporting PyTorch model to ONNX: {onnx_model_path}...")
    try:
        torch.onnx.export(
            deploy_model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_data'],
            output_names=['output_probabilities' if with_softmax else 'output_logits'],
            dynamic_axes=None,
            dynamo=False
        )
    except (TypeError, ValueError):
        torch.onnx.export(
            deploy_model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_data'],
            output_names=['output_probabilities' if with_softmax else 'output_logits'],
            dynamic_axes=None
        )
    print(f"PyTorch ONNX model saved to {onnx_model_path}")


def export_random_forest_to_onnx(rf_model_path, onnx_model_path, input_dim=2):
    """
    Exports Scikit-Learn Random Forest to ONNX using skl2onnx (ONNX-ML domain).
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    print(f"Loading Random Forest from {rf_model_path}...")
    rf_model = joblib.load(rf_model_path)

    initial_type = [('float_input', FloatTensorType([1, input_dim]))]  # input_dim = 8 windowed features
    print(f"Converting Random Forest to ONNX via skl2onnx...")
    onx = convert_sklearn(
        rf_model,
        initial_types=initial_type,
        target_opset={'': 12, 'ai.onnx.ml': 2},
        options={type(rf_model): {'zipmap': False}}  # Output raw class probabilities tensor
    )

    with open(onnx_model_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Random Forest ONNX model saved to {onnx_model_path}")


def export_random_forest_to_c(rf_model_path, c_output_path, h_output_path):
    """
    Exports Random Forest directly to standalone C code using m2cgen.
    This provides a zero-dependency C fallback for STM32 microcontrollers.
    """
    import m2cgen as m2c

    print(f"Exporting Random Forest to native C via m2cgen...")
    rf_model = joblib.load(rf_model_path)
    c_code = m2c.export_to_c(rf_model, function_name="rf_predict_score")

    with open(c_output_path, "w") as f:
        f.write("/* Auto-generated Random Forest C implementation for STM32 */\n")
        f.write('#include "random_forest_model.h"\n\n')
        f.write(c_code)
        f.write("\n\nint rf_predict_class(const double *features) {\n")
        f.write(f"    double scores[{config.NUM_CLASSES}];\n")
        f.write("    rf_predict_score(features, scores);\n")
        f.write("    int best_class = 0;\n")
        f.write("    double max_score = scores[0];\n")
        f.write(f"    for (int i = 1; i < {config.NUM_CLASSES}; i++) {{\n")
        f.write("        if (scores[i] > max_score) {\n")
        f.write("            max_score = scores[i];\n")
        f.write("            best_class = i;\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("    return best_class;\n")
        f.write("}\n")

    with open(h_output_path, "w") as f:
        f.write("#ifndef RANDOM_FOREST_MODEL_H\n")
        f.write("#define RANDOM_FOREST_MODEL_H\n\n")
        f.write(f"#define RF_NUM_CLASSES {config.NUM_CLASSES}\n")
        f.write(f"#define RF_NUM_FEATURES 2\n\n")
        f.write("void rf_predict_score(const double *input, double *output);\n")
        f.write("int rf_predict_class(const double *features);\n\n")
        f.write("#endif /* RANDOM_FOREST_MODEL_H */\n")

    print(f"Random Forest C code saved to {c_output_path} and {h_output_path}")


def export_scaler_header(scaler_path, header_path):
    """
    Exports the StandardScaler mean and scale parameters to a C header file
    for on-device feature normalization on STM32 before inference:
    x_normalized = (raw_adc_value - mean) * (1.0 / scale)
    """
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    with open(header_path, "w") as f:
        f.write("/* Auto-generated Scaler parameters for STM32 ADC feature preprocessing */\n")
        f.write("#ifndef SCALER_PARAMS_H\n")
        f.write("#define SCALER_PARAMS_H\n\n")
        f.write(f"#define NUM_FEATURES {len(scaler.mean_)}\n\n")
        f.write("/* Feature indices */\n")
        for i, col in enumerate(config.FEATURE_COLS):
            safe_name = col.upper().replace(' ', '_')
            f.write(f"#define FEAT_IDX_{safe_name} {i}\n")
        f.write("\n")

        means_str = ", ".join([f"{m:.8f}f" for m in scaler.mean_])
        scales_str = ", ".join([f"{s:.8f}f" for s in scaler.scale_])
        inv_scales_str = ", ".join([f"{(1.0/s):.8f}f" for s in scaler.scale_])

        f.write(f"static const float SCALER_MEAN[NUM_FEATURES] = {{{means_str}}};\n")
        f.write(f"static const float SCALER_SCALE[NUM_FEATURES] = {{{scales_str}}};\n")
        f.write(f"static const float SCALER_INV_SCALE[NUM_FEATURES] = {{{inv_scales_str}}};\n\n")
        f.write("/* Inline helper to normalize raw ADC features on-target */\n")
        f.write("static inline void normalize_features(const float *raw_in, float *norm_out) {\n")
        f.write(f"    for (int i = 0; i < NUM_FEATURES; i++) {{\n")
        f.write("        norm_out[i] = (raw_in[i] - SCALER_MEAN[i]) * SCALER_INV_SCALE[i];\n")
        f.write("    }\n")
        f.write("}\n\n")
        f.write("#endif /* SCALER_PARAMS_H */\n")

    print(f"Scaler parameters exported to C header: {header_path}")


def generate_xcube_ai_test_vectors(num_samples=25):
    """
    Generates test vectors for X-CUBE-AI validation:
    1. .npy / .npz format (ST-recommended for automated validation)
    2. .csv format (CubeMX GUI validation)
    3. .h format (C header for bare-metal testing on STM32)
    Features are the 8 windowed statistics: V_mean, V_std, V_rms, V_p2p, I_mean, I_std, I_rms, I_p2p
    """
    print(f"\nGenerating {num_samples} test vectors for X-CUBE-AI validation...")
    _, X_test, _, y_test = load_and_preprocess_data()

    indices = np.random.RandomState(config.RANDOM_STATE).choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    test_inputs = X_test[indices].astype(np.float32)
    test_labels = y_test[indices].astype(np.int32)

    # Calculate model predictions for expected reference outputs
    pt_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.pt")
    base_model = FaultPredictionMLP(input_dim=test_inputs.shape[1], num_classes=config.NUM_CLASSES)
    base_model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    base_model.eval()

    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(test_inputs)
        logits = base_model(inputs_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy().astype(np.float32)
        predicted_classes = np.argmax(probabilities, axis=1).astype(np.int32)

    vectors_dir = os.path.join(config.MODEL_SAVE_DIR, "xcube_ai_test_vectors")
    os.makedirs(vectors_dir, exist_ok=True)

    # 1. NumPy format (.npy & .npz)
    input_npy = os.path.join(vectors_dir, "test_inputs.npy")
    output_npy = os.path.join(vectors_dir, "test_outputs.npy")
    np.save(input_npy, test_inputs)
    np.save(output_npy, probabilities)

    npz_path = os.path.join(vectors_dir, "test_data.npz")
    np.savez(npz_path, input_data=test_inputs, output_probabilities=probabilities, ground_truth=test_labels)
    print(f"Saved NumPy validation vectors: {input_npy}, {output_npy}, {npz_path}")

    # 2. CSV format (.csv)
    input_csv = os.path.join(vectors_dir, "test_inputs.csv")
    output_csv = os.path.join(vectors_dir, "test_outputs.csv")
    np.savetxt(input_csv, test_inputs, delimiter=",", fmt="%.8f")
    np.savetxt(output_csv, probabilities, delimiter=",", fmt="%.8f")
    print(f"Saved CSV validation vectors: {input_csv}, {output_csv}")

    # 3. C header format (test_vectors.h for STM32CubeIDE)
    header_path = os.path.join(vectors_dir, "test_vectors.h")
    with open(header_path, "w") as f:
        f.write("/* Reference test vectors for STM32 validation */\n")
        f.write("#ifndef TEST_VECTORS_H\n")
        f.write("#define TEST_VECTORS_H\n\n")
        f.write(f"#define TEST_VECTOR_COUNT {len(test_inputs)}\n")
        f.write(f"#define TEST_FEATURE_DIM {test_inputs.shape[1]}\n")
        f.write(f"#define TEST_NUM_CLASSES {config.NUM_CLASSES}\n\n")

        # Inputs array
        f.write("static const float TEST_INPUTS[TEST_VECTOR_COUNT][TEST_FEATURE_DIM] = {\n")
        for sample in test_inputs:
            row = ", ".join([f"{v:.8f}f" for v in sample])
            f.write(f"    {{{row}}},\n")
        f.write("};\n\n")

        # Expected classes
        classes_str = ", ".join([str(c) for c in predicted_classes])
        f.write(f"static const int TEST_EXPECTED_PREDICTIONS[TEST_VECTOR_COUNT] = {{{classes_str}}};\n\n")

        # Ground truth labels
        labels_str = ", ".join([str(c) for c in test_labels])
        f.write(f"static const int TEST_GROUND_TRUTH_LABELS[TEST_VECTOR_COUNT] = {{{labels_str}}};\n\n")

        # Expected probabilities
        f.write("static const float TEST_EXPECTED_PROBABILITIES[TEST_VECTOR_COUNT][TEST_NUM_CLASSES] = {\n")
        for probs in probabilities:
            row_str = ", ".join([f"{p:.6f}f" for p in probs])
            f.write(f"    {{{row_str}}},\n")
        f.write("};\n\n")
        f.write("#endif /* TEST_VECTORS_H */\n")

    print(f"Saved C header test vectors: {header_path}")


def main():
    print("=== Exporting Models and Test Vectors for X-CUBE-AI Deployment ===")
    
    pt_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.pt")
    # pipeline.py now uses clean_name from "Baseline 2a - Random Forest" -> "Random_Forest"
    rf_path = os.path.join(config.MODEL_SAVE_DIR, "Random_Forest.joblib")
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, "scaler.joblib")

    input_dim = len(config.FEATURE_COLS)  # 8 windowed features
    print(f"Input dimensions: {input_dim} features | Classes: {config.NUM_CLASSES}")

    # 1. Export PyTorch MLP to ONNX
    mlp_onnx_path = os.path.join(config.MODEL_SAVE_DIR, "FaultPredictionMLP.onnx")
    export_pytorch_to_onnx(pt_path, mlp_onnx_path, input_dim=input_dim, num_classes=config.NUM_CLASSES)

    # 2. Export Random Forest to ONNX (ONNX-ML)
    rf_onnx_path = os.path.join(config.MODEL_SAVE_DIR, "Random_Forest.onnx")
    export_random_forest_to_onnx(rf_path, rf_onnx_path, input_dim=input_dim)

    # 3. Export Random Forest to standalone C (m2cgen fallback)
    rf_c_path = os.path.join(config.MODEL_SAVE_DIR, "random_forest_model.c")
    rf_h_path = os.path.join(config.MODEL_SAVE_DIR, "random_forest_model.h")
    export_random_forest_to_c(rf_path, rf_c_path, rf_h_path)

    # 4. Export Scaler parameters to C header
    scaler_h_path = os.path.join(config.MODEL_SAVE_DIR, "scaler_params.h")
    export_scaler_header(scaler_path, scaler_h_path)

    # 5. Generate validation test vectors
    generate_xcube_ai_test_vectors(num_samples=25)

    print("\nAll export artifacts generated successfully!")

if __name__ == "__main__":
    main()
