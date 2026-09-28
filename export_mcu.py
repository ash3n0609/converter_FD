import os
import joblib
import numpy as np
import config
from data_loader import load_and_preprocess_data
from models import get_mcu_rf_model
from evaluate import evaluate_model

def tree_to_c(tree, feature_names, tree_id):
    """
    Transpiles a single Scikit-Learn DecisionTree into a pure C function.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    lines = []
    lines.append(f"static void evaluate_tree_{tree_id}(const float features[8], float votes[{config.NUM_CLASSES}]) {{")
    
    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != -2:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            feat_idx = feature_names.index(name)
            lines.append(f"{indent}if (features[{feat_idx}] <= {threshold:.6f}f) {{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            # Leaf node: value is array of sample counts per class
            values = tree_.value[node][0]
            norm_values = values / np.sum(values)
            for class_idx, val in enumerate(norm_values):
                if val > 0:
                    lines.append(f"{indent}votes[{class_idx}] += {val:.6f}f;")
                    
    recurse(0, 1)
    lines.append("}\n")
    return "\n".join(lines)

def generate_c_classifier(rf_model, scaler, feature_names, output_dir="mcu"):
    """
    Generates standalone fault_classifier.h and fault_classifier.c files.
    Includes built-in feature scaling and multi-tree decision voting.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate fault_classifier.h
    header_content = f"""/*
 * Fault Classifier for Buck-Boost DC-DC Converter
 * Automatically generated standalone MCU code.
 *
 * Target: 32-bit Microcontrollers (STM32, TI C2000, ESP32, ARM Cortex-M)
 * Memory Footprint: Flash < 30 KB, SRAM < 2 KB
 * Execution Time: < 50 microseconds @ 100MHz
 */

#ifndef FAULT_CLASSIFIER_H
#define FAULT_CLASSIFIER_H

#ifdef __cplusplus
extern "C" {{
#endif

#define NUM_FEATURES {len(feature_names)}
#define NUM_CLASSES {config.NUM_CLASSES}
#define WINDOW_SIZE {config.WINDOW_SIZE}

// Fault Class Enums
typedef enum {{
    FAULT_HEALTHY = 0,
    FAULT_ESR_MILD = 1,
    FAULT_ESR_SEVERE = 2,
    FAULT_SWITCH_DEG = 3,
    FAULT_GATE_DEG = 4,
    FAULT_SENSOR_FAULT = 5
}} FaultClass;

// Function Prototypes
const char* get_fault_name(int fault_code);
int predict_fault_raw_features(const float raw_features[NUM_FEATURES]);
int predict_fault_scaled_features(const float scaled_features[NUM_FEATURES]);
void get_class_probabilities(const float raw_features[NUM_FEATURES], float out_probs[NUM_CLASSES]);

#ifdef __cplusplus
}}
#endif

#endif // FAULT_CLASSIFIER_H
"""
    header_path = os.path.join(output_dir, "fault_classifier.h")
    with open(header_path, "w") as f:
        f.write(header_content)
        
    # 2. Generate fault_classifier.c
    c_lines = []
    c_lines.append("""/*
 * Standalone Implementation of MCU Fault Classifier
 */

#include "fault_classifier.h"
#include <math.h>

// Feature Scaling Parameters (StandardScaler)
""")
    
    # Write Scaler means and scales
    c_lines.append(f"static const float SCALER_MEAN[{len(feature_names)}] = {{")
    c_lines.append("    " + ", ".join([f"{m:.6f}f" for m in scaler.mean_]))
    c_lines.append("};\n")
    
    c_lines.append(f"static const float SCALER_SCALE[{len(feature_names)}] = {{")
    c_lines.append("    " + ", ".join([f"{s:.6f}f" for s in scaler.scale_]))
    c_lines.append("};\n")
    
    # Class names array
    c_lines.append("static const char* FAULT_NAMES[NUM_CLASSES] = {")
    for i in range(config.NUM_CLASSES):
        name = config.CLASSES.get(i, f"Class_{i}")
        c_lines.append(f'    "{name}",')
    c_lines.append("};\n")
    
    # Transpile individual trees
    for idx, estimator in enumerate(rf_model.estimators_):
        tree_code = tree_to_c(estimator, feature_names, idx)
        c_lines.append(tree_code)
        
    # Main prediction logic
    c_lines.append(f"""
const char* get_fault_name(int fault_code) {{
    if (fault_code >= 0 && fault_code < NUM_CLASSES) {{
        return FAULT_NAMES[fault_code];
    }}
    return "UNKNOWN_FAULT";
}}

void get_class_probabilities(const float raw_features[NUM_FEATURES], float out_probs[NUM_CLASSES]) {{
    float scaled[NUM_FEATURES];
    float votes[NUM_CLASSES] = {{0.0f}};
    
    // Step 1: Scale raw window features
    for (int i = 0; i < NUM_FEATURES; i++) {{
        scaled[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }}
    
    // Step 2: Accumulate votes across all {len(rf_model.estimators_)} trees
""")
    
    for idx in range(len(rf_model.estimators_)):
        c_lines.append(f"    evaluate_tree_{idx}(scaled, votes);")
        
    c_lines.append(f"""
    // Step 3: Normalize to probability distribution
    float total_votes = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {{
        total_votes += votes[c];
    }}
    if (total_votes > 0.0f) {{
        for (int c = 0; c < NUM_CLASSES; c++) {{
            out_probs[c] = votes[c] / total_votes;
        }}
    }}
}}

int predict_fault_raw_features(const float raw_features[NUM_FEATURES]) {{
    float probs[NUM_CLASSES];
    get_class_probabilities(raw_features, probs);
    
    int max_idx = 0;
    float max_prob = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {{
        if (probs[c] > max_prob) {{
            max_prob = probs[c];
            max_idx = c;
        }}
    }}
    return max_idx;
}}

int predict_fault_scaled_features(const float scaled_features[NUM_FEATURES]) {{
    float votes[NUM_CLASSES] = {{0.0f}};
""")
    for idx in range(len(rf_model.estimators_)):
        c_lines.append(f"    evaluate_tree_{idx}(scaled_features, votes);")
        
    c_lines.append("""
    int max_idx = 0;
    float max_votes = votes[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (votes[c] > max_votes) {
            max_votes = votes[c];
            max_idx = c;
        }
    }
    return max_idx;
}
""")
    
    c_path = os.path.join(output_dir, "fault_classifier.c")
    with open(c_path, "w") as f:
        f.write("\n".join(c_lines))
        
    # 3. Generate MCU Integration Example file (mcu_example.c)
    example_code = """/*
 * Microcontroller Integration Example (STM32 / TI C2000 / ESP32)
 * Demonstrates real-time ADC sampling, window feature extraction,
 * and fault classification.
 */

#include <stdio.h>
#include <math.h>
#include "fault_classifier.h"

// Ring buffer to hold 50 samples of Load Voltage and Load Current
static float v_buffer[WINDOW_SIZE];
static float i_buffer[WINDOW_SIZE];
static int buffer_index = 0;

void process_adc_sample(float v_sample, float i_sample) {
    v_buffer[buffer_index] = v_sample;
    i_buffer[buffer_index] = i_sample;
    buffer_index++;
    
    // Every WINDOW_SIZE samples (50 points), run feature extraction & inference
    if (buffer_index >= WINDOW_SIZE) {
        buffer_index = 0; // Reset index
        
        // Compute 8 Window Statistics
        float v_sum = 0.0f, v_sq_sum = 0.0f;
        float i_sum = 0.0f, i_sq_sum = 0.0f;
        float v_min = v_buffer[0], v_max = v_buffer[0];
        float i_min = i_buffer[0], i_max = i_buffer[0];
        
        for (int k = 0; k < WINDOW_SIZE; k++) {
            float v = v_buffer[k];
            float i = i_buffer[k];
            
            v_sum += v;
            v_sq_sum += v * v;
            if (v < v_min) v_min = v;
            if (v > v_max) v_max = v;
            
            i_sum += i;
            i_sq_sum += i * i;
            if (i < i_min) i_min = i;
            if (i > i_max) i_max = i;
        }
        
        float v_mean = v_sum / WINDOW_SIZE;
        float v_rms = sqrtf(v_sq_sum / WINDOW_SIZE);
        float v_p2p = v_max - v_min;
        
        // Variance for standard deviation
        float v_var = (v_sq_sum / WINDOW_SIZE) - (v_mean * v_mean);
        float v_std = sqrtf(v_var > 0.0f ? v_var : 0.0f);
        
        float i_mean = i_sum / WINDOW_SIZE;
        float i_rms = sqrtf(i_sq_sum / WINDOW_SIZE);
        float i_p2p = i_max - i_min;
        float i_var = (i_sq_sum / WINDOW_SIZE) - (i_mean * i_mean);
        float i_std = sqrtf(i_var > 0.0f ? i_var : 0.0f);
        
        float features[NUM_FEATURES] = {
            v_mean, v_std, v_rms, v_p2p,
            i_mean, i_std, i_rms, i_p2p
        };
        
        // Run inference
        int predicted_fault = predict_fault_raw_features(features);
        const char* fault_name = get_fault_name(predicted_fault);
        
        if (predicted_fault != FAULT_HEALTHY) {
            printf("[WARNING] Converter Fault Detected! Class %d: %s\\n", predicted_fault, fault_name);
            // Trigger MCU PWM Shutdown / Relay Protection here
        } else {
            printf("[OK] Converter Operating Normally (Healthy)\\n");
        }
    }
}

int main() {
    printf("=== MCU Fault Classifier Test Run ===\\n");
    // Dummy sample input (Simulating 50 ADC voltage/current readings)
    for (int s = 0; s < WINDOW_SIZE; s++) {
        process_adc_sample(24.1f, 2.05f);
    }
    return 0;
}
"""
    example_path = os.path.join(output_dir, "mcu_example.c")
    with open(example_path, "w") as f:
        f.write(example_code)
        
    print(f"Successfully exported MCU C files to directory: '{output_dir}'")
    print(f"  - Header file: {header_path}")
    print(f"  - Implementation file: {c_path}")
    print(f"  - MCU Integration Example: {example_path}")

def main():
    print("=== MCU Model Training & Standalone C Code Export ===")
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. Train MCU Compact Random Forest
    print("\nTraining Compact MCU Random Forest (30 trees, max_depth=5)...")
    mcu_model = get_mcu_rf_model()
    mcu_model.fit(X_train, y_train)
    
    # 3. Evaluate model
    y_pred = mcu_model.predict(X_test)
    metrics = evaluate_model("MCU Compact Random Forest", y_test, y_pred)
    print("\n--- Evaluation Results on Reserved Test Unit (unit3) ---")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"  {k:<12}: {v:.4f}")
            
    # Save scikit-learn model & scaler
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    model_path = os.path.join(config.MODEL_SAVE_DIR, "MCU_Random_Forest.joblib")
    joblib.dump(mcu_model, model_path)
    print(f"\nSaved MCU Random Forest model to {model_path}")
    
    # 4. Generate C code
    print("\nTranspiling Random Forest to C source code...")
    generate_c_classifier(mcu_model, scaler, config.FEATURE_COLS, output_dir="mcu")

if __name__ == "__main__":
    main()
