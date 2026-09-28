"""
CLI Inference Script for Fault Diagnostics in DC-DC Buck-Boost Converters.
Uses the primary model: Hybrid 1 - Stacking Ensemble (QDA + Quadratic SVM + RF + XGBoost -> Logistic Regression).
"""

import os
import joblib
import numpy as np
import pandas as pd
import config
from data_loader import extract_window_features

def load_scaler():
    """Loads the saved StandardScaler."""
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run pipeline.py first.")
    return joblib.load(scaler_path)

def load_hybrid_model():
    """Loads the trained Hybrid 1 Stacking Ensemble."""
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.STACKING_MODEL_NAME}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stacking model not found at {model_path}. Run pipeline.py first.")
    return joblib.load(model_path)

def expand_features_if_needed(sample):
    """
    Expands an 8-element raw window feature vector into the full 12-element
    feature vector by deriving the invariant power electronic metrics.
    """
    sample = np.array(sample, dtype=float).flatten()
    if len(sample) == len(config.FEATURE_COLS):
        return sample.reshape(1, -1)
    
    if len(sample) == 8:
        # Expected: V_mean, V_std, V_rms, V_p2p, I_mean, I_std, I_rms, I_p2p
        v_mean, v_std, v_rms, v_p2p, i_mean, i_std, i_rms, i_p2p = sample
        eps = 1e-6
        v_ripple_ratio = v_p2p / (abs(v_mean) + eps)
        i_ripple_ratio = i_p2p / (abs(i_mean) + eps)
        r_apparent = v_mean / (abs(i_mean) + eps)
        v_crest_factor = v_p2p / (2.0 * np.sqrt(2.0) * v_rms + eps)
        p_ratio = (v_mean * i_mean) / (v_rms * i_rms + eps)
        
        full_sample = np.array([
            v_mean, v_std, v_rms, v_p2p,
            i_mean, i_std, i_rms, i_p2p,
            v_ripple_ratio, i_ripple_ratio, r_apparent, v_crest_factor,
            p_ratio
        ])
        return full_sample.reshape(1, -1)
    
    raise ValueError(f"Expected 8 or {len(config.FEATURE_COLS)} feature values, got {len(sample)}.")

def predict_fault(sample):
    """
    Predicts the fault class for a single feature window vector using Hybrid 1 Stacking Ensemble.
    """
    sample_arr = expand_features_if_needed(sample)
    scaler = load_scaler()
    sample_scaled = scaler.transform(sample_arr)
    
    model = load_hybrid_model()
    predicted_class = int(model.predict(sample_scaled)[0])
    
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample_scaled)[0]
        confidence = float(probabilities[predicted_class])
        
    class_name = config.CLASSES.get(predicted_class, f"Class {predicted_class}")
    return predicted_class, class_name, confidence

def predict_csv_file(csv_path):
    """
    Loads a raw Simulink CSV file, extracts window features,
    and returns predicted fault classes using Hybrid 1 Stacking Ensemble.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    print(f"\n--- Running Hybrid 1 Stacking Ensemble Inference on: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    if 'Load Voltage' not in df.columns or 'Load Current' not in df.columns:
        raise ValueError("CSV must contain 'Load Voltage' and 'Load Current' columns.")
        
    # Extract window features (skipping cold-start transient)
    features_df = extract_window_features(df, window_size=config.WINDOW_SIZE, skip_transient=True)
    X = features_df[config.FEATURE_COLS].values
    
    scaler = load_scaler()
    X_scaled = scaler.transform(X)
    
    model = load_hybrid_model()
    predictions = model.predict(X_scaled)
    
    pred_counts = pd.Series(predictions).value_counts().to_dict()
    print(f"Total window samples evaluated: {len(predictions)}")
    print("Class Distribution Breakdown:")
    for class_id, count in sorted(pred_counts.items()):
        name = config.CLASSES.get(class_id, f"Class {class_id}")
        percentage = (count / len(predictions)) * 100
        print(f"  - [{class_id}] {name:<16}: {count:<5} windows ({percentage:.1f}%)")
        
    majority_class_id = int(pd.Series(predictions).mode()[0])
    majority_name = config.CLASSES.get(majority_class_id, f"Class {majority_class_id}")
    print(f"\nDominant Diagnosis: {majority_name} (Class ID: {majority_class_id})")
    
    return predictions, majority_name

if __name__ == "__main__":
    print("=" * 65)
    print(" Fault Prediction CLI - Hybrid 1: Stacking Ensemble")
    print("=" * 65)
    
    # 1. Single Window Test Sample (Healthy nominal steady state)
    dummy_sample = [4.973, 0.0045, 4.973, 0.015, 1.428, 0.150, 1.437, 0.392]
    print(f"\n1. Testing Single Window Feature Vector ({len(dummy_sample)} raw features):")
    try:
        pred_id, pred_name, conf = predict_fault(dummy_sample)
        print(f"   Predicted Diagnosis : {pred_name} (Class {pred_id})")
        print(f"   Confidence          : {conf * 100:.2f}%")
    except Exception as e:
        print(f"   Prediction notice: {e}")
        
    # 2. Test full CSV file if available
    sample_csv = "dataset_v5/unit3_esr_severe_r2.5.csv"
    if os.path.exists(sample_csv):
        try:
            predict_csv_file(sample_csv)
        except Exception as e:
            print(f"CSV inference notice: {e}")
