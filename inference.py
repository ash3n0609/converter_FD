import os
import joblib
import numpy as np
import torch
import config
from models import FaultPredictionMLP

def load_scaler():
    """Loads the saved StandardScaler."""
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run pipeline.py first.")
    return joblib.load(scaler_path)

def load_scikit_model(model_name="Random_Forest"):
    """Loads a saved Scikit-Learn or XGBoost model."""
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run pipeline.py first.")
    return joblib.load(model_path)

def load_pytorch_model(input_dim):
    """Loads the saved PyTorch model."""
    model_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model not found at {model_path}. Run pipeline.py first.")
    
    model = FaultPredictionMLP(input_dim, config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_fault(sample, model_type="Random_Forest"):
    """
    Predicts the fault class for a given new sample.
    sample: A list or 1D numpy array of features: [Vin, Vout, IL, duty_cycle, switch_temp]
    """
    sample = np.array(sample).reshape(1, -1)
    
    # Preprocess
    scaler = load_scaler()
    sample_scaled = scaler.transform(sample)
    
    predicted_class = -1
    confidence = 0.0
    
    if model_type == "PyTorch_MLP":
        model = load_pytorch_model(input_dim=sample.shape[1])
        sample_tensor = torch.FloatTensor(sample_scaled)
        with torch.no_grad():
            outputs = model(sample_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
    else:
        model = load_scikit_model(model_type)
        predicted_class = model.predict(sample_scaled)[0]
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample_scaled)[0]
            confidence = probabilities[predicted_class]
            
    class_name = config.CLASSES.get(predicted_class, f"Unknown ({predicted_class})")
    
    return predicted_class, class_name, confidence

def predict_csv_file(csv_path, model_type="SVM_RBF"):
    """
    Loads a raw Simulink CSV file, extracts window features,
    and returns predicted fault classes for the entire file.
    """
    import pandas as pd
    from data_loader import extract_window_features
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    print(f"\n--- Running Inference on CSV File: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    if 'Load Voltage' not in df.columns or 'Load Current' not in df.columns:
        raise ValueError("CSV must contain 'Load Voltage' and 'Load Current' columns.")
        
    # Extract window features
    features_df = extract_window_features(df, window_size=config.WINDOW_SIZE)
    X = features_df[config.FEATURE_COLS].values
    
    # Scale features
    scaler = load_scaler()
    X_scaled = scaler.transform(X)
    
    # Load model and predict
    if model_type == "PyTorch_MLP":
        model = load_pytorch_model(input_dim=X.shape[1])
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, 1)
            predictions = preds.numpy()
    else:
        model = load_scikit_model(model_type)
        predictions = model.predict(X_scaled)
        
    # Summarize predictions
    pred_counts = pd.Series(predictions).value_counts().to_dict()
    print("Prediction Summary across window samples:")
    for class_id, count in pred_counts.items():
        name = config.CLASSES.get(class_id, f"Class {class_id}")
        percentage = (count / len(predictions)) * 100
        print(f"  - {name:<18}: {count:<5} windows ({percentage:.1f}%)")
        
    majority_class_id = pd.Series(predictions).mode()[0]
    majority_name = config.CLASSES.get(majority_class_id, f"Class {majority_class_id}")
    print(f"Overall Predicted File Label ({model_type}): {majority_name}")
    
    return predictions, majority_name

if __name__ == "__main__":
    print("=== Fault Prediction Inference Demo ===")
    
    # Sample 1: Single Window Test Sample
    dummy_sample = [5.1, 0.05, 5.1002, 0.2, 2.0, 0.02, 2.0001, 0.1]
    print(f"Input window features ({len(config.FEATURE_COLS)} dimensions): {dummy_sample}")
    
    model_types = ["Random_Forest", "XGBoost", "SVM_RBF", "PyTorch_MLP"]
    
    for mt in model_types:
        try:
            pred_id, pred_name, conf = predict_fault(dummy_sample, model_type=mt)
            print(f"Model: {mt:<15} | Predicted: {pred_name:<20} | Confidence: {conf*100:.2f}%")
        except FileNotFoundError as e:
            print(e)
            
    # Sample 2: Test an entire raw CSV file
    sample_csv = "dataset_v5/unit3_esr_severe_r2.5.csv"
    if os.path.exists(sample_csv):
        predict_csv_file(sample_csv, model_type="SVM_RBF")

