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

if __name__ == "__main__":
    print("=== Fault Prediction Inference Demo ===")
    
    # Dummy sample: [Load Voltage, Load Current]
    # Let's simulate a random test case
    dummy_sample = [5.1, 2.0]
    
    print(f"Input features: {dummy_sample}")
    
    model_types = ["Random_Forest", "XGBoost", "SVM", "PyTorch_MLP"]
    
    for mt in model_types:
        try:
            pred_id, pred_name, conf = predict_fault(dummy_sample, model_type=mt)
            print(f"Model: {mt:<15} | Predicted: {pred_name:<20} | Confidence: {conf*100:.2f}%")
        except FileNotFoundError as e:
            print(e)
