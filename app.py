import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import config
from data_loader import extract_window_features
from models import get_stacking_hybrid_model

app = Flask(__name__, template_folder='templates', static_folder='static')

LOADED_MODEL = None
SCALER = None

HYBRID_MODEL_NAME = "Hybrid Stacking Ensemble (LR + Calibrated Linear SVC + Regularized RF + XGBoost)"

HEALTH_IMPACT = {
    0: 100, # Healthy
    1: 75,  # ESR Mild
    2: 20,  # ESR Severe
    3: 40,  # Switch Deg
    4: 55   # Gate Deg
}

def load_scaler():
    global SCALER
    if SCALER is None:
        scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
        if os.path.exists(scaler_path):
            SCALER = joblib.load(scaler_path)
    return SCALER

def get_hybrid_model():
    global LOADED_MODEL
    if LOADED_MODEL is None:
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.STACKING_MODEL_NAME}.joblib")
        if os.path.exists(model_path):
            LOADED_MODEL = joblib.load(model_path)
        else:
            print("Training Hybrid 1 Stacking Ensemble for server initialization...")
            from data_loader import load_and_preprocess_data
            X_tr, _, y_tr, _ = load_and_preprocess_data()
            LOADED_MODEL = get_stacking_hybrid_model()
            LOADED_MODEL.fit(X_tr, y_tr)
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            joblib.dump(LOADED_MODEL, model_path)
    return LOADED_MODEL

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json or {}
        features = data.get("features", {})
        
        # Base 8 features
        raw_names = ['V_mean', 'V_std', 'V_rms', 'V_p2p', 'I_mean', 'I_std', 'I_rms', 'I_p2p']
        raw_vals = [float(features.get(f, 0.0)) for f in raw_names]
        
        v_mean, v_std, v_rms, v_p2p, i_mean, i_std, i_rms, i_p2p = raw_vals
        eps = 1e-6
        v_ripple_ratio = float(features.get('V_ripple_ratio', v_p2p / (abs(v_mean) + eps)))
        i_ripple_ratio = float(features.get('I_ripple_ratio', i_p2p / (abs(i_mean) + eps)))
        r_apparent = float(features.get('R_apparent', v_mean / (abs(i_mean) + eps)))
        v_crest_factor = float(features.get('V_crest_factor', v_p2p / (2.0 * np.sqrt(2.0) * v_rms + eps)))
        p_ratio = float(features.get('P_ratio', (v_mean * i_mean) / (v_rms * i_rms + eps)))
        
        sample = raw_vals + [v_ripple_ratio, i_ripple_ratio, r_apparent, v_crest_factor, p_ratio]
        sample_arr = np.array(sample).reshape(1, -1)
        
        scaler = load_scaler()
        if scaler is not None:
            sample_scaled = scaler.transform(sample_arr)
        else:
            sample_scaled = sample_arr
            
        model = get_hybrid_model()
        predicted_class = int(model.predict(sample_scaled)[0])
        predicted_label = config.CLASSES.get(predicted_class, f"Class {predicted_class}")
        
        probabilities = {}
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(sample_scaled)[0]
                classes = getattr(model, "classes_", list(range(len(probs))))
                for cls_idx, prob in zip(classes, probs):
                    cls_name = config.CLASSES.get(int(cls_idx), f"Class {cls_idx}")
                    probabilities[cls_name] = round(float(prob) * 100, 2)
                confidence = probabilities.get(predicted_label, 0.0)
            except Exception:
                probabilities[predicted_label] = 100.0
                confidence = 100.0
        else:
            probabilities[predicted_label] = 100.0
            confidence = 100.0
            
        health_index = HEALTH_IMPACT.get(predicted_class, 100)
        
        return jsonify({
            "model_name": HYBRID_MODEL_NAME,
            "predicted_class_id": predicted_class,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 2),
            "health_index": health_index,
            "probabilities": probabilities,
            "input_features": dict(zip(config.FEATURE_COLS, sample))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename selected"}), 400
            
        df = pd.read_csv(file)
        
        # Normalize column names
        col_map = {}
        for col in df.columns:
            c_clean = col.strip()
            if 'voltage' in c_clean.lower() or c_clean == 'Vout' or c_clean == 'V':
                col_map[col] = 'Load Voltage'
            elif 'current' in c_clean.lower() or c_clean == 'IL' or c_clean == 'I':
                col_map[col] = 'Load Current'
            elif 'time' in c_clean.lower() or c_clean.startswith('Time'):
                col_map[col] = 'Time / s'
                
        df = df.rename(columns=col_map)
        
        if 'Load Voltage' not in df.columns or 'Load Current' not in df.columns:
            return jsonify({
                "error": f"Invalid CSV columns. Required columns: 'Load Voltage' and 'Load Current' (or 'Time / s', 'Load Current', 'Load Voltage'). Found columns: {list(df.columns)}"
            }), 400
            
        # Extract features over sliding windows
        window_size = config.WINDOW_SIZE
        features_df = extract_window_features(df, window_size=window_size, skip_transient=True)
        
        if len(features_df) == 0:
            return jsonify({"error": f"Dataset has too few rows ({len(df)}) for window size = {window_size}."}), 400
            
        X = features_df[config.FEATURE_COLS].values
        
        scaler = load_scaler()
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        model = get_hybrid_model()
        predictions = model.predict(X_scaled)
        
        # Calculate window counts & percentages
        pred_series = pd.Series(predictions)
        counts = pred_series.value_counts().to_dict()
        
        window_breakdown = []
        total_windows = len(predictions)
        total_health_score = 0
        
        for cls_id in range(config.NUM_CLASSES):
            cnt = int(counts.get(cls_id, 0))
            pct = round((cnt / total_windows) * 100, 1)
            cls_name = config.CLASSES.get(cls_id, f"Class {cls_id}")
            health_weight = HEALTH_IMPACT.get(cls_id, 100)
            total_health_score += (cnt * health_weight)
            
            window_breakdown.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "count": cnt,
                "percentage": pct
            })
            
        # In power electronics, if hardware degradation (ESR, Switch, Gate) persists in >= 10% of windows,
        # prioritize that fault in primary diagnosis so regulated steady-state windows do not mask active degradation.
        fault_counts = {cls_id: cnt for cls_id, cnt in counts.items() if cls_id != 0}
        if fault_counts:
            top_fault_id = max(fault_counts, key=fault_counts.get)
            top_fault_cnt = fault_counts[top_fault_id]
            if (top_fault_cnt / total_windows) >= 0.10:
                diagnosis_class_id = int(top_fault_id)
            else:
                diagnosis_class_id = int(pred_series.mode()[0])
        else:
            diagnosis_class_id = int(pred_series.mode()[0])

        diagnosis_class_name = config.CLASSES.get(diagnosis_class_id, f"Class {diagnosis_class_id}")

        # If a critical component fault is actively diagnosed, adjust overall health score accordingly
        if diagnosis_class_id != 0:
            fault_base_health = HEALTH_IMPACT.get(diagnosis_class_id, 50)
            overall_health_index = min(round(total_health_score / total_windows, 1), fault_base_health)
        else:
            overall_health_index = round(total_health_score / total_windows, 1)

        majority_class_id = diagnosis_class_id
        majority_class_name = diagnosis_class_name
        
        # Prepare window timeline data for graphing
        timeline = []
        for idx, pred in enumerate(predictions):
            timeline.append({
                "window": idx + 1,
                "time_s": round((idx + 1) * 0.01, 3),
                "predicted_class_id": int(pred),
                "predicted_label": config.CLASSES.get(int(pred), f"Class {pred}")
            })
            
        return jsonify({
            "filename": file.filename,
            "total_rows": len(df),
            "total_windows": total_windows,
            "overall_health_index": overall_health_index,
            "majority_fault_id": majority_class_id,
            "majority_fault_label": majority_class_name,
            "breakdown": window_breakdown,
            "timeline": timeline
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Buck Converter Health & Fault Prediction Dashboard...")
    app.run(host='0.0.0.0', port=5000, debug=True)
