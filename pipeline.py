"""
Training and Evaluation Pipeline for Fault Prediction in DC-DC Buck-Boost Converters.
Primary Architecture: Hybrid 1 - Stacking Ensemble (Linear Margin SVM + Multiclass LR + Regularized RF + XGBoost).
"""

import os
import sys
import joblib
from sklearn.metrics import classification_report
import config
from data_loader import load_and_preprocess_data
from models import get_stacking_hybrid_model
from evaluate import evaluate_model, plot_confusion_matrix, print_summary_table

# Ensure stdout flushes immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

def main():
    print("=" * 70)
    print("   DC-DC Converter Fault Prediction Pipeline")
    print("   Primary Model: Hybrid 1 - Stacking Ensemble")
    print("   (QDA + Quadratic SVM + Random Forest + XGBoost -> Logistic Regression)")
    print("=" * 70)
    
    # 1. Load and preprocess data (Group Split across test unit)
    print("\n--- Step 1: Ingestion & Feature Engineering ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Testing samples (Unseen {config.TEST_UNIT}): {X_test.shape[0]:,}")
    print(f"Engineered Feature Dimensions: {X_train.shape[1]}")
    print(f"Features: {config.FEATURE_COLS}")
    
    # 2. Build and Train Hybrid 1 Stacking Ensemble
    print("\n--- Step 2: Training Hybrid 1 Stacking Ensemble ---")
    model_name = "Hybrid 1 - Stacking Ensemble"
    model = get_stacking_hybrid_model()
    
    print("Fitting Hybrid 1 Ensemble across calibrated linear models and regularized trees...")
    model.fit(X_train, y_train)
    print("Fitting completed successfully.")
    
    # 3. Evaluation on unseen unit
    print("\n--- Step 3: Evaluation on Unseen Test Set ---")
    y_pred = model.predict(X_test)
    
    res = evaluate_model(model_name, y_test, y_pred)
    print_summary_table([res])
    
    # Detailed per-class classification report
    class_names = [config.CLASSES.get(i, f"Class {i}") for i in range(config.NUM_CLASSES)]
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # 4. Generate and save confusion matrix
    plot_confusion_matrix(model_name, y_test, y_pred)
    
    # 5. Serialize Model
    save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.STACKING_MODEL_NAME}.joblib")
    joblib.dump(model, save_path)
    print(f"\nModel successfully saved to: {save_path}")
    print("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
