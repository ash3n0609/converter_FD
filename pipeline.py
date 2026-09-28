import os
import joblib
import config
from data_loader import load_and_preprocess_data
from models import (
    get_trivial_model, get_rf_model, get_xgb_model, get_svm_model, 
    get_qda_model, get_quadratic_svm_model, get_stacking_hybrid_model,
    HierarchicalCascadeHybrid, train_pytorch_model, predict_pytorch_model,
    export_pytorch_to_onnx, HAS_TORCH
)
from evaluate import evaluate_model, plot_confusion_matrix, print_summary_table, plot_feature_importance

def main():
    print("=== Starting ML Pipeline for Buck-Boost Converter Fault Prediction ===")
    
    # 1. Load and preprocess data (Group Split to avoid data leakage)
    print("\n--- Step 1: Data Loading & Preprocessing ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # 2. Define Baselines & Hybrid Models
    print("\n--- Step 2: Training & Evaluating Models & Hybrids ---")
    
    models = {
        "Baseline 1 - Trivial (Majority)": get_trivial_model(),
        "Baseline 2a - Random Forest": get_rf_model(),
        "Baseline 2b - XGBoost": get_xgb_model(),
        "Baseline 2c - SVM (RBF)": get_svm_model(),
        "Baseline 2d - QDA": get_qda_model(),
        "Baseline 2e - SVM (Quadratic)": get_quadratic_svm_model(),
        "Hybrid 1 - Stacking Ensemble": get_stacking_hybrid_model(),
        "Hybrid 2 - Hierarchical Cascade": HierarchicalCascadeHybrid()
    }
    
    results = []
    
    # Train Scikit-Learn / XGBoost / Hybrid models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        res = evaluate_model(name, y_test, y_pred)
        results.append(res)
        
        # Plot and Save
        plot_confusion_matrix(name, y_test, y_pred)
        plot_feature_importance(name, model)
        
        # Save model
        clean_name = name.replace(" - ", "_").replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"{clean_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
        
    # Baseline 3: Train PyTorch Deep Learning Model
    if HAS_TORCH:
        print("\nTraining Baseline 3 - PyTorch MLP...")
        pytorch_model = train_pytorch_model(X_train, y_train)
        
        # Predict and evaluate PyTorch model
        print("Evaluating PyTorch MLP...")
        y_pred_pt, _ = predict_pytorch_model(pytorch_model, X_test)
        res_pt = evaluate_model("Baseline 3 - PyTorch MLP", y_test, y_pred_pt)
        results.append(res_pt)
        plot_confusion_matrix("Baseline 3 - PyTorch MLP", y_test, y_pred_pt)
        
        # Save PyTorch model & export ONNX for ESP32-S3
        pt_model_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.pt")
        torch.save(pytorch_model.state_dict(), pt_model_path)
        print(f"Saved PyTorch MLP to {pt_model_path}")
        
        try:
            onnx_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.onnx")
            export_pytorch_to_onnx(pytorch_model, (1, X_train.shape[1]), onnx_path)
        except Exception as e:
            print(f"ONNX export warning: {e}")
    else:
        print("\nPyTorch not yet installed - Skipping Baseline 3 (PyTorch MLP).")
    
    # Baseline 4: Same-Board / Oracle Baseline (Intra-Unit split benchmark)
    print("\nEvaluating Baseline 4 - Same-Board Oracle (Random Forest on Train Set)...")
    rf_oracle = get_rf_model()
    from sklearn.model_selection import train_test_split
    X_tr_or, X_te_or, y_tr_or, y_te_or = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_train
    )
    rf_oracle.fit(X_tr_or, y_tr_or)
    y_pred_oracle = rf_oracle.predict(X_te_or)
    res_oracle = evaluate_model("Baseline 4 - Same-Board Oracle", y_te_or, y_pred_oracle)
    results.append(res_oracle)
    plot_confusion_matrix("Baseline 4 - Same-Board Oracle", y_te_or, y_pred_oracle)
    
    # 3. Summary
    print("\n--- Step 3: Evaluation Summary (Macro F1 & Accuracy) ---")
    print_summary_table(results)
    
    print("Pipeline completed successfully! Check the 'saved_models' and 'plots' directories.")

if __name__ == "__main__":
    main()

