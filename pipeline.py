import os
import joblib
import torch
import config
from data_loader import load_and_preprocess_data
from models import get_rf_model, get_xgb_model, get_svm_model, train_pytorch_model, predict_pytorch_model
from evaluate import evaluate_model, plot_confusion_matrix, print_summary_table, plot_feature_importance

def main():
    print("=== Starting ML Pipeline for Fault Prediction ===")
    
    # 1. Load and preprocess data
    print("\n--- Step 1: Data Loading & Preprocessing ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # 2. Define Models
    print("\n--- Step 2: Training Models ---")
    models = {
        "Random Forest": get_rf_model(),
        "XGBoost": get_xgb_model()
        # "SVM": get_svm_model() # Disabled because SVM on 2.5 million rows will take too long
    }
    
    results = []
    
    # Train Scikit-Learn / XGBoost models
    for name, model in models.items():
        print(f"Training {name}...")
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
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")
        
    # Train PyTorch Deep Learning Model
    pytorch_model = train_pytorch_model(X_train, y_train)
    
    # Predict and evaluate PyTorch model
    print("Evaluating PyTorch MLP...")
    y_pred_pt, _ = predict_pytorch_model(pytorch_model, X_test)
    res_pt = evaluate_model("PyTorch MLP", y_test, y_pred_pt)
    results.append(res_pt)
    plot_confusion_matrix("PyTorch MLP", y_test, y_pred_pt)
    
    # Save PyTorch model
    pt_model_path = os.path.join(config.MODEL_SAVE_DIR, "PyTorch_MLP.pt")
    torch.save(pytorch_model.state_dict(), pt_model_path)
    print(f"Saved PyTorch MLP to {pt_model_path}")
    
    # 3. Summary
    print("\n--- Step 3: Evaluation Summary ---")
    print_summary_table(results)
    
    print("Pipeline completed successfully! Check the 'saved_models' and 'plots' directories.")

if __name__ == "__main__":
    main()
