try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class DummyNNModule:
        pass
    nn = type('nn', (), {'Module': DummyNNModule})

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
import config

# --- Baseline Models ---

def get_trivial_model():
    """
    Baseline 1: Trivial Baseline (Majority Class Classifier)
    """
    return DummyClassifier(strategy='most_frequent')

def get_rf_model():
    """
    Baseline 2a: Classical ML - Random Forest
    """
    return RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

def get_xgb_model():
    """
    Baseline 2b: Classical ML - XGBoost
    """
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

def get_svm_model():
    """
    Baseline 2c: Classical ML - Support Vector Machine (RBF Kernel)
    """
    return SVC(
        kernel='rbf',
        C=1.0,
        probability=True, 
        random_state=config.RANDOM_STATE
    )

def get_qda_model():
    """
    Baseline 2d: Quadratic Discriminant Analysis (Top model for Healthy class detection)
    """
    return QuadraticDiscriminantAnalysis(reg_param=0.1)

def get_quadratic_svm_model():
    """
    Baseline 2e: Support Vector Machine (Quadratic/Polynomial Kernel - Top model for Sensor Fault)
    """
    return SVC(
        kernel='poly',
        degree=2,
        C=1.0,
        probability=True,
        random_state=config.RANDOM_STATE
    )


# --- PyTorch Deep Learning Model ---

class FaultPredictionMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for tabular data classification.
    """
    def __init__(self, input_dim, num_classes):
        super(FaultPredictionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the PyTorch MLP model with GPU acceleration (if available) and batching.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training PyTorch MLP on device: {device}...")
    
    input_dim = X_train.shape[1]
    model = FaultPredictionMLP(input_dim, config.NUM_CLASSES).to(device)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    model.train()
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.EPOCHS:
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")
            
    return model.to('cpu')

def predict_pytorch_model(model, X_test):
    """
    Predicts using the trained PyTorch model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu()
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu()
        
    return predicted.numpy(), probabilities.numpy()


# --- Hybrid Models ---

def get_stacking_hybrid_model():
    """
    Hybrid Model 1: Stacking Ensemble combining QDA, Quadratic SVM, Random Forest, 
    and XGBoost with a Logistic Regression Meta-Learner.
    Leverages QDA for 'Healthy' recall + Quadratic SVM for 'Sensor Fault' recall + RF/XGB for component faults.
    """
    base_estimators = [
        ('qda', QuadraticDiscriminantAnalysis(reg_param=0.1)),
        ('svm_quad', SVC(kernel='poly', degree=2, C=1.0, probability=True, random_state=config.RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=config.RANDOM_STATE, n_jobs=-1))
    ]
    
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_STATE)
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    return stacking_model


class HierarchicalCascadeHybrid(BaseEstimator, ClassifierMixin):
    """
    Hybrid Model 2: Two-Stage Hierarchical Cascade Classifier.
    Stage 1: Broad Classifier separating Component Degradation (ESR/Switch) vs Signal Anomalies (Healthy/Sensor Fault/Gate Deg).
    Stage 2A: Random Forest / XGBoost specialized on Component Degradation.
    Stage 2B: QDA + Quadratic SVM specialized on Healthy vs Sensor Fault.
    """
    def __init__(self):
        self.stage1_model = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
        self.stage2_component_model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
        self.stage2_signal_model = SVC(kernel='poly', degree=2, C=1.0, probability=True, random_state=config.RANDOM_STATE)
        
    def fit(self, X, y):
        component_mask = np.isin(y, [1, 2, 3])
        y_stage1 = component_mask.astype(int)
        
        self.stage1_model.fit(X, y_stage1)
        
        if np.any(component_mask):
            self.stage2_component_model.fit(X[component_mask], y[component_mask])
        if np.any(~component_mask):
            self.stage2_signal_model.fit(X[~component_mask], y[~component_mask])
            
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        stage1_preds = self.stage1_model.predict(X)
        y_pred = np.zeros(len(X), dtype=int)
        
        comp_indices = np.where(stage1_preds == 1)[0]
        sig_indices = np.where(stage1_preds == 0)[0]
        
        if len(comp_indices) > 0:
            y_pred[comp_indices] = self.stage2_component_model.predict(X[comp_indices])
        if len(sig_indices) > 0:
            y_pred[sig_indices] = self.stage2_signal_model.predict(X[sig_indices])
            
        return y_pred


def export_pytorch_to_onnx(model, dummy_input_shape=(1, 8), save_path='saved_models/model.onnx'):
    """
    Exports PyTorch model to ONNX format for TFLite / ESP32-S3 deployment.
    """
    model.eval()
    dummy_input = torch.randn(*dummy_input_shape)
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported PyTorch model to ONNX: {save_path}")

