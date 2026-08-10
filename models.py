import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import config

# --- Scikit-Learn / XGBoost Models ---

def get_rf_model():
    return RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

def get_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

def get_svm_model():
    return SVC(
        probability=True, 
        random_state=config.RANDOM_STATE
    )


# --- PyTorch Deep Learning Model ---

class FaultPredictionMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for tabular data classification.
    """
    def __init__(self, input_dim, num_classes):
        super(FaultPredictionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(X_train, y_train, X_val=None, y_val=None):
    """
    Trains the PyTorch MLP model.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    input_dim = X_train.shape[1]
    model = FaultPredictionMLP(input_dim, config.NUM_CLASSES)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print("Training PyTorch MLP...")
    model.train()
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")
            
    return model

def predict_pytorch_model(model, X_test):
    """
    Predicts using the trained PyTorch model.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        outputs = model(X_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        
    return predicted.numpy(), probabilities.numpy()
