import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

def generate_synthetic_data(num_samples=1000):
    """
    Generates dummy time-series/tabular data matching the expected schema.
    This allows testing the pipeline before the real Simulink data arrives.
    """
    np.random.seed(config.RANDOM_STATE)
    
    # Generate baseline normal values
    vin = np.random.normal(loc=12.0, scale=0.5, size=num_samples)
    vout = np.random.normal(loc=24.0, scale=0.5, size=num_samples)
    il = np.random.normal(loc=2.0, scale=0.2, size=num_samples)
    duty_cycle = np.random.normal(loc=0.5, scale=0.05, size=num_samples)
    switch_temp = np.random.normal(loc=45.0, scale=5.0, size=num_samples)
    
    # Assign labels randomly (0 to NUM_CLASSES-1)
    labels = np.random.randint(0, config.NUM_CLASSES, size=num_samples)
    
    # Perturb features slightly based on labels to give models something to learn
    # This is just for demonstration so the models don't just guess randomly
    for i in range(num_samples):
        if labels[i] == 1: # Switch open (low current, high temp)
            il[i] *= 0.1
            switch_temp[i] += 20
        elif labels[i] == 2: # Switch short (high current)
            il[i] *= 3.0
            vout[i] *= 0.5
        elif labels[i] == 3: # Diode failure
            vout[i] *= 0.8
        elif labels[i] == 4: # Capacitor degradation
            vout[i] += np.random.normal(0, 3.0) # High ripple/variance
        elif labels[i] == 5: # Inductor fault
            il[i] += np.random.normal(0, 1.0) # High current ripple
            
    df = pd.DataFrame({
        'Vin': vin,
        'Vout': vout,
        'IL': il,
        'duty_cycle': duty_cycle,
        'switch_temp': switch_temp,
        'fault_label': labels
    })
    
    return df

def load_and_preprocess_data():
    """
    Loads data, handles missing values, normalizes features, and splits into train/test.
    """
    if config.USE_SYNTHETIC_DATA:
        print("Generating synthetic dataset...")
        df = generate_synthetic_data()
    else:
        print(f"Loading real data from {config.DATA_PATH}...")
        if not os.path.exists(config.DATA_PATH):
            raise FileNotFoundError(f"Data file {config.DATA_PATH} not found.")
        df = pd.read_csv(config.DATA_PATH)
        
    # Handle missing values (simple forward fill or mean fill)
    df.fillna(df.mean(), inplace=True)
    
    X = df[config.FEATURE_COLS].values
    y = df[config.LABEL_COL].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Normalization / Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test
