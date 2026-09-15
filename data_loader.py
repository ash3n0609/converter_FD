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

def extract_window_features(df, window_size=config.WINDOW_SIZE):
    """
    Extracts time-series window statistics (mean, std, RMS, peak-to-peak)
    for each simulation run file.
    """
    feature_rows = []
    
    groupby_col = config.RUN_COL if config.RUN_COL in df.columns else None
    
    if groupby_col:
        groups = df.groupby(groupby_col)
    else:
        groups = [('all', df)]
        
    for run_id, group_df in groups:
        v = group_df['Load Voltage'].values
        i = group_df['Load Current'].values
        labels = group_df['fault_label'].values if 'fault_label' in group_df.columns else np.zeros(len(group_df), dtype=int)
        unit_ids = group_df['unit_id'].values if 'unit_id' in group_df.columns else np.repeat('unit1', len(group_df))
        
        n_samples = len(group_df)
        for start_idx in range(0, n_samples - window_size + 1, window_size):
            end_idx = start_idx + window_size
            
            v_win = v[start_idx:end_idx]
            i_win = i[start_idx:end_idx]
            label_win = labels[start_idx]
            unit_win = unit_ids[start_idx]
            
            v_mean = np.mean(v_win)
            v_std = np.std(v_win)
            v_rms = np.sqrt(np.mean(v_win**2))
            v_p2p = np.ptp(v_win)
            
            i_mean = np.mean(i_win)
            i_std = np.std(i_win)
            i_rms = np.sqrt(np.mean(i_win**2))
            i_p2p = np.ptp(i_win)
            
            feature_rows.append({
                'V_mean': v_mean, 'V_std': v_std, 'V_rms': v_rms, 'V_p2p': v_p2p,
                'I_mean': i_mean, 'I_std': i_std, 'I_rms': i_rms, 'I_p2p': i_p2p,
                'fault_label': label_win,
                'unit_id': unit_win,
                'run_file': run_id
            })
            
    return pd.DataFrame(feature_rows)

def load_and_preprocess_data():
    """
    Loads data, extracts windowed features, performs Group Split by simulation unit/run
    to eliminate temporal data leakage, and normalizes features.
    """
    if config.USE_SYNTHETIC_DATA:
        print("Generating synthetic dataset...")
        df = generate_synthetic_data()
    else:
        print(f"Loading real data from {config.DATA_PATH}...")
        if not os.path.exists(config.DATA_PATH):
            raise FileNotFoundError(f"Data file {config.DATA_PATH} not found.")
        df = pd.read_csv(config.DATA_PATH)
        
    print(f"Raw dataframe shape: {df.shape}")
    
    # Check if raw features exist or if already windowed
    if 'Load Voltage' in df.columns and 'Load Current' in df.columns:
        print(f"Extracting window features (Window Size = {config.WINDOW_SIZE})...")
        features_df = extract_window_features(df, window_size=config.WINDOW_SIZE)
    else:
        features_df = df
        
    print(f"Extracted feature dataset shape: {features_df.shape}")
    
    # Handle missing values
    features_df.fillna(features_df.mean(numeric_only=True), inplace=True)
    
    # Group-based split to prevent leakage across simulation runs
    if config.USE_GROUP_SPLIT and 'unit_id' in features_df.columns:
        test_unit = config.TEST_UNIT
        print(f"Performing Cross-Unit Group Split: Train on units != '{test_unit}', Test on unit '{test_unit}'...")
        train_df = features_df[features_df['unit_id'] != test_unit].copy()
        test_df = features_df[features_df['unit_id'] == test_unit].copy()
        
        # If test_df is empty fallback to train_test_split
        if len(test_df) == 0:
            print("Warning: Reserved test unit not found in dataset. Falling back to stratified split.")
            X_all = features_df[config.FEATURE_COLS].values
            y_all = features_df[config.LABEL_COL].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_all
            )
        else:
            X_train = train_df[config.FEATURE_COLS].values
            y_train = train_df[config.LABEL_COL].values
            X_test = test_df[config.FEATURE_COLS].values
            y_test = test_df[config.LABEL_COL].values
    else:
        X_all = features_df[config.FEATURE_COLS].values
        y_all = features_df[config.LABEL_COL].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y_all
        )
        
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Normalization / Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    scaler_path = os.path.join(config.MODEL_SAVE_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

