import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

def generate_synthetic_data(num_samples=2500):
    """
    Generates dummy time-series data matching the raw Simulink schema.
    """
    np.random.seed(config.RANDOM_STATE)
    
    # 6 classes matching config.CLASSES
    v = np.random.normal(loc=5.0, scale=0.05, size=num_samples)
    i = np.random.normal(loc=1.0, scale=0.02, size=num_samples)
    labels = np.random.randint(0, config.NUM_CLASSES, size=num_samples)
    
    for idx in range(num_samples):
        lbl = labels[idx]
        if lbl == 1: # ESR Mild
            v[idx] += np.random.normal(0, 0.15)
        elif lbl == 2: # ESR Severe
            v[idx] += np.random.normal(0, 0.6)
        elif lbl == 3: # Switch Deg
            v[idx] -= 0.4
        elif lbl == 4: # Gate Deg
            v[idx] += np.random.normal(0, 0.25)
            i[idx] += np.random.normal(0, 0.1)
                
    return pd.DataFrame({
        'Load Voltage': v,
        'Load Current': i,
        'unit_id': 'unit1',
        'run_file': 'synthetic_run',
        'fault_label': labels
    })

def extract_window_features(df, window_size=config.WINDOW_SIZE, skip_transient=True):
    """
    Extracts time-series window statistics (mean, std, RMS, peak-to-peak,
    ripple ratio, apparent resistance, crest factor) for each simulation run file.
    Filters out cold-start startup transients to eliminate label noise.
    """
    dfs = []
    groupby_col = config.RUN_COL if config.RUN_COL in df.columns else None
    
    if groupby_col:
        groups = df.groupby(groupby_col)
    else:
        groups = [('all', df)]
        
    for run_id, group_df in groups:
        n_samples = len(group_df)
        
        # Discard initial power-up cold-start transient if file is long enough
        if skip_transient and n_samples > 500:
            start_idx = int(n_samples * getattr(config, 'TRANSIENT_SKIP_RATIO', 0.08))
            sub_df = group_df.iloc[start_idx:]
        else:
            sub_df = group_df
            
        v = sub_df['Load Voltage'].values
        i = sub_df['Load Current'].values
        labels = sub_df['fault_label'].values if 'fault_label' in sub_df.columns else np.zeros(len(sub_df), dtype=int)
        unit_ids = sub_df['unit_id'].values if 'unit_id' in sub_df.columns else np.repeat('unit1', len(sub_df))
        
        n_valid = len(v)
        n_windows = n_valid // window_size
        if n_windows == 0:
            continue
            
        v_sub = v[:n_windows * window_size].reshape(n_windows, window_size)
        i_sub = i[:n_windows * window_size].reshape(n_windows, window_size)
        
        v_mean = np.mean(v_sub, axis=1)
        v_std = np.std(v_sub, axis=1)
        v_rms = np.sqrt(np.mean(v_sub**2, axis=1))
        v_p2p = np.ptp(v_sub, axis=1)
        
        i_mean = np.mean(i_sub, axis=1)
        i_std = np.std(i_sub, axis=1)
        i_rms = np.sqrt(np.mean(i_sub**2, axis=1))
        i_p2p = np.ptp(i_sub, axis=1)
        
        eps = 1e-6
        v_ripple_ratio = v_p2p / (np.abs(v_mean) + eps)
        i_ripple_ratio = i_p2p / (np.abs(i_mean) + eps)
        r_apparent = v_mean / (np.abs(i_mean) + eps)
        v_crest_factor = v_p2p / (2.0 * np.sqrt(2.0) * v_rms + eps)
        p_ratio = (v_mean * i_mean) / (v_rms * i_rms + eps)
        
        lbl_win = labels[::window_size][:n_windows]
        unit_win = unit_ids[::window_size][:n_windows]
        
        chunk = pd.DataFrame({
            'V_mean': v_mean, 'V_std': v_std, 'V_rms': v_rms, 'V_p2p': v_p2p,
            'I_mean': i_mean, 'I_std': i_std, 'I_rms': i_rms, 'I_p2p': i_p2p,
            'V_ripple_ratio': v_ripple_ratio,
            'I_ripple_ratio': i_ripple_ratio,
            'R_apparent': r_apparent,
            'V_crest_factor': v_crest_factor,
            'P_ratio': p_ratio,
            'fault_label': lbl_win,
            'unit_id': unit_win,
            'run_file': run_id
        })
        dfs.append(chunk)
        
    if len(dfs) == 0:
        return pd.DataFrame(columns=config.FEATURE_COLS)
    return pd.concat(dfs, ignore_index=True)

def load_and_preprocess_data():
    """
    Loads data, extracts windowed features, performs Group Split by simulation unit/run
    to eliminate temporal data leakage, and normalizes features.
    """
    cached_window_file = 'data/window_features.csv'
    features_df = None
    if os.path.exists(cached_window_file):
        print(f"Loading cached window features from {cached_window_file}...")
        cached_df = pd.read_csv(cached_window_file)
        if all(col in cached_df.columns for col in config.FEATURE_COLS):
            features_df = cached_df
        else:
            print("Cached window features are missing updated feature columns. Recomputing...")

    if features_df is None:
        if config.USE_SYNTHETIC_DATA:
            print("Generating synthetic dataset...")
            df = generate_synthetic_data()
        else:
            print(f"Loading real data from {config.DATA_PATH}...")
            if not os.path.exists(config.DATA_PATH):
                raise FileNotFoundError(f"Data file {config.DATA_PATH} not found.")
            df = pd.read_csv(config.DATA_PATH)
            
        print(f"Raw dataframe shape: {df.shape}")
        if 'Load Voltage' in df.columns and 'Load Current' in df.columns:
            print(f"Extracting window features (Window Size = {config.WINDOW_SIZE})...")
            features_df = extract_window_features(df, window_size=config.WINDOW_SIZE)
        else:
            features_df = df
            
        print(f"Saving extracted window features to {cached_window_file}...")
        os.makedirs('data', exist_ok=True)
        features_df.to_csv(cached_window_file, index=False)
        
    # Exclude any sensor_fault runs (duplicate healthy files removed from taxonomy)
    if 'fault_label' in features_df.columns:
        features_df = features_df[features_df['fault_label'].isin(config.CLASSES.keys())].copy()
    if 'run_file' in features_df.columns:
        features_df = features_df[~features_df['run_file'].astype(str).str.contains('sensor_fault', case=False, na=False)].copy()

    print(f"Extracted feature dataset shape (after excluding sensor fault duplicates): {features_df.shape}")
    
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

