import os
import pandas as pd

# Configuration
RAW_DATA_DIR = 'dataset_v5'
OUTPUT_DIR = 'data'
OUTPUT_FILE = 'simulink_export.csv'

LABEL_MAP = {
    'healthy': 0,
    'esr_mild': 1,
    'esr_severe': 2,
    'switch_deg': 3,
    'gate_deg': 4
}

def extract_label_from_filename(filename):
    for label_str in LABEL_MAP.keys():
        if label_str in filename:
            return LABEL_MAP[label_str]
    return None

def build_dataset():
    print("=== Building Dataset from Raw Files (Row-by-Row) ===")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dfs = []
    
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith('.csv'):
            continue
            
        if 'sensor_fault' in filename:
            print(f"Skipping {filename} (sensor_fault runs excluded).")
            continue
            
        label = extract_label_from_filename(filename)
        if label is None:
            print(f"Warning: Could not determine label for {filename}, skipping.")
            continue
            
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df = pd.read_csv(filepath)
        
        # Expect columns: "Time / s", "Load Current", "Load Voltage"
        if "Load Current" not in df.columns or "Load Voltage" not in df.columns:
            print(f"Warning: {filename} missing required columns. Skipping.")
            continue
            
        # We only keep the raw values as features
        df = df[['Load Voltage', 'Load Current']].copy()
        
        # Assign the fault label to every row in this file
        df['fault_label'] = label
        
        dfs.append(df)
        print(f"Loaded {filename} -> Label: {label}, Rows: {len(df)}")
        
    print("\nConcatenating all files...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"Saving to {out_path}...")
    final_df.to_csv(out_path, index=False)
    print(f"Dataset successfully built and saved to {out_path}")
    print(f"Total samples (rows): {len(final_df)}")

if __name__ == "__main__":
    build_dataset()
