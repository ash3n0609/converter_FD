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
    'gate_deg': 4,
    'sensor_fault': 5
}

def extract_label_from_filename(filename):
    for label_str, label_val in LABEL_MAP.items():
        if label_str in filename:
            return label_val
    return None

def extract_unit_from_filename(filename):
    # e.g., 'unit1_esr_mild_r2.5.csv' -> 'unit1'
    if filename.startswith('unit'):
        return filename.split('_')[0]
    return 'unit1'

def build_dataset():
    print("=== Building Dataset from Raw Files (Row-by-Row with Metadata) ===")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dfs = []
    
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith('.csv'):
            continue
            
        label = extract_label_from_filename(filename)
        if label is None:
            print(f"Warning: Could not determine label for {filename}, skipping.")
            continue
            
        unit_id = extract_unit_from_filename(filename)
        run_file = filename.replace('.csv', '')
            
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df = pd.read_csv(filepath)
        
        # Expect columns: "Time / s", "Load Current", "Load Voltage"
        if "Load Current" not in df.columns or "Load Voltage" not in df.columns:
            print(f"Warning: {filename} missing required columns. Skipping.")
            continue
            
        # Keep features + metadata
        df = df[['Load Voltage', 'Load Current']].copy()
        df['unit_id'] = unit_id
        df['run_file'] = run_file
        df['fault_label'] = label
        
        dfs.append(df)
        print(f"Loaded {filename} -> Unit: {unit_id}, Label: {label}, Rows: {len(df)}")
        
    print("\nConcatenating all files...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"Saving to {out_path}...")
    final_df.to_csv(out_path, index=False)
    print(f"Dataset successfully built and saved to {out_path}")
    print(f"Total samples (rows): {len(final_df)}")

if __name__ == "__main__":
    build_dataset()
