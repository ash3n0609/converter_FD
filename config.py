import os

# --- Data Configuration ---
USE_SYNTHETIC_DATA = False  # Set to False when real data is ready
DATA_PATH = 'data/simulink_export.csv'  # Path to real data CSV

# Feature, Windowing, and Label definitions
RAW_FEATURE_COLS = ['Load Voltage', 'Load Current']
WINDOW_SIZE = 50
TRANSIENT_SKIP_RATIO = 0.08  # Ignore initial cold-start transient samples per run
FEATURE_COLS = [
    'V_mean', 'V_std', 'V_rms', 'V_p2p',
    'I_mean', 'I_std', 'I_rms', 'I_p2p',
    'V_ripple_ratio', 'I_ripple_ratio', 'R_apparent', 'V_crest_factor',
    'P_ratio'
]
LABEL_COL = 'fault_label'
GROUP_COL = 'unit_id'
RUN_COL = 'run_file'
USE_GROUP_SPLIT = True
TEST_UNIT = 'unit3'  # Unit 3 reserved as unseen test set for cross-unit generalisation
STACKING_MODEL_NAME = "Hybrid_1_Stacking_Ensemble"

# The 5 classes of faults (Healthy + 4 Component Degradation states)
CLASSES = {
    0: "Healthy",
    1: "ESR Mild",
    2: "ESR Severe",
    3: "Switch Deg",
    4: "Gate Deg"
}
NUM_CLASSES = len(CLASSES)

# --- Training Configuration ---
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Validation size from the remaining training data for PyTorch
RANDOM_STATE = 42

# --- PyTorch Configuration ---
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.002

# --- Output Configuration ---
MODEL_SAVE_DIR = 'saved_models'
PLOT_SAVE_DIR = 'plots'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

