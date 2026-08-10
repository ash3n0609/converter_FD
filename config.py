import os

# --- Data Configuration ---
USE_SYNTHETIC_DATA = False  # Set to False when real data is ready
DATA_PATH = 'data/simulink_export.csv'  # Path to real data CSV

# Feature and label definitions
FEATURE_COLS = ['Load Voltage', 'Load Current']
LABEL_COL = 'fault_label'

# The classes of faults. 0 is typically "healthy"
# Update these based on your actual Simulink fault labels
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
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Output Configuration ---
MODEL_SAVE_DIR = 'saved_models'
PLOT_SAVE_DIR = 'plots'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
