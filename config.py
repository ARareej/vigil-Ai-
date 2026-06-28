import os

# Data paths
DATA_PATH = "data/vigil_ai_final_dataset.csv"
BALANCED_DATA_PATH = "data/training_balanced_clean.csv"
PROCESSED_DATA_PATH = "data/processed/vigil_ai_final_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SOURCE_COLUMN = "source"

# Models
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2"
}

# Directories
OUTPUT_MODELS_DIR = "outputs/models"
OUTPUT_REPORTS_DIR = "outputs/reports"
OUTPUT_LOGS_DIR = "outputs/logs"

# Training parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Percentage of total data
RANDOM_STATE = 42

# Class weights
CLASS_WEIGHT_MODE = "clip" # Options: "none", "sqrt", "clip"
CLASS_WEIGHT_MIN = 0.5
CLASS_WEIGHT_MAX = 10.0

# Oversampling
OVERSAMPLE_MIN_COUNT = 1500

# Focal Loss
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.5

# Evaluation
PRIMARY_METRIC = "f1_macro"
METRIC_FOR_BEST_MODEL = "eval_f1_macro"

# Ensure directories exist
for directory in [OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_LOGS_DIR, "data/raw", "data/processed"]:
    os.makedirs(directory, exist_ok=True)
