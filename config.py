"""Central configuration. Override the dataset path with the GTSRB_DATASET_DIR environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("GTSRB_DATASET_DIR", r"C:\Users\Aster\Downloads\AI_project"))

IMG_SIZE = 32
NUM_CLASSES = 43
SEED = 42
VAL_SPLIT = 0.2

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_FILTERS = (32, 64, 128)
DEFAULT_DENSE = 256
DEFAULT_DROPOUT = 0.4

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "traffic_sign_cnn.keras"
CLASS_MAP_PATH = MODELS_DIR / "class_mapping.json"
PREPROC_PATH = MODELS_DIR / "preprocessing.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"
DATASET_REPORT_PATH = OUTPUTS_DIR / "dataset_report.json"
TRAIN_STATUS_PATH = OUTPUTS_DIR / "training_status.json"
TRAIN_LOG_PATH = OUTPUTS_DIR / "training.log"
PREDICTION_HISTORY_PATH = OUTPUTS_DIR / "prediction_history.csv"

CONFIDENCE_THRESHOLD = 0.70
MAX_UPLOAD_MB = 10

# Standard GTSRB class names (class id -> name).
CLASS_NAMES = {
    0: "Speed limit (20 km/h)", 1: "Speed limit (30 km/h)", 2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)", 4: "Speed limit (70 km/h)", 5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)", 7: "Speed limit (100 km/h)", 8: "Speed limit (120 km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 t",
    11: "Right-of-way at next intersection", 12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "Vehicles over 3.5 t prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve to the left", 20: "Dangerous curve to the right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left", 38: "Keep right",
    39: "Keep left", 40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 t",
}

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
