from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

YOLO_MODEL_PATH = MODELS_DIR / "best.onnx"
CLASSIFIER_PATH = MODELS_DIR / "defect_classifier.joblib"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMAGE_SIZE = 1500

CLASS_NAMES = [
    "no_defect",
    "missing_cap",
    "missing_label",
    "damaged_bottle",
    "misplaced_label",
]