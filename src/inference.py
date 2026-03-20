import json
import joblib
import numpy as np
from ultralytics import YOLO

from src.config import YOLO_MODEL_PATH, CLASSIFIER_PATH, CLASS_NAMES_PATH, IMAGE_SIZE
from src.feature_engineering import build_feature_row_from_xywh


def load_yolo_model():
    return YOLO(str(YOLO_MODEL_PATH), task="detect")


def load_classifier():
    model = joblib.load(CLASSIFIER_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def extract_features_from_yolo_result(result, image_size: int = IMAGE_SIZE):
    data = {
        "bottle_x": 0, "bottle_y": 0, "bottle_w": 0, "bottle_h": 0,
        "label_x": 0, "label_y": 0, "label_w": 0, "label_h": 0,
        "cap_x": 0, "cap_y": 0, "cap_w": 0, "cap_h": 0,
    }

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy

        class_idx = int(box.cls.item())
        label = result.names[class_idx]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        if label == "butelka":
            data["bottle_x"] = cx
            data["bottle_y"] = cy
            data["bottle_w"] = w
            data["bottle_h"] = h
        elif label == "etykieta":
            data["label_x"] = cx
            data["label_y"] = cy
            data["label_w"] = w
            data["label_h"] = h
        elif label == "korek":
            data["cap_x"] = cx
            data["cap_y"] = cy
            data["cap_w"] = w
            data["cap_h"] = h

    features = build_feature_row_from_xywh(
        bottle_x=data["bottle_x"],
        bottle_y=data["bottle_y"],
        bottle_w=data["bottle_w"],
        bottle_h=data["bottle_h"],
        label_x=data["label_x"],
        label_y=data["label_y"],
        label_w=data["label_w"],
        label_h=data["label_h"],
        cap_x=data["cap_x"],
        cap_y=data["cap_y"],
        cap_w=data["cap_w"],
        cap_h=data["cap_h"],
        image_size=image_size,
    )

    return np.array(list(features.values())).reshape(1, -1)