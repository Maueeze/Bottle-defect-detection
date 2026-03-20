import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# =========================
# Konfiguracja
# =========================
MODEL_PATH = "best.onnx"
WEIGHTS_PATH = "wagi.csv"
IMAGE_SIZE = 1500

CLASS_NAMES = [
    "Bez wad",
    "Bez korka",
    "Bez etykiety",
    "Zła butelka",
    "Złe miejsce etykiety"
]

BOX_COLORS = {
    "butelka": "red",
    "etykieta": "green",
    "korek": "blue",
}


# =========================
# Ładowanie zasobów
# =========================
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


@st.cache_resource
def load_perceptron_weights(weights_path: str):
    return np.loadtxt(weights_path, delimiter=",")


# =========================
# Funkcje pomocnicze
# =========================
def resize_image(input_path: str, output_path: str, size=(IMAGE_SIZE, IMAGE_SIZE)):
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path)


def sign_function(x: float) -> int:
    return 1 if x >= 0 else 0


def perceptron(weights: np.ndarray, inputs: np.ndarray) -> int:
    score = np.dot(weights, inputs)
    return sign_function(score)


def build_feature_vector(result, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Buduje 13-elementowy wektor cech:
    [flag_butelka,
     butelka_x, butelka_y, butelka_w, butelka_h,
     etykieta_x, etykieta_y, etykieta_w, etykieta_h,
     korek_x, korek_y, korek_w, korek_h]
    """
    features = np.zeros(13, dtype=float)

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy

        class_index = int(box.cls.item())
        label = result.names[class_index]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        if label == "butelka":
            features[0] = 1
            features[1] = center_x / image_size
            features[2] = center_y / image_size
            features[3] = width / image_size
            features[4] = height / image_size

        elif label == "etykieta":
            features[5] = center_x / image_size
            features[6] = center_y / image_size
            features[7] = width / image_size
            features[8] = height / image_size

        elif label == "korek":
            features[9] = center_x / image_size
            features[10] = center_y / image_size
            features[11] = width / image_size
            features[12] = height / image_size

    return features


def predict_multiclass(features: np.ndarray, weights: np.ndarray):
    """
    Zwraca:
    - binarne wyniki one-vs-all dla każdej klasy
    - finalną klasę jako argmax po score'ach
    """
    raw_scores = weights @ features
    binary_outputs = [sign_function(score) for score in raw_scores]
    predicted_index = int(np.argmax(raw_scores))
    predicted_label = CLASS_NAMES[predicted_index]

    pred_df = pd.DataFrame([binary_outputs], columns=CLASS_NAMES)
    score_df = pd.DataFrame([raw_scores], columns=CLASS_NAMES)

    return predicted_label, predicted_index, pred_df, score_df


def draw_boxes_from_features(image_path: str, output_path: str, features: np.ndarray, image_size: int = IMAGE_SIZE):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            font = ImageFont.load_default()

        objects = [
            ("butelka", features[1], features[2], features[3], features[4]),
            ("etykieta", features[5], features[6], features[7], features[8]),
            ("korek", features[9], features[10], features[11], features[12]),
        ]

        for label, x, y, w, h in objects:
            if x * y * w * h == 0:
                continue

            cx = x * image_size
            cy = y * image_size
            bw = w * image_size
            bh = h * image_size

            top_left = (cx - bw / 2, cy - bh / 2)
            bottom_right = (cx + bw / 2, cy + bh / 2)

            color = BOX_COLORS.get(label, "yellow")
            draw.rectangle([top_left, bottom_right], outline=color, width=4)
            draw.text((top_left[0] + 10, top_left[1] + 10), label.capitalize(), fill=color, font=font)

        img.save(output_path)


# =========================
# Aplikacja Streamlit
# =========================
def main():
    st.set_page_config(page_title="Klasyfikacja wad butelek", layout="wide")

    st.title("WIDZENIE MASZYNOWE")
    st.subheader("Klasyfikacja wad butelek na linii produkcyjnej")
    st.markdown("---")

    # Ładowanie modelu ONNX i wag perceptronu
    model = load_model(MODEL_PATH)
    weights = load_perceptron_weights(WEIGHTS_PATH)

    uploaded_file = st.file_uploader("Wgraj obraz", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_path = tmpdir / uploaded_file.name
            resized_path = tmpdir / "resized.jpg"
            output_path = tmpdir / "prediction.jpg"

            # zapis pliku
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # resize
            resize_image(str(input_path), str(resized_path))

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Obraz wejściowy")
                st.image(str(resized_path), use_container_width=True)

            # predykcja YOLO ONNX
            results = model.predict(
                source=str(resized_path),
                save=False,
                verbose=False,
                conf=0.25
            )
            result = results[0]

            # budowa wektora cech
            features = build_feature_vector(result)

            # klasyfikacja defektu
            final_label, final_idx, pred_df, score_df = predict_multiclass(features, weights)

            # rysowanie boxów
            draw_boxes_from_features(str(resized_path), str(output_path), features)

            with col2:
                st.markdown("### Detekcja obiektów")
                st.image(str(output_path), use_container_width=True)

            st.markdown("---")
            st.markdown("### Wynik klasyfikacji")
            st.success(f"Predykowana klasa: **{final_label}**")

            st.markdown("### Wektor cech")
            feature_names = [
                "butelka_exists",
                "butelka_x", "butelka_y", "butelka_w", "butelka_h",
                "etykieta_x", "etykieta_y", "etykieta_w", "etykieta_h",
                "korek_x", "korek_y", "korek_w", "korek_h"
            ]
            features_df = pd.DataFrame([features], columns=feature_names)
            st.dataframe(features_df, use_container_width=True)

            st.markdown("### Wyjścia perceptronów one-vs-all")
            st.dataframe(pred_df, use_container_width=True)

            st.markdown("### Surowe score'y")
            st.dataframe(score_df, use_container_width=True)


if __name__ == "__main__":
    main()
