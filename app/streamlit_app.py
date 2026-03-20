import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import tempfile
import streamlit as st
from PIL import Image

from src.inference import (
    load_yolo_model,
    load_classifier,
    extract_features_from_yolo_result,
)


DISPLAY_NAMES = {
    "no_defect": "Bez wad",
    "missing_cap": "Bez korka",
    "missing_label": "Bez etykiety",
    "damaged_bottle": "Uszkodzona butelka",
    "misplaced_label": "Złe położenie etykiety",
}


@st.cache_resource
def get_yolo_model():
    return load_yolo_model()


@st.cache_resource
def get_classifier():
    return load_classifier()


def main():
    st.set_page_config(
        page_title="Bottle Defect Detection",
        layout="centered"
    )

    st.title("Bottle Defect Detection")
    st.subheader("Detekcja obiektów YOLO + klasyfikacja wady produktu")
    st.markdown("---")

    model = get_yolo_model()
    classifier, class_names = get_classifier()

    uploaded_file = st.file_uploader(
        "Wgraj obraz butelki",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / uploaded_file.name

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            image = Image.open(input_path).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Obraz wejściowy")
                st.image(image, width=1500)

            results = model.predict(
                source=str(input_path),
                save=False,
                verbose=False,
                conf=0.25
            )
            result = results[0]

            features = extract_features_from_yolo_result(result)
            pred_idx = classifier.predict(features)[0]
            pred_label = class_names[pred_idx]
            display_label = DISPLAY_NAMES.get(pred_label, pred_label)

            with col2:
                st.markdown("### Detekcja YOLO")
                plotted = result.plot()
                st.image(plotted, width=1500)

            st.markdown("---")
            st.markdown("### Wynik klasyfikacji")
            st.success(f"Predykowana klasa: **{display_label}**")

            st.markdown("### Cechy wejściowe")
            st.write({
                "bottle_exists": float(features[0][0]),
                "bottle_x": float(features[0][1]),
                "bottle_y": float(features[0][2]),
                "bottle_w": float(features[0][3]),
                "bottle_h": float(features[0][4]),
                "label_x": float(features[0][5]),
                "label_y": float(features[0][6]),
                "label_w": float(features[0][7]),
                "label_h": float(features[0][8]),
                "cap_x": float(features[0][9]),
                "cap_y": float(features[0][10]),
                "cap_w": float(features[0][11]),
                "cap_h": float(features[0][12]),
            })


if __name__ == "__main__":
    main()