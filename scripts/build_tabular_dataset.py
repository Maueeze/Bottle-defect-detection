from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    FEATURES_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    IMAGE_SIZE,
    CLASS_NAMES,
)
from src.feature_engineering import (
    build_feature_row_from_xywh,
    ensure_feature_order,
)

# Stała kolejność kolumn w plikach CSV bez nagłówka
CSV_COLUMNS = [
    "bottle_x", "bottle_y", "bottle_w", "bottle_h",
    "label_x", "label_y", "label_w", "label_h",
    "cap_x", "cap_y", "cap_w", "cap_h",
]


def build_dataset_from_class_file(file_path: Path, target_name: str) -> pd.DataFrame:
    # header=None -> pierwszy wiersz to dane, nie nagłówki
    df = pd.read_csv(file_path, header=None)

    # Nadaj nazwy kolumn po ustalonej kolejności
    if df.shape[1] != 12:
        raise ValueError(
            f"Plik {file_path} ma {df.shape[1]} kolumn, a oczekiwano 12."
        )

    df.columns = CSV_COLUMNS

    rows = []

    for _, row in df.iterrows():
        features = build_feature_row_from_xywh(
            bottle_x=row["bottle_x"],
            bottle_y=row["bottle_y"],
            bottle_w=row["bottle_w"],
            bottle_h=row["bottle_h"],

            label_x=row["label_x"],
            label_y=row["label_y"],
            label_w=row["label_w"],
            label_h=row["label_h"],

            cap_x=row["cap_x"],
            cap_y=row["cap_y"],
            cap_w=row["cap_w"],
            cap_h=row["cap_h"],

            image_size=IMAGE_SIZE,
        )

        features["target"] = target_name
        rows.append(features)

    return pd.DataFrame(rows)


def main():
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for class_name in CLASS_NAMES:
        file_path = RAW_DATA_DIR / f"{class_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Brakuje pliku: {file_path}")

        class_df = build_dataset_from_class_file(file_path, class_name)
        print(f"{class_name}: {class_df.shape}")
        all_dfs.append(class_df)

    features_df = pd.concat(all_dfs, ignore_index=True)
    features_df = ensure_feature_order(features_df)

    print("\nPreview of processed features:")
    print(features_df.head(10).to_string())

    features_df.to_csv(FEATURES_PATH, index=False)

    train_df, temp_df = train_test_split(
        features_df,
        test_size=0.30,
        stratify=features_df["target"],
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["target"],
        random_state=42,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nSaved files:")
    print(f"- {FEATURES_PATH}")
    print(f"- {TRAIN_PATH}")
    print(f"- {VAL_PATH}")
    print(f"- {TEST_PATH}")

    print("\nShapes:")
    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

    print("\nClass distribution:")
    print(features_df["target"].value_counts())


if __name__ == "__main__":
    main()