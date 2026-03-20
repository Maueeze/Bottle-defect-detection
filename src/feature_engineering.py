import pandas as pd


FEATURE_COLUMNS = [
    "bottle_exists",
    "bottle_x", "bottle_y", "bottle_w", "bottle_h",
    "label_x", "label_y", "label_w", "label_h",
    "cap_x", "cap_y", "cap_w", "cap_h",
]


def normalize_xywh(x, y, w, h, image_size: int):
    return x / image_size, y / image_size, w / image_size, h / image_size


def build_feature_row_from_xywh(
    bottle_x=0, bottle_y=0, bottle_w=0, bottle_h=0,
    label_x=0, label_y=0, label_w=0, label_h=0,
    cap_x=0, cap_y=0, cap_w=0, cap_h=0,
    image_size: int = 1500,
):
    bottle_exists = 1 if bottle_w > 0 and bottle_h > 0 else 0

    bx, by, bw, bh = normalize_xywh(bottle_x, bottle_y, bottle_w, bottle_h, image_size)
    lx, ly, lw, lh = normalize_xywh(label_x, label_y, label_w, label_h, image_size)
    cx, cy, cw, ch = normalize_xywh(cap_x, cap_y, cap_w, cap_h, image_size)

    return {
        "bottle_exists": bottle_exists,
        "bottle_x": bx,
        "bottle_y": by,
        "bottle_w": bw,
        "bottle_h": bh,
        "label_x": lx,
        "label_y": ly,
        "label_w": lw,
        "label_h": lh,
        "cap_x": cx,
        "cap_y": cy,
        "cap_w": cw,
        "cap_h": ch,
    }


def ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS + ["target"]]