from pathlib import Path
from ultralytics import YOLO
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.onnx"
INPUT_DIR = BASE_DIR / "data" / "sample_images"
OUTPUT_DIR = BASE_DIR / "assets"

OUTPUT_DIR.mkdir(exist_ok=True)

print("MODEL_PATH:", MODEL_PATH)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

model = YOLO(str(MODEL_PATH), task="detect")

for image_path in INPUT_DIR.glob("*"):
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    results = model.predict(str(image_path), save=False, verbose=False, conf=0.25)
    plotted = results[0].plot()

    output_path = OUTPUT_DIR / f"demo_{image_path.stem}.png"
    Image.fromarray(plotted).save(output_path)

    print(f"Saved: {output_path}")