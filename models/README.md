# Models

This folder contains lightweight artifacts used by the application:

- `defect_classifier.joblib` — trained machine learning classifier for final defect classification
- `class_names.json` — mapping of class indices to labels

## YOLO model

The object detection model `best.onnx` is not included in this repository due to file size limitations.

To run the app, place the exported YOLO model here:

```text
models/best.onnx
