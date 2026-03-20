# Bottle Defect Detection

Hybrid computer vision project for detecting bottle defects on a simulated production line.

This system combines:
- **YOLO object detection** for locating bottle components
- **feature engineering** based on bounding box geometry
- **machine learning classification** for final defect recognition
- **Streamlit deployment** for interactive testing

---

## Project Overview

The goal of this project was to build an end-to-end defect detection pipeline for bottles in a production-like environment.

The system first detects:
- bottle
- label
- cap

Then it converts detected bounding boxes into a **13-feature vector** and uses a machine learning classifier to assign one of the final defect classes:

- `no_defect`
- `missing_cap`
- `missing_label`
- `damaged_bottle`
- `misplaced_label`

---

## Pipeline

![Architecture](assets/architecture.png)

**Workflow:**

1. Input image
2. YOLO detects bottle components
3. Bounding boxes are converted into geometric features
4. A machine learning model predicts the final defect class
5. The result is displayed in a Streamlit web app

---

## Example Application View

![App Screenshot](assets/screenshot_1.png)

---

## Example Predictions

### Prediction Example 1
![Demo 1](assets/demo_detection/demo_prediction_1.png)

### Prediction Example 2
![Demo 2](assets/demo_detection/demo_prediction_2.png)

### Prediction Example 3
![Demo 3](assets/demo_detection/demo_prediction_3.png)

---

## Dataset

The project used two data sources:

### 1. Object detection dataset
- approximately **3000 augmented images**
- used for YOLO training
- annotated for:
  - bottle
  - label
  - cap

### 2. Defect classification dataset
- **5 CSV files**, one per class
- around **100 samples per class**
- each row contains bounding box positions for bottle, label, and cap

The CSV structure is:

- bottle_x
- bottle_y
- bottle_w
- bottle_h
- label_x
- label_y
- label_w
- label_h
- cap_x
- cap_y
- cap_w
- cap_h

---

## Feature Engineering

Each sample is transformed into a **13-feature vector**:

- `bottle_exists`
- `bottle_x`, `bottle_y`, `bottle_w`, `bottle_h`
- `label_x`, `label_y`, `label_w`, `label_h`
- `cap_x`, `cap_y`, `cap_w`, `cap_h`

These features are normalized and used as input to the classification model.

---

## Model Comparison

The following models were compared:

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- MLP

The best-performing model was:

**Random Forest**

### Cross-validation comparison
![Model Comparison](assets/model_comparison.png)

### Confusion matrix of best classifier
![Classifier Confusion Matrix](assets/confusion_matrix.png)

---

## YOLO Training Artifacts

The repository also includes files from the YOLO training process in:

```text
training/yolo_training_run/
