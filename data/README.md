# Data

This repository contains the tabular feature pipeline used for defect classification.

## Available files

### Raw class-wise CSV files
These files contain object box positions for:
- bottle
- label
- cap

Each row contains 12 values in the following order:

1. bottle_x
2. bottle_y
3. bottle_w
4. bottle_h
5. label_x
6. label_y
7. label_w
8. label_h
9. cap_x
10. cap_y
11. cap_w
12. cap_h

### Processed files
- `features.csv`
- `train.csv`
- `val.csv`
- `test.csv`

## Sample images
The `sample_images/` folder contains example bottle images that can be used to test the Streamlit application.

## Dataset note
The original project used approximately:
- 3000 augmented images for YOLO object detection
- 5 class-specific CSV files for the defect classification stage
