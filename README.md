# Drone-Based Binary Terrain Mapping with Road Measurement

## Overview

This project performs patch-based terrain classification on aerial drone images using a custom CNN model.

The system classifies terrain into:

- Road
- Non-Road

It additionally computes:

- Road Length (in pixels)
- Road Width (in pixels)
- Road Area
- Road Coverage Percentage
- Visual annotated output

---

## Model Details

- Custom CNN architecture
- Batch Normalization
- Dropout regularization
- Confidence thresholding
- Patch size experimentation (32, 56, 112 tested)
- 56 chosen as optimal

---

## Technical Highlights

- Patch-wise inference strategy
- Softmax confidence filtering
- Contour detection using OpenCV
- Rotated bounding box estimation (minAreaRect)
- Road coverage percentage calculation
- Hyperparameter tuning (patch size optimization)

## Project Structure

## Sample Output

The system generates:

- Classification Map
- Binary Road Mask
- Final Annotated Output with Measurements
