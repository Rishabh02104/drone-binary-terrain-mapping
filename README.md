# Drone-Based Binary Terrain Mapping with Road Measurement

## Overview

This project performs patch-based terrain classification on aerial drone images using a custom Convolutional Neural Network (CNN).

The system classifies terrain into:
- Road
- Non-Road

In addition to classification, the system computes:

- Road Length (in meters)
- Road Width (in meters)
- Road Area (in square meters)
- Road Coverage Percentage
- Binary Mask Visualization
- Annotated Output Overlay

Pixel-based measurements are converted into real-world metric estimations using drone altitude (80 meters) and camera field-of-view (84°) based geometric projection.

---

## Model Details

- Custom CNN architecture
- Batch Normalization
- Dropout regularization
- Confidence thresholding
- Patch size experimentation (32, 56, 112 tested)
- 56 chosen as optimal

---

## Real-World Measurement Conversion

To convert pixel measurements into real-world units, drone altitude (80m) and camera field-of-view approximation (84°) were used.

Ground Width (meters) is computed as:

Ground Width = 2 × Height × tan(FOV / 2)

Meters per Pixel = Ground Width / Image Width

Using this geometric approximation, pixel-based length, width, and area values are converted into:

- Road Length (meters)
- Road Width (meters)
- Road Area (m²)

This enhancement upgrades the system from pixel estimation to real-world infrastructure measurement.

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
