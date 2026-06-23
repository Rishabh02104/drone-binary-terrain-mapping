# 🛸 Drone-Based Terrain Mapping & Infrastructure Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

An advanced drone-based aerial imaging pipeline designed to segment, classify, and mathematically analyze road networks and land coverage from high-resolution UAV imagery. Features dual architectures (**Patch-CNN** for land coverage and **U-Net** for semantic segmentation), dynamic metric estimation (curved skeleton lengths and local width profiles), a live **Streamlit** dashboard with interactive manual mask annotators, and geoprojected **GIS / GeoJSON** exporters.

---

## 🚀 Key Features

*   **Dual Segmentation Engines**:
    *   **U-Net Semantic Segmentation**: Dense pixel-level classification with combined Dice-BCE loss for smooth, continuous road boundaries.
    *   **Patch-based CNN**: Sliding window classifier optimized for multi-class terrain mapping.
*   **Dynamic Geometric Analytics**:
    *   **Curved Length Tracing**: Centerline extraction via topological skeletonization (`scikit-image`).
    *   **Perpendicular Width Profiles**: Continuous local width calculations using Euclidean Distance Transform (EDT) along the skeleton.
*   **Interactive Web Dashboard**:
    *   Parameters sidebar (altitude, camera field of view, thresholding).
    *   **Manual Annotation Canvas**: Freehand paint correction tool that saves labels directly to the training set.
    *   Interactive class distribution graphs and visual layers toggler.
*   **GIS / GeoJSON Exporter**:
    *   Converts pixel centerlines and bounding polygons into real-world geographic coordinates using drone metadata (GPS center, flight heading).
*   **Multi-Class Land Coverage**:
    *   9-category mapping: *Buildings, Cemented Road, Paver Path, Grass Ground, Non-Road, Road, Sand Path, Vegetations, Solar Panels*.

---

## 📐 Mathematical Formulation

### 1. Geometric Spatial Projection
Ground distance representation per pixel is calculated using the drone's altitude and the camera's field of view (FOV):

$$\text{Ground Footprint Width } (W_{\text{ground}}) = 2 \times H \times \tan\left(\frac{\text{FOV}_{\text{diagonal}}}{2}\right) \times \cos(\theta_{\text{aspect}})$$

$$\text{Meters Per Pixel Scale } (S) = \frac{W_{\text{ground}}}{W_{\text{image}}}$$

Where:
*   $H$ is the drone's flight altitude above ground level (e.g., $80\text{ m}$).
*   $\text{FOV}$ is the diagonal camera Field of View (e.g., $84^\circ$).
*   $W_{\text{image}}$ is the image width in pixels.

### 2. Centerline Width EDT Measurement
For every coordinate $(x, y)$ along the road's skeleton (centerline):

$$\text{Local Road Width } (w_i) = 2 \times \text{EDT}(x, y) \times S$$

Where $\text{EDT}(x, y)$ is the Euclidean Distance Transform calculating the minimum distance from $(x, y)$ to the nearest non-road boundary pixel. To eliminate boundary noise, the minimum and maximum widths exclude the endpoints (outer $15\%$).

### 3. Dice-BCE Optimization Loss
The U-Net semantic segmentation network is trained by minimizing the sum of Binary Cross-Entropy and Dice Loss to handle class imbalance:

$$\mathcal{L}_{\text{Dice-BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right] + \left( 1 - \frac{2 \sum_i p_i y_i + 1}{\sum_i p_i + \sum_i y_i + 1} \right)$$

---

## 📂 Project Structure

```bash
├── app.py                      # Streamlit interactive web dashboard
├── surface_map.py              # Core geometry, skeletonization & model inference logic
├── train_unet.py               # U-Net architecture, custom dataset loader & training loop
├── train_cnn.py                # Multi-class and Binary Patch-CNN training pipeline
├── generate_masks.py           # Mask bootstrapping utility for DJI image datasets
├── evaluate_test_image.py      # CLI comparative evaluation script
├── segmentation_dataset/       # U-Net dataset directory
│   ├── Images/                 # High-resolution drone training images
│   └── masks/                  # Corresponding binary segmentation masks
├── images/                     # Evaluation images directory
│   └── test_image.jpg          # Test image for metric calibration
├── models/                     # Saved network weights (.pth files)
└── requirements.txt            # Python dependencies list
```

---

## 💻 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Rishabh02104/drone-binary-terrain-mapping.git
    cd drone-binary-terrain-mapping
    ```

2.  **Install Dependencies**:
    Make sure you have Python 3.11+ installed. Install the pinned dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Running the Pipeline

### 1. Launch the Live Streamlit Dashboard
The web app includes the live manual annotator canvas, GeoJSON export, and real-time metric visualizer:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Run Comparative Evaluation (CLI)
Compare results between Patch-CNN and U-Net models on the test image:
```bash
python evaluate_test_image.py
```

### 3. Bootstrapping Baseline Masks
If you add raw images to `segmentation_dataset/Images/` and want to pre-generate baseline binary masks using the Patch-CNN classifier:
```bash
python generate_masks.py
```

---

## 📊 Quantitative Metrics Example
*Measured on test crop at 80m altitude:*

| Model Metric | Patch-based CNN | U-Net Segmentation |
| :--- | :---: | :---: |
| **Centerline Length** | $26.46\text{ m}$ | **$66.75\text{ m}$** *(Continuous path)* |
| **Average Road Width** | $25.12\text{ m}$ | **$23.57\text{ m}$** *(Smooth edges)* |
| **Total Road Area** | $664.63\text{ m}^2$ | **$1573.59\text{ m}^2$** |
| **Visual Quality** | Blocky/Noisy | **Pixel-Perfect Contour** |
