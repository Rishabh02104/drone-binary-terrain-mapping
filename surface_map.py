import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import os

# ======================================
# SETTINGS
# ======================================

MODEL_PATH = "models/cnn_binary_model.pth"
UNET_MODEL_PATH = "models/unet_road_model.pth"
MULTICLASS_MODEL_PATH = "models/cnn_multiclass_model.pth"
IMAGE_PATH = "images/test_image.jpg"

PATCH_SIZE = 56
STRIDE = 16
ROAD_THRESHOLD = 0.65

DRONE_HEIGHT = 80
FOV_DEG = 84

USE_CALIBRATION = True
KNOWN_ROAD_WIDTH = 25  # adjust (25–30)

# ======================================
# MODEL ARCHITECTURES
# ======================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ======================================
# CENTERLINE PATH TRACING
# ======================================

def trace_skeleton(skeleton):
    """
    Traces the skeleton pixels sequentially from one endpoint to another.
    This converts a set of pixels into an ordered line path.
    """
    points = list(zip(*np.where(skeleton)))
    if not points:
        return []
    
    points_set = set(points)
    start_pt = points[0]
    
    # Find an endpoint (point with only 1 neighbor in 8-neighborhood)
    for pt in points:
        y, x = pt
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in points_set:
                    neighbors += 1
        if neighbors == 1:
            start_pt = pt
            break
            
    path = []
    curr = start_pt
    if curr in points_set:
        points_set.remove(curr)
    path.append(curr)
    
    while len(points_set) > 0:
        y, x = curr
        next_pt = None
        # Look for 8-connected neighbors first
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in points_set:
                    next_pt = neighbor
                    break
            if next_pt:
                break
                
        if next_pt:
            curr = next_pt
            points_set.remove(curr)
            path.append(curr)
        else:
            # Handle small gaps in skeleton: find nearest remaining point
            nearest = None
            min_dist = float('inf')
            for pt in points_set:
                dist = np.hypot(pt[0]-y, pt[1]-x)
                if dist < min_dist:
                    min_dist = dist
                    nearest = pt
            if nearest and min_dist < 20:  # jump gap up to 20 pixels
                curr = nearest
                points_set.remove(curr)
                path.append(curr)
            else:
                break
                
    return path

# ======================================
# SHARED GEOMETRIC & VISUAL ANALYSIS
# ======================================

def extract_metrics_and_visuals(image, road_mask,
                                drone_height=80, fov_deg=84,
                                use_calibration=True, known_road_width=25):
    """
    Given a binary road mask, computes physical metrics (length, dynamic widths, area)
    and generates visual overlays (skeleton centerline, bounding box, combined view).
    """
    h, w, _ = image.shape
    
    # 1. Clean mask morphologically
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Skeletonization & Distance Transform
    binary_road = (road_mask > 0)
    skeleton = np.zeros_like(binary_road)
    local_widths_px = np.array([])
    length_px = 0.0
    
    if np.any(binary_road):
        skeleton = skeletonize(binary_road)
        distance_map = distance_transform_edt(binary_road)
        
        # Trace skeleton sequentially
        ordered_pts = trace_skeleton(skeleton)
        
        # Compute curved length by summing Euclidean distances between sequential centerline points
        if len(ordered_pts) > 1:
            for i in range(len(ordered_pts) - 1):
                y1, x1 = ordered_pts[i]
                y2, x2 = ordered_pts[i+1]
                length_px += np.hypot(x2 - x1, y2 - y1)
        else:
            length_px = float(np.sum(skeleton))
            
        # Compute dynamic widths along the centerline
        # EXCLUDE endpoints (outer 15% on each side) to avoid end-boundary artifacts (width tapering to 0)
        if len(ordered_pts) > 10:
            margin = int(len(ordered_pts) * 0.15)
            middle_pts = ordered_pts[margin:-margin]
            local_widths_px = np.array([2 * distance_map[y, x] for y, x in middle_pts])
            local_widths_px = local_widths_px[local_widths_px > 0]
        else:
            skeleton_y, skeleton_x = np.where(skeleton)
            if len(skeleton_y) > 0:
                local_widths_px = 2 * distance_map[skeleton]
                local_widths_px = local_widths_px[local_widths_px > 0]

    # Calculate scale factor (meters per pixel)
    fov_rad = np.radians(fov_deg)
    ground_width_m = 2 * drone_height * np.tan(fov_rad / 2)
    meters_per_pixel = ground_width_m / w
    
    if len(local_widths_px) > 0:
        # Use median width for calibration to be robust against local outliers/spurs
        median_width_px = np.median(local_widths_px)
        if use_calibration:
            meters_per_pixel = known_road_width / median_width_px
            
        real_length = length_px * meters_per_pixel
        real_widths = local_widths_px * meters_per_pixel
        avg_width_m = np.mean(real_widths)
        min_width_m = np.min(real_widths)
        max_width_m = np.max(real_widths)
    else:
        real_length = 0.0
        avg_width_m = 0.0
        min_width_m = 0.0
        max_width_m = 0.0

    # Total area is computed from Length * Avg Width to filter out blocky patch artifacts
    real_area = real_length * avg_width_m
    coverage_percent = (np.sum(binary_road) / (h * w)) * 100

    metrics = {
        "length_m": real_length,
        "avg_width_m": avg_width_m,
        "min_width_m": min_width_m,
        "max_width_m": max_width_m,
        "area_m2": real_area,
        "coverage_percent": coverage_percent,
        "meters_per_pixel": meters_per_pixel
    }

    # 3. Visualizations
    # Colored classification map
    classification_map = np.zeros_like(image)
    classification_map[road_mask == 255] = (0, 0, 255)   # Road → Red
    classification_map[road_mask == 0] = (0, 255, 0)     # Non-road → Green
    classification_blend = cv2.addWeighted(image, 0.6, classification_map, 0.4, 0)

    # Annotated Output
    output = image.copy()
    output[skeleton] = (255, 0, 0)  # Draw centerline in bright blue
    
    # Draw minAreaRect bounding box (bright yellow)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output, [box], 0, (0, 255, 255), 2)

    # Blend overlay mask
    colored_mask = np.zeros_like(image)
    colored_mask[road_mask == 255] = (0, 0, 255)
    blended = cv2.addWeighted(output, 0.7, colored_mask, 0.3, 0)

    # Draw Text Panel
    cv2.rectangle(blended, (10, 10), (460, 180), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blended, "DRONE TERRAIN ANALYSIS (SKELETON)", (20, 35), font, 0.6, (0, 255, 255), 2)
    cv2.putText(blended, f"Length: {round(real_length, 2)} m", (20, 65), font, 0.55, (255, 255, 255), 2)
    cv2.putText(blended, f"Avg Width: {round(avg_width_m, 2)} m (min: {round(min_width_m, 2)}, max: {round(max_width_m, 2)})", (20, 90), font, 0.5, (255, 255, 255), 2)
    cv2.putText(blended, f"Area: {round(real_area, 2)} m^2", (20, 115), font, 0.55, (255, 255, 255), 2)
    cv2.putText(blended, f"Coverage: {round(coverage_percent, 1)}%", (20, 140), font, 0.55, (255, 255, 255), 2)

    # Combined views
    top_row = np.hstack((image, classification_blend))
    bottom_row = np.hstack((cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR), blended))
    combined_view = np.vstack((top_row, bottom_row))

    return road_mask, metrics, skeleton, classification_blend, blended, combined_view

# ======================================
# METHOD 1: PATCH-BASED CNN INFERENCE (BINARY)
# ======================================

def process_image_patch_cnn(image, model, device,
                            patch_size=56, stride=16, road_threshold=0.65,
                            drone_height=80, fov_deg=84,
                            use_calibration=True, known_road_width=25):
    """
    Runs patch-based CNN inference on the image using batching.
    """
    h, w, _ = image.shape
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    patches_coords = []
    patches_tensors = []

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            input_tensor = transform(patch)
            patches_coords.append((x, y))
            patches_tensors.append(input_tensor)

    road_mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(patches_tensors) > 0:
        batch_size = 128
        for i in range(0, len(patches_tensors), batch_size):
            batch_tensors = torch.stack(patches_tensors[i:i+batch_size]).to(device)
            with torch.no_grad():
                outputs = model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
            
            for j, (x, y) in enumerate(patches_coords[i:i+batch_size]):
                if probs[j][1].item() > road_threshold:
                    road_mask[y:y+patch_size, x:x+patch_size] = 255

    return extract_metrics_and_visuals(
        image, road_mask, drone_height, fov_deg, use_calibration, known_road_width
    )

# ======================================
# METHOD 2: U-NET SEMANTIC SEGMENTATION INFERENCE
# ======================================

def process_image_unet(image, model, device,
                       drone_height=80, fov_deg=84,
                       use_calibration=True, known_road_width=25):
    """
    Runs pixel-level U-Net segmentation on the image.
    """
    h, w, _ = image.shape
    
    # 1. Resize and normalize
    img_resized = cv2.resize(image, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 2. Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # 3. Postprocess mask back to original resolution
    road_mask_resized = (prob > 0.5).astype(np.uint8) * 255
    road_mask = cv2.resize(road_mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return extract_metrics_and_visuals(
        image, road_mask, drone_height, fov_deg, use_calibration, known_road_width
    )

# ======================================
# METHOD 3: PATCH-BASED CNN INFERENCE (MULTICLASS)
# ======================================

def process_image_multiclass(image, model, device, class_mapping,
                            patch_size=56, stride=16):
    """
    Runs patch-based multiclass classification on the image to generate
    a rich, color-coded terrain coverage map.
    """
    h, w, _ = image.shape
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    patches_coords = []
    patches_tensors = []

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            input_tensor = transform(patch)
            patches_coords.append((x, y))
            patches_tensors.append(input_tensor)

    multiclass_map = np.zeros_like(image)
    
    # Reverse key-values for lookup
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Color palette for 9 terrain categories: (B, G, R)
    class_colors = {
        "Buildings": (128, 128, 128),           # Gray
        "Cemented road": (220, 220, 220),       # Light Gray
        "Coloured paver path": (180, 105, 255),  # Violet/Pink
        "Grass Ground": (50, 220, 50),          # Bright Green
        "Non-Road": (30, 80, 150),              # Brown
        "Road": (0, 0, 255),                    # Red
        "Sand path": (100, 220, 220),           # Sand/Yellow
        "Vegetations": (0, 100, 0),             # Dark Green
        "solar panels": (255, 0, 0)             # Blue
    }

    if len(patches_tensors) > 0:
        batch_size = 128
        for i in range(0, len(patches_tensors), batch_size):
            batch_tensors = torch.stack(patches_tensors[i:i+batch_size]).to(device)
            with torch.no_grad():
                outputs = model(batch_tensors)
                preds = torch.argmax(outputs, dim=1)
            
            for j, (x, y) in enumerate(patches_coords[i:i+batch_size]):
                pred_idx = preds[j].item()
                class_name = idx_to_class.get(pred_idx, "Non-Road")
                color = class_colors.get(class_name, (255, 255, 255))
                multiclass_map[y:y+patch_size, x:x+patch_size] = color

    # Clean map boundary slightly using opening/closing
    kernel = np.ones((3, 3), np.uint8)
    multiclass_map = cv2.morphologyEx(multiclass_map, cv2.MORPH_OPEN, kernel)

    # Blend with original image
    blended = cv2.addWeighted(image, 0.5, multiclass_map, 0.5, 0)
    return multiclass_map, blended

# ======================================
# MAIN EXECUTION (STANDALONE RUN)
# ======================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train_cnn.py first.")
        exit()

    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}.")
        exit()

    image = cv2.imread(IMAGE_PATH)
    
    # Process
    print("Processing image with batch inference and skeleton measurements...")
    import time
    start_time = time.time()
    road_mask, metrics, skeleton, classification_blend, blended, combined_view = process_image_patch_cnn(
        image, model, device,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        road_threshold=ROAD_THRESHOLD,
        drone_height=DRONE_HEIGHT,
        fov_deg=FOV_DEG,
        use_calibration=USE_CALIBRATION,
        known_road_width=KNOWN_ROAD_WIDTH
    )
    print(f"Processed in {time.time() - start_time:.2f} seconds.")

    # Print Results
    print("\n==== FINAL RESULTS (SKELETON MEASUREMENT) ====")
    print("Road Length (m):", round(metrics["length_m"], 2))
    print("Avg Road Width (m):", round(metrics["avg_width_m"], 2))
    print("Min Road Width (m):", round(metrics["min_width_m"], 2))
    print("Max Road Width (m):", round(metrics["max_width_m"], 2))
    print("Road Area (m²):", round(metrics["area_m2"], 2))
    print("Coverage (%):", round(metrics["coverage_percent"], 2))

    # Save Outputs
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/road_mask.jpg", road_mask)
    cv2.imwrite("outputs/classification_map.jpg", classification_blend)
    cv2.imwrite("outputs/final_output.jpg", blended)
    cv2.imwrite("outputs/combined_view.jpg", combined_view)
    print("Outputs saved successfully in 'outputs/' folder.")

    # Show Windows
    import sys
    if "--no-show" not in sys.argv:
        cv2.imshow("1. Original Image", image)
        cv2.imshow("2. Road Mask (Binary)", road_mask)
        cv2.imshow("3. Road vs Non-Road", classification_blend)
        cv2.imshow("4. Final Output (Skeleton Overlay)", blended)
        cv2.imshow("5. Combined View (Presentation)", combined_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Headless mode: skipped displaying OpenCV windows.")