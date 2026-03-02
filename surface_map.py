import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# ======================================
# SETTINGS
# ======================================

MODEL_PATH = "models/cnn_binary_model.pth"
IMAGE_PATH = "images/test_image.jpg"

PATCH_SIZE = 56
ROAD_THRESHOLD = 0.65

DRONE_HEIGHT = 80  # meters
FOV_DEG = 84       # typical DJI camera FOV (adjust if known)

classes = ["Non-Road", "Road"]

colors = {
    "Road": (0, 0, 255),
    "Non-Road": (0, 255, 0)
}

# ======================================
# CNN MODEL (same as training)
# ======================================

class SimpleCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ======================================
# LOAD MODEL
# ======================================

model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======================================
# TRANSFORM
# ======================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================================
# LOAD IMAGE
# ======================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load image.")
    exit()

h, w, _ = image.shape

# ======================================
# PIXEL TO METER CONVERSION
# ======================================

fov_rad = np.radians(FOV_DEG)
ground_width_m = 2 * DRONE_HEIGHT * np.tan(fov_rad / 2)
meters_per_pixel = ground_width_m / w

# ======================================
# PATCH PROCESSING
# ======================================

output_map = np.zeros_like(image)
road_mask = np.zeros((h, w), dtype=np.uint8)

for y in range(0, h, PATCH_SIZE):
    for x in range(0, w, PATCH_SIZE):

        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            continue

        input_tensor = transform(patch).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)

        road_prob = probs[0][1].item()

        if road_prob > ROAD_THRESHOLD:
            class_name = "Road"
            road_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255
        else:
            class_name = "Non-Road"

        output_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = colors[class_name]

# ======================================
# BLEND RESULT
# ======================================

blended = cv2.addWeighted(image, 0.6, output_map, 0.4, 0)

# ======================================
# ROAD MEASUREMENT
# ======================================

contours, _ = cv2.findContours(
    road_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    print("No road detected.")
else:
    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    cv2.drawContours(blended, [box], 0, (255, 0, 0), 2)

    width_pixels = min(rect[1])
    length_pixels = max(rect[1])
    road_area_pixels = cv2.contourArea(largest_contour)

    # Convert to real-world units
    real_length = length_pixels * meters_per_pixel
    real_width = width_pixels * meters_per_pixel
    real_area = road_area_pixels * (meters_per_pixel ** 2)

    coverage_percent = (road_area_pixels / (h * w)) * 100

    print("Road Length (meters):", round(real_length, 2))
    print("Road Width (meters):", round(real_width, 2))
    print("Road Area (m²):", round(real_area, 2))
    print("Road Coverage (%):", round(coverage_percent, 2))

    # ======================================
    # OVERLAY METRICS
    # ======================================

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(blended, (10, 10), (420, 180), (0, 0, 0), -1)

    cv2.putText(blended, "DRONE TERRAIN ANALYSIS",
                (20, 35), font, 0.7, (0, 255, 255), 2)

    cv2.putText(blended, f"Length: {round(real_length,2)} m",
                (20, 70), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Width: {round(real_width,2)} m",
                (20, 95), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Area: {round(real_area,2)} m^2",
                (20, 120), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Coverage: {round(coverage_percent,1)}%",
                (20, 145), font, 0.6, (255,255,255), 2)

# ======================================
# SAVE OUTPUTS
# ======================================

cv2.imwrite("outputs/classification_map.jpg", output_map)
cv2.imwrite("outputs/road_mask.jpg", road_mask)
cv2.imwrite("outputs/surface_output.jpg", blended)

cv2.imshow("Final Output", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()