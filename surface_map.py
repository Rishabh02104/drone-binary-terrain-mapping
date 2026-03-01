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

classes = ["Non-Road", "Road"]

colors = {
    "Road": (0, 0, 255),
    "Non-Road": (0, 255, 0)
}

# ======================================
# CNN MODEL (Must Match Training)
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
h, w, _ = image.shape

output_map = np.zeros_like(image)
road_mask = np.zeros((h, w), dtype=np.uint8)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load image. Check IMAGE_PATH.")
    exit()

# ======================================
# PATCH PROCESSING
# ======================================

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

        color = colors[class_name]
        output_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = color

# ======================================
# BLEND RESULT
# ======================================

blended = cv2.addWeighted(image, 0.6, output_map, 0.4, 0)

# ======================================
# MEASUREMENT
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
    road_area = cv2.contourArea(largest_contour)

    total_pixels = h * w
    coverage_percent = (road_area / total_pixels) * 100

    print("Road Length (pixels):", round(length_pixels, 2))
    print("Road Width (pixels):", round(width_pixels, 2))
    print("Road Area (pixels):", round(road_area, 2))
    print("Road Coverage (%):", round(coverage_percent, 2))

    # ======================================
    # OVERLAY TEXT
    # ======================================

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(blended, (10, 10), (380, 160), (0, 0, 0), -1)

    cv2.putText(blended, "DRONE-BASED BINARY TERRAIN",
                (20, 35), font, 0.7, (0, 255, 255), 2)

    cv2.putText(blended, f"Length: {round(length_pixels,1)} px",
                (20, 70), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Width: {round(width_pixels,1)} px",
                (20, 95), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Area: {int(road_area)} px^2",
                (20, 120), font, 0.6, (255,255,255), 2)

    cv2.putText(blended, f"Coverage: {round(coverage_percent,1)}%",
                (20, 145), font, 0.6, (255,255,255), 2)

# ======================================
# LEGEND
# ======================================

legend_y = h - 80

cv2.rectangle(blended, (20, legend_y), (45, legend_y + 25), (0,0,255), -1)
cv2.putText(blended, "Road",
            (55, legend_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

cv2.rectangle(blended, (20, legend_y + 35), (45, legend_y + 60), (0,255,0), -1)
cv2.putText(blended, "Non-Road",
            (55, legend_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ======================================
# SAVE & DISPLAY
# ======================================

cv2.imwrite("surface_output.jpg", blended)
cv2.imwrite("classification_map.jpg", output_map)
cv2.imwrite("road_mask.jpg", road_mask)

print("All output images saved successfully.")

cv2.imshow("Original Image", image)
cv2.imshow("Classification Map", output_map)
cv2.imshow("Road Mask", road_mask)
cv2.imshow("Final Output", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()