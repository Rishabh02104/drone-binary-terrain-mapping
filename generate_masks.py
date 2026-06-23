import os
import cv2
import torch
import numpy as np
from surface_map import SimpleCNN, process_image_patch_cnn

# =====================================
# CONFIGURATION
# =====================================
MODEL_PATH = "models/cnn_binary_model.pth"
IMAGES_DIR = "segmentation_dataset/Images"
MASKS_DIR = "segmentation_dataset/masks"

PATCH_SIZE = 56
STRIDE = 16
ROAD_THRESHOLD = 0.65

def bootstrap_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Train it first using train_cnn.py.")
        return

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Create destination dir
    os.makedirs(MASKS_DIR, exist_ok=True)

    # Get images
    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.dng'))])
    print(f"Found {len(images)} images to process.")

    for i, img_name in enumerate(images):
        base_name, _ = os.path.splitext(img_name)
        mask_path = os.path.join(MASKS_DIR, f"{base_name}.png")
        
        # Skip if already exists
        if os.path.exists(mask_path):
            print(f"[{i+1}/{len(images)}] Mask already exists for {img_name}. Skipping.")
            continue

        print(f"[{i+1}/{len(images)}] Generating baseline mask for {img_name}...")
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error: Could not load image {img_name}")
            continue

        # Process image using patch-based model
        # We only need the road_mask output
        road_mask, _, _, _, _, _ = process_image_patch_cnn(
            image, model, device,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            road_threshold=ROAD_THRESHOLD,
            use_calibration=False # no calibration needed for raw mask
        )

        # Save mask as PNG
        cv2.imwrite(mask_path, road_mask)
        print(f"Saved mask to {mask_path}")

    print("\n✅ Bootstrapping complete! All images now have a baseline mask in 'segmentation_dataset/masks'.")

if __name__ == "__main__":
    bootstrap_masks()
