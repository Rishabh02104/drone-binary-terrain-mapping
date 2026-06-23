import os
import cv2
import torch
import json
import numpy as np
from surface_map import (
    SimpleCNN, 
    process_image_patch_cnn, 
    process_image_unet, 
    process_image_multiclass
)
from train_unet import UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image = cv2.imread("images/test_image.jpg")
    if image is None:
        print("Error: Could not load images/test_image.jpg")
        return

    print("Running Patch-based CNN...")
    cnn_model = SimpleCNN(num_classes=2).to(device)
    if os.path.exists("models/cnn_binary_model.pth"):
        cnn_model.load_state_dict(torch.load("models/cnn_binary_model.pth", map_location=device))
        cnn_model.eval()
        mask, metrics, _, _, blended, _ = process_image_patch_cnn(
            image, cnn_model, device,
            drone_height=80, fov_deg=84, use_calibration=True, known_road_width=25
        )
        cv2.imwrite("outputs/eval_patch_cnn.jpg", blended)
        print("Patch CNN Metrics:")
        print(json.dumps(metrics, indent=2))
    else:
        print("cnn_binary_model.pth not found.")

    print("\nRunning U-Net Segmentation...")
    unet_model = UNet().to(device)
    if os.path.exists("models/unet_road_model.pth"):
        unet_model.load_state_dict(torch.load("models/unet_road_model.pth", map_location=device))
        unet_model.eval()
        mask_u, metrics_u, _, _, blended_u, _ = process_image_unet(
            image, unet_model, device,
            drone_height=80, fov_deg=84, use_calibration=True, known_road_width=25
        )
        cv2.imwrite("outputs/eval_unet.jpg", blended_u)
        print("U-Net Metrics:")
        print(json.dumps(metrics_u, indent=2))
    else:
        print("unet_road_model.pth not found.")

    print("\nRunning Multi-Class CNN...")
    if os.path.exists("models/cnn_multiclass_model.pth") and os.path.exists("models/multiclass_mapping.json"):
        with open("models/multiclass_mapping.json", "r") as f:
            class_mapping = json.load(f)
        multiclass_model = SimpleCNN(num_classes=len(class_mapping)).to(device)
        multiclass_model.load_state_dict(torch.load("models/cnn_multiclass_model.pth", map_location=device))
        multiclass_model.eval()
        multiclass_map, blended_multi = process_image_multiclass(
            image, multiclass_model, device, class_mapping
        )
        cv2.imwrite("outputs/eval_multiclass.jpg", blended_multi)
        
        # Reverse mapping for names
        idx_to_class = {v: k for k, v in class_mapping.items()}
        
        # Color palette definition matching BGR colors
        class_colors = {
            "Buildings": (128, 128, 128),
            "Cemented road": (220, 220, 220),
            "Coloured paver path": (180, 105, 255),
            "Grass Ground": (50, 220, 50),
            "Non-Road": (30, 80, 150),
            "Road": (0, 0, 255),
            "Sand path": (100, 220, 220),
            "Vegetations": (0, 100, 0),
            "solar panels": (255, 0, 0)
        }
        
        h, w, _ = image.shape
        flat = multiclass_map.reshape(-1, 3)
        unique, counts = np.unique(flat, axis=0, return_counts=True)
        
        print("Multi-Class Coverage:")
        for uc, count in zip(unique, counts):
            percent = (count / (h * w)) * 100
            # Match colors back to class names
            matched_class = "Unknown"
            for name, bgr in class_colors.items():
                if np.array_equal(uc, bgr):
                    matched_class = name
                    break
            print(f" - {matched_class}: {percent:.2f}%")
    else:
        print("cnn_multiclass_model.pth or multiclass_mapping.json not found.")

if __name__ == "__main__":
    main()
