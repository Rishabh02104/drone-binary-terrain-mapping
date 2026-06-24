import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image

# =====================================
# SETTINGS
# =====================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 30
LR = 0.0001
IMG_SIZE = (256, 256)

IMAGES_DIR = "segmentation_dataset/Images"
MASKS_DIR = "segmentation_dataset/masks"
MODEL_SAVE_PATH = "models/unet_road_model.pth"

# =====================================
# DATASET
# =====================================

class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(256, 256), domain='all'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        
        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.dng'))])
        
        # Filter by domain prefix if requested
        if domain == 'campus':
            self.images = [f for f in self.images if f.startswith("campus_")]
        elif domain == 'auroville':
            self.images = [f for f in self.images if f.startswith("auroville_")]
            
        # Filter images that have a corresponding mask
        self.valid_pairs = []
        for img_name in self.images:
            base_name, _ = os.path.splitext(img_name)
            # Find any mask file with matching base name (e.g. base_name.png, base_name.jpg)
            mask_found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_name = base_name + ext
                if os.path.exists(os.path.join(masks_dir, mask_name)):
                    self.valid_pairs.append((img_name, mask_name))
                    mask_found = True
                    break
        
        print(f"Loaded {len(self.valid_pairs)} image-mask pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image (OpenCV loads BGR, convert to RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Handle possible DNG or other formats
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_path).convert("L"))
            
        # Resize
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize and convert to FloatTensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        # Threshold mask to ensure binary 0 or 1
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor

# =====================================
# U-NET MODEL
# =====================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        
        b = self.bottleneck(p4)
        
        u4 = self.up4(b)
        if u4.shape != d4.shape:
            u4 = nn.functional.interpolate(u4, size=d4.shape[2:])
        merge4 = torch.cat([u4, d4], dim=1)
        c4 = self.conv_up4(merge4)
        
        u3 = self.up3(c4)
        if u3.shape != d3.shape:
            u3 = nn.functional.interpolate(u3, size=d3.shape[2:])
        merge3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv_up3(merge3)
        
        u2 = self.up2(c3)
        if u2.shape != d2.shape:
            u2 = nn.functional.interpolate(u2, size=d2.shape[2:])
        merge2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv_up2(merge2)
        
        u1 = self.up1(c2)
        if u1.shape != d1.shape:
            u1 = nn.functional.interpolate(u1, size=d1.shape[2:])
        merge1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv_up1(merge1)
        
        return self.out_conv(c1)

# =====================================
# DICE LOSS (FOR IMPROVED SEGMENTATION)
# =====================================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1.0):
        bce_loss = self.bce(inputs, targets)
        
        # Sigmoid activation to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        
        return bce_loss + dice_loss

# =====================================
# TRAINING PIPELINE
# =====================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train U-Net road segmentation model.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "campus", "auroville"], help="Subset of domain data to train on")
    args = parser.parse_args()

    if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
        print("Error: Dataset folders not found. Please setup segmentation_dataset/Images and segmentation_dataset/masks.")
        exit()

    # Load dataset
    dataset = RoadDataset(IMAGES_DIR, MASKS_DIR, img_size=IMG_SIZE, domain=args.domain)
    if len(dataset) == 0:
        print("No masks found in segmentation_dataset/masks. Please annotate some images using the dashboard first!")
        exit()

    # Split train/val with safeguards for small datasets
    if len(dataset) < 4:
        train_dataset = dataset
        val_dataset = dataset
        actual_batch_size = 1
    else:
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        actual_batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)

    model = UNet().to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nStarting training on {DEVICE} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model improved and saved.")
            
    print("Training complete.")
