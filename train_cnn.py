import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import os
import argparse

# =====================================
# SETTINGS
# =====================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0005

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

BINARY_MAPPING = {
    "Buildings": 0,
    "Grass Ground": 0,
    "Non-Road": 0,
    "Vegetations": 0,
    "solar panels": 0,
    "Road": 1,
    "Cemented road": 1,
    "Coloured paver path": 1,
    "Sand path": 1
}

# =====================================
# CUSTOM DATASETS
# =====================================

class BinaryTerrainDataset(Dataset):
    """
    Wraps an ImageFolder dataset to map its multi-class targets to binary targets (Road vs Non-Road).
    """
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = ["Non-Road", "Road"]
        self.class_to_idx = {"Non-Road": 0, "Road": 1}
        
        # Get source class index mappings
        self.mapping = []
        for class_name, idx in self.dataset.class_to_idx.items():
            # If the class name exists in our binary mapping, use it, else default to Non-Road (0)
            binary_label = BINARY_MAPPING.get(class_name, 0)
            self.mapping.append(binary_label)
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        binary_label = self.mapping[label]
        return img, binary_label

# =====================================
# CNN MODEL
# =====================================

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

# =====================================
# TRAINING & VALIDATION PIPELINE
# =====================================

def train(mode="binary", epochs=20):
    print(f"\n[Train] Starting training in {mode.upper()} mode...")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if mode == "binary":
        # Load binary mapped datasets
        train_dataset = BinaryTerrainDataset(TRAIN_DIR, transform=train_transform)
        val_dataset = BinaryTerrainDataset(VAL_DIR, transform=val_transform)
        num_classes = 2
        save_path = "models/cnn_binary_model.pth"
    else:
        # Load full 9-class dataset
        full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        class_mapping = full_train_dataset.class_to_idx
        print("Class Mapping:", class_mapping)
        num_classes = len(class_mapping)
        
        # Save mapping to file
        os.makedirs("models", exist_ok=True)
        with open("models/multiclass_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=4)
            
        # Split train dataset randomly to create val dataset with all classes
        dataset_size = len(full_train_dataset)
        train_size = int(0.85 * dataset_size)
        val_size = dataset_size - train_size
        
        # Split indexes
        indices = list(range(dataset_size))
        # Shuffle indices
        import random
        random.seed(42)
        random.shuffle(indices)
        
        train_dataset = Subset(full_train_dataset, indices[:train_size])
        val_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), indices[train_size:])
        save_path = "models/cnn_multiclass_model.pth"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        correct = 0
        total = 0
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item() * images.size(0)

        train_acc = 100 * correct / total
        epoch_loss = train_loss / len(train_loader.dataset)

        # ---- VALIDATION ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("Model improved and saved.")

    print("\nTraining complete.")
    print("Best Validation Accuracy:", best_val_acc)

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Train CNN model on drone dataset.")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"],
                        help="Training mode: 'binary' or 'multiclass'")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    args = parser.parse_args()
    
    train(mode=args.mode, epochs=args.epochs)