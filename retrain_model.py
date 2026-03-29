"""
Retrain YOLOv8 model with new data (palm images).
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# Path to your dataset
dataset_dir = Path("dataset")
weights_path = Path("runs/detect/train/weights/best.pt")

# Load the current trained weights (or start fresh)
model = YOLO(str(weights_path))  # Load existing weights

device = 0 if torch.cuda.is_available() else "cpu"
batch_size = 16 if torch.cuda.is_available() else 8

print("Starting training with new palm images...")
print(f"Dataset: {dataset_dir.resolve()}")
print(f"Weights: {weights_path.resolve()}")
print(f"Device: {device}")

# Train the model
results = model.train(
    data="data.yaml",  # Your dataset config (if exists)
    epochs=50,
    imgsz=640,
    batch=batch_size,
    device=device,  # Use GPU when available, else CPU
    patience=20,
    save=True,
    project="runs/detect",
    name="train_v2",  # New training run name
    verbose=True,
)

print("\n✅ Training complete!")
print(f"Weights saved to: runs/detect/train_v2/weights/best.pt")
