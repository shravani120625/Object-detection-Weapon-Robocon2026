"""
Prepare raw JPEG images for YOLOv8 training.
Converts raw images to .npy format and creates YOLO labels.
"""

import os
import numpy as np
import cv2
from pathlib import Path

# Configuration
RAW_IMAGES_DIR = r"C:\Users\ASUS\OneDrive\Documents\Prixx Robotics\Weapon detection Robocon2026\New raw"
OUTPUT_IMAGES_DIR = r"C:\Users\ASUS\OneDrive\Documents\Prixx Robotics\Weapon detection Robocon2026\dataset\train\images"
OUTPUT_LABELS_DIR = r"C:\Users\ASUS\OneDrive\Documents\Prixx Robotics\Weapon detection Robocon2026\dataset\train\labels"

# Class mapping (map filenames to class IDs)
CLASS_MAPPING = {
    "fist": 0,
    "palm": 1,
    "spearhead": 2,
}

TARGET_SIZE = 640  # Resize all images to 640x640


def extract_class_from_filename(filename):
    """Extract class name from filename."""
    filename_lower = filename.lower()
    
    for class_name in CLASS_MAPPING.keys():
        if class_name in filename_lower:
            return class_name
    
    print(f"⚠️  Warning: Could not determine class for '{filename}'. Skipping...")
    return None


def preprocess_image(image_path, target_size=640):
    """Load image, resize to target size, normalize."""
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size (maintain aspect ratio with padding if needed)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target_size x target_size
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)  # Gray padding
    )
    
    # Ensure it's exactly target_size x target_size
    img_final = img_padded[:target_size, :target_size]
    
    # Convert to uint8 if not already
    if img_final.dtype != np.uint8:
        img_final = np.uint8(img_final)
    
    return img_final


def create_yolo_label(class_id):
    """
    Create a YOLO label file content for full image classification.
    Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    For full image: class = class_id, box = (0.5, 0.5, 1.0, 1.0)
    """
    return f"{class_id} 0.5 0.5 1.0 1.0\n"


def main():
    # Create output directories if they don't exist
    Path(OUTPUT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_LABELS_DIR).mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted([f for f in os.listdir(RAW_IMAGES_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not raw_files:
        print("❌ No image files found in:", RAW_IMAGES_DIR)
        return
    
    print(f"\n📸 Found {len(raw_files)} images to process\n")
    
    successful = 0
    skipped = 0
    
    for i, filename in enumerate(raw_files, 1):
        print(f"[{i}/{len(raw_files)}] Processing: {filename}")
        
        # Extract class from filename
        class_name = extract_class_from_filename(filename)
        if class_name is None:
            skipped += 1
            continue
        
        class_id = CLASS_MAPPING[class_name]
        
        try:
            # Load and preprocess image
            image_path = os.path.join(RAW_IMAGES_DIR, filename)
            img_array = preprocess_image(image_path, TARGET_SIZE)
            
            # Generate output filename (without extension for .npy)
            base_name = Path(filename).stem
            npy_path = os.path.join(OUTPUT_IMAGES_DIR, f"{base_name}.npy")
            label_path = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")
            
            # Save as .npy
            np.save(npy_path, img_array)
            
            # Save YOLO label
            with open(label_path, 'w') as f:
                f.write(create_yolo_label(class_id))
            
            print(f"   ✅ Saved: {base_name}.npy ({class_name})")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {successful} images")
    print(f"⚠️  Skipped: {skipped} images")
    print(f"{'='*60}")
    print(f"\n📁 Output saved to:")
    print(f"   Images: {OUTPUT_IMAGES_DIR}")
    print(f"   Labels: {OUTPUT_LABELS_DIR}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the converted images visually (optional)")
    print(f"   2. Run: python -m ultralytics.yolo detect train")
    print(f"   3. Or use your existing training script")


if __name__ == "__main__":
    main()
