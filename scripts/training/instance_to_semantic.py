import os
import cv2
import numpy as np
from glob import glob

def convert_yolo_to_masks(image_dir, label_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
        
    print(f"Found {len(image_paths)} images. Processing masks...")
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # Read image to obtain exact pixel dimensions (H, W)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        h, w = img.shape[:2]
        
        # Initialize an empty single-channel mask (0 = background)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check if corresponding label text file exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            instance_id = 1  # Assign a unique integer ID to each object instance
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                # First element is class ID, remaining are normalized x, y polygon vertices
                class_id = int(parts[0])
                coords = [float(val) for val in parts[1:]]
                
                if len(coords) < 6:
                    continue  # Skip malformed lines (need at least 3 points -> 6 values)
                
                # Reshape into pairs of (x, y)
                coords = np.array(coords, dtype=np.float32).reshape(-1, 2)
                
                # Denormalize coordinates back to pixel space
                coords[:, 0] *= w
                coords[:, 1] *= h
                
                # Convert to integer points for OpenCV rendering
                pts = coords.astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw filled polygon on the mask
                cv2.fillPoly(mask, [pts], color=instance_id)
                instance_id += 1
        
        # Save the generated mask as a lossless PNG
        out_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(out_path, mask)
        
    print(f"Conversion complete! Masks successfully saved to: {output_dir}")

if __name__ == "__main__":
    # Update these paths to match your actual local directory layout
    IMAGE_DIR = "./Animal_Project_Main-17/train/images"
    LABEL_DIR = "./Animal_Project_Main-17/train/labels"
    OUTPUT_DIR = "./Animal_Project_Main-17/train/masks"
    
    convert_yolo_to_masks(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)