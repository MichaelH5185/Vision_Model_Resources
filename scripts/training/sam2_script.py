""" 
This script doesn't produce very good results, we should return to this idea in the future though
"""
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Configuration & Paths
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml" # Match your trained architecture
    finetuned_checkpoint = "sam2_finetuned_inference_weights.pt" # Path to your weights
    
    val_img_dir = "/home/mjhf89/ondemand/data/sys/dashboard/batch_connect/sys/jupyter/Animal_Project_Main-17/test/images"
    val_mask_dir = "/home/mjhf89/ondemand/data/sys/dashboard/batch_connect/sys/jupyter/Animal_Project_Main-17/test/masks"
    output_vis_dir = "./evaluation_outputs"
    os.makedirs(output_vis_dir, exist_ok=True)
    
    print("Loading fine-tuned SAM 2 model...")
    model = build_sam2(model_cfg, finetuned_checkpoint, device=device)
    model.eval()
    
    # 2. Initialize Automatic Mask Generator (Acts like a standard CNN predictor)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.85,
        min_mask_region_area=100
    )
    
    image_files = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))])
    ious = []
    
    print(f"Running evaluation on {len(image_files)} validation images...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(val_img_dir, img_name)
        mask_path = os.path.join(val_mask_dir, os.path.splitext(img_name)[0] + ".png")
        
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None: gt_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # 3. Generate masks automatically (Dense grid prediction)
        masks = mask_generator.generate(image_rgb)
        
        # Combine generated automatic masks into a single binary prediction map
        pred_binary = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for ann in masks:
            pred_binary = np.logical_or(pred_binary, ann['segmentation']).astype(np.uint8)
            
        # 4. Calculate Metric (IoU)
        iou = calculate_iou(pred_binary, gt_binary)
        ious.append(iou)
        
        # 5. Save Visual Comparison Overlay
        overlay = image_rgb.copy()
        overlay[pred_binary == 1] = [0, 255, 0] # Highlight predictions in green
        vis_img = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_vis_dir, f"pred_{img_name}"), vis_img_bgr)

    mean_iou = np.mean(ious) if ious else 0.0
    print(f"\n--- Evaluation Complete ---")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print(f"Visual results saved to {output_vis_dir}")

evaluate_and_visualize()