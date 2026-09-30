import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.roboflow_parser import InstanceAnnotation
from core.mask_transformer import MaskTransformer

COLOR_PALETTE_BGR = {
    0: (0, 0, 0),        # 0: Background -> Black
    1: (100, 220, 0),    # 1: Intact -> Green
    2: (0, 200, 255),    # 2: Minor Damage -> Yellow
    3: (0, 120, 255),    # 3: Major Damage -> Orange
    4: (40, 40, 230),    # 4: Destroyed -> Red
    5: (220, 50, 160),   # 5: Unclassified -> Purple
    6: (240, 210, 30)    # 6: Obscured Building -> Bright Cyan/Teal
}

class SemanticExporter:
    """
    Exports synchronized dual-stream image pairs and 0-6 integer semantic masks 
    for Siamese Swin-UNet building damage segmentation.
    Supports Train / Val / Test sequence-grouped dataset splitting to prevent data leakage.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.dirs = {
            "before": os.path.join(output_dir, "before_aligned"),
            "after": os.path.join(output_dir, "after_reference"),
            "masks_before": os.path.join(output_dir, "semantic_masks_before"),
            "masks_after": os.path.join(output_dir, "semantic_masks_after"),
            "masks_before_rgb": os.path.join(output_dir, "semantic_masks_before_rgb"),
            "masks_after_rgb": os.path.join(output_dir, "semantic_masks_after_rgb"),
            "masks": os.path.join(output_dir, "semantic_masks"),
            "masks_rgb": os.path.join(output_dir, "semantic_masks_rgb"),
            "masks_scaled": os.path.join(output_dir, "semantic_masks_scaled"),
            "numpy": os.path.join(output_dir, "numpy_tensors"),
            "vis": os.path.join(output_dir, "visualizations")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
            
    def export_sample(
        self,
        sample_name: str,
        before_aligned: np.ndarray,
        after_ref: np.ndarray,
        annotations: List[InstanceAnnotation],
        before_annotations: Optional[List[InstanceAnnotation]] = None,
        override_dirs: Optional[Dict[str, str]] = None,
        min_spatial_iou: float = 0.15,
        min_overlap: float = 0.20
    ) -> Dict[str, str]:
        """
        Exports a single Siamese sample pair with pruned Pre-Disaster and Post-Disaster semantic masks:
          - Pre-Disaster mask (0: Background, 1: Intact Footprint, 6: Obscured)
          - Post-Disaster mask (0: Background, 1: Intact, 2: Minor, 3: Major, 4: Destroyed, 5: Unclassified, 6: Obscured)
          - Dual-channel numpy array of shape (2, H, W) -> [pre_mask, post_mask]
        """
        target_dirs = override_dirs or self.dirs
        h, w = after_ref.shape[:2]
        
        # Prepare initial before annotations list
        init_before_anns = []
        source_anns = before_annotations if (before_annotations is not None and len(before_annotations) > 0) else annotations
        
        for ann in source_anns:
            b_cid = 1 if ann.class_id in (0, 1, 2, 3, 4) else ann.class_id
            b_cname = "building_intact" if b_cid == 1 else ann.class_name
            init_before_anns.append(InstanceAnnotation(
                class_id=b_cid,
                class_name=b_cname,
                polygon=ann.polygon,
                bbox=ann.bbox,
                confidence=ann.confidence
            ))


        # Spatial Overlap Pruning: Retain only building masks that share spatial correspondence across image space
        pruned_before_anns, pruned_after_anns, prune_stats = MaskTransformer.prune_spatial_mask_pairs(
            init_before_anns, annotations, h, w, min_iou=min_spatial_iou, min_overlap=min_overlap
        )
        
        # Render clean, pruned Post-Disaster and Pre-Disaster Semantic Masks
        after_mask = MaskTransformer.render_semantic_mask(pruned_after_anns, h, w)
        before_mask = MaskTransformer.render_semantic_mask(pruned_before_anns, h, w)


        
        # Save before and after images
        before_path = os.path.join(target_dirs["before"], f"{sample_name}.png")
        cv2.imwrite(before_path, before_aligned)
        
        after_path = os.path.join(target_dirs["after"], f"{sample_name}.png")
        cv2.imwrite(after_path, after_ref)
        
        # Save separate 1-channel pre and post PNG masks
        mask_before_path = os.path.join(target_dirs["masks_before"], f"{sample_name}.png")
        cv2.imwrite(mask_before_path, before_mask)

        mask_after_path = os.path.join(target_dirs["masks_after"], f"{sample_name}.png")
        cv2.imwrite(mask_after_path, after_mask)

        # Legacy combined mask for backwards compatibility
        mask_path = os.path.join(target_dirs["masks"], f"{sample_name}.png")
        cv2.imwrite(mask_path, after_mask)

        # RGB Visual Masks for Pre and Post streams
        rgb_before_mask = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_after_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, color_bgr in COLOR_PALETTE_BGR.items():
            rgb_before_mask[before_mask == cid] = color_bgr
            rgb_after_mask[after_mask == cid] = color_bgr
            
        mask_before_rgb_path = os.path.join(target_dirs["masks_before_rgb"], f"{sample_name}_rgb.png")
        cv2.imwrite(mask_before_rgb_path, rgb_before_mask)

        mask_after_rgb_path = os.path.join(target_dirs["masks_after_rgb"], f"{sample_name}_rgb.png")
        cv2.imwrite(mask_after_rgb_path, rgb_after_mask)

        mask_rgb_path = os.path.join(target_dirs["masks_rgb"], f"{sample_name}_rgb.png")
        cv2.imwrite(mask_rgb_path, rgb_after_mask)

        # Scaled Grayscale PNG (0..240)
        scaled_mask = (after_mask.astype(np.uint16) * 40).clip(0, 255).astype(np.uint8)
        mask_scaled_path = os.path.join(target_dirs["masks_scaled"], f"{sample_name}_scaled.png")
        cv2.imwrite(mask_scaled_path, scaled_mask)
        
        # 3. Save Dual-Channel numpy tensor (2, H, W)
        dual_tensor = np.stack([before_mask, after_mask], axis=0)
        npy_path = os.path.join(target_dirs["numpy"], f"{sample_name}.npy")
        np.save(npy_path, dual_tensor)
        
        # 4. Create color-coded overlay visual preview
        overlay_before = cv2.addWeighted(before_aligned, 0.65, rgb_before_mask, 0.35, 0)
        overlay_after = cv2.addWeighted(after_ref, 0.65, rgb_after_mask, 0.35, 0)
        
        vis_panel = np.hstack([overlay_before, overlay_after])
        vis_path = os.path.join(target_dirs["vis"], f"{sample_name}_vis.jpg")
        cv2.imwrite(vis_path, vis_panel)
        
        return {
            "before_path": before_path,
            "after_path": after_path,
            "mask_before_path": mask_before_path,
            "mask_after_path": mask_after_path,
            "mask_path": mask_path,
            "mask_rgb_path": mask_rgb_path,
            "mask_scaled_path": mask_scaled_path,
            "npy_path": npy_path,
            "vis_path": vis_path
        }


    @staticmethod
    def partition_dataset_splits(
        sample_names: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, str]:
        """
        Groups sample names by video sequence prefix to prevent spatial data leakage,
        then partitions groups into Train, Val, and Test splits.
        """
        import re
        import random
        
        random.seed(seed)
        
        # Group by prefix sequence identifier (e.g., scene1_frame01 -> scene1)
        sequence_groups: Dict[str, List[str]] = {}
        for name in sample_names:
            match = re.match(r"^([a-zA-Z0-9_\-]+?)(?:_frame|_img|\d+)?$", name)
            seq_id = match.group(1) if match else name
            sequence_groups.setdefault(seq_id, []).append(name)
            
        group_keys = list(sequence_groups.keys())
        random.shuffle(group_keys)
        
        n_groups = len(group_keys)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio)) if val_ratio > 0 and n_groups >= 3 else (1 if n_groups >= 2 else 0)
        
        train_groups = set(group_keys[:n_train])
        val_groups = set(group_keys[n_train:n_train + n_val])
        test_groups = set(group_keys[n_train + n_val:])
        
        split_assignment = {}
        for seq_id, names in sequence_groups.items():
            if seq_id in train_groups:
                split = "train"
            elif seq_id in val_groups:
                split = "val"
            else:
                split = "test"
                
            for name in names:
                split_assignment[name] = split
                
        return split_assignment


