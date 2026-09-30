import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from core.roboflow_parser import InstanceAnnotation

class MaskTransformer:
    """
    Transforms polygon vertices, bounding boxes, and binary/semantic masks 
    across coordinate spaces, image resizes, and Homography perspective warps.
    """
    
    @staticmethod
    def transform_annotation(
        ann: InstanceAnnotation,
        H: Optional[np.ndarray] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        clip_bounds: Optional[Tuple[int, int]] = None
    ) -> InstanceAnnotation:
        """
        Applies scaling (scale_x, scale_y) and/or Homography transformation H 
        to an InstanceAnnotation.
        """
        pts = np.array(ann.polygon, dtype=np.float32)
        if len(pts) == 0:
            return ann
            
        # Apply scaling
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        
        # Apply Homography H matrix if provided
        if H is not None and not np.array_equal(H, np.eye(3)):
            # Convert to homogeneous coordinates (N, 3)
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_homo = np.hstack([pts, ones])
            
            # H dot pts^T -> (3, N)
            warped_homo = np.dot(H, pts_homo.T).T
            
            # Divide by homogeneous z coordinate
            z = np.maximum(warped_homo[:, 2:3], 1e-7)
            warped_pts = warped_homo[:, 0:2] / z
            pts = warped_pts
            
        # Clip points to image bounds if specified
        if clip_bounds is not None:
            w_max, h_max = clip_bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, w_max - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h_max - 1)
            
        new_poly = [(float(p[0]), float(p[1])) for p in pts]
        
        # Recompute bounding box [x, y, w, h]
        xs = [p[0] for p in new_poly]
        ys = [p[1] for p in new_poly]
        new_bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        
        return InstanceAnnotation(
            class_id=ann.class_id,
            class_name=ann.class_name,
            polygon=new_poly,
            bbox=new_bbox,
            is_normalized=False,
            confidence=ann.confidence
        )

    @staticmethod
    def render_instance_mask(ann: InstanceAnnotation, height: int, width: int) -> np.ndarray:
        """Renders a single binary instance mask array (height, width) with values 0 or 1."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if not ann.polygon:
            return mask
            
        pts = np.array(ann.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def render_semantic_mask(annotations: List[InstanceAnnotation], height: int, width: int) -> np.ndarray:
        """
        Renders a multi-class semantic integer array (height, width) with values 0..5.
        Higher damage severity classes overwrite background or lower classes.
        """
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Sort annotations by class_id so higher damage classes take precedence if overlapping
        sorted_anns = sorted(annotations, key=lambda a: a.class_id)
        
        for ann in sorted_anns:
            if not ann.polygon:
                continue
            pts = np.array(ann.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(semantic_mask, [pts], int(ann.class_id))
            
        return semantic_mask

    @staticmethod
    def prune_spatial_mask_pairs(
        before_anns: List[InstanceAnnotation],
        after_anns: List[InstanceAnnotation],
        height: int,
        width: int,
        min_iou: float = 0.15,
        min_overlap: float = 0.20
    ) -> Tuple[List[InstanceAnnotation], List[InstanceAnnotation], Dict[str, Any]]:
        """
        Prunes pre-disaster and post-disaster building mask pairs based on spatial pixel overlap.
        Retains only mask pairs that share spatial correspondence across the image canvas.
        
        Returns:
          - pruned_before_anns: Pruned pre-disaster building annotations
          - pruned_after_anns: Pruned post-disaster building annotations
          - pruning_stats: Dictionary summarizing total, kept, and pruned counts
        """
        if not before_anns or not after_anns:
            return before_anns, after_anns, {
                "before_total": len(before_anns),
                "after_total": len(after_anns),
                "before_kept": len(before_anns),
                "after_kept": len(after_anns),
                "pruned_count": 0
            }

        # Render binary masks for each instance annotation
        before_masks = [MaskTransformer.render_instance_mask(ann, height, width) for ann in before_anns]
        after_masks = [MaskTransformer.render_instance_mask(ann, height, width) for ann in after_anns]

        n_b = len(before_anns)
        n_a = len(after_anns)
        
        iou_matrix = np.zeros((n_b, n_a), dtype=np.float32)
        overlap_matrix = np.zeros((n_b, n_a), dtype=np.float32)

        for i in range(n_b):
            m_b = before_masks[i]
            area_b = np.sum(m_b > 0)
            if area_b == 0:
                continue
            for j in range(n_a):
                m_a = after_masks[j]
                area_a = np.sum(m_a > 0)
                if area_a == 0:
                    continue
                intersection = np.sum((m_b > 0) & (m_a > 0))
                if intersection > 0:
                    union = area_b + area_a - intersection
                    iou = float(intersection / union) if union > 0 else 0.0
                    overlap = float(intersection / min(area_b, area_a))
                    iou_matrix[i, j] = iou
                    overlap_matrix[i, j] = overlap

        # Bijective 1-to-1 Greedy Matching: Pair highest overlap masks first and eliminate duplicates
        candidate_pairs = []
        for i in range(n_b):
            for j in range(n_a):
                iou = float(iou_matrix[i, j])
                overlap = float(overlap_matrix[i, j])
                if iou >= min_iou or overlap >= min_overlap:
                    score = float(iou + overlap)
                    candidate_pairs.append((score, i, j))

        # Sort candidate pairs by descending overlap score
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        matched_before_set = set()
        matched_after_set = set()
        bijective_pairs = []

        for score, i, j in candidate_pairs:
            if i not in matched_before_set and j not in matched_after_set:
                matched_before_set.add(i)
                matched_after_set.add(j)
                bijective_pairs.append((i, j))

        # Filter annotations to maintain 1-to-1 correspondence
        pruned_before = [before_anns[i] for i, j in bijective_pairs]
        pruned_after = [after_anns[j] for i, j in bijective_pairs]


        stats = {
            "before_total": n_b,
            "after_total": n_a,
            "before_kept": len(pruned_before),
            "after_kept": len(pruned_after),
            "before_pruned": n_b - len(pruned_before),
            "after_pruned": n_a - len(pruned_after),
            "min_iou": min_iou,
            "min_overlap": min_overlap
        }

        return pruned_before, pruned_after, stats

