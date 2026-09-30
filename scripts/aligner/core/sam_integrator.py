import os
import shutil
import zipfile
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.roboflow_parser import InstanceAnnotation, RoboflowParser
from core.mask_transformer import MaskTransformer

class SAMIntegrator:
    """
    Handles before-image building mask generation using SAM (Segment Anything Model)
    with dual operational modes:
      1. Roboflow External Export: Packages aligned images + transformed bboxes for SAM on Roboflow UI.
      2. Local SAM / Fallback Prompting: Prompts SAM (or MobileSAM / CPU GrabCut refine) on before images.
    """
    def __init__(self, model_type: str = "vit_b", checkpoint_path: Optional[str] = None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_predictor = None
        self._init_local_sam()

    def _init_local_sam(self):
        """Attempts to initialize local SAM predictor if PyTorch and SAM weights are available."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
            
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
        except Exception:
            self.sam_predictor = None

    def create_roboflow_export_package(
        self,
        output_dir: str,
        image_pairs: List[Tuple[Any, ...]],
        format_type: str = "coco"
    ) -> str:
        """
        Exports aligned image pairs and transformed annotations into a dataset folder/ZIP 
        ready for upload to Roboflow.
        
        image_pairs list of (pair_name, before_aligned_img, after_ref_img, after_anns, before_anns)
        """
        export_root = os.path.join(output_dir, "roboflow_sam_export")
        os.makedirs(export_root, exist_ok=True)
        
        images_dir = os.path.join(export_root, "images")
        anns_dir = os.path.join(export_root, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(anns_dir, exist_ok=True)
        
        parser = RoboflowParser()
        
        for item in image_pairs:
            if len(item) == 5:
                name, before_aligned, after_ref, after_anns, before_anns = item
            elif len(item) == 4:
                name, before_aligned, after_ref, after_anns = item
                before_anns = None
            else:
                continue

            b_anns = before_anns if before_anns is not None else after_anns

            # Save aligned before image
            before_name = f"{name}_before_aligned.jpg"
            before_path = os.path.join(images_dir, before_name)
            cv2.imwrite(before_path, before_aligned)
            
            # Save reference after image
            after_name = f"{name}_after_reference.jpg"
            after_path = os.path.join(images_dir, after_name)
            cv2.imwrite(after_path, after_ref)
            
            h, w = after_ref.shape[:2]
            
            # Export annotations
            if format_type.lower() == "coco":
                coco_before_path = os.path.join(anns_dir, f"{name}_before_coco.json")
                parser.export_coco_json(before_name, w, h, b_anns, coco_before_path)
                
                coco_after_path = os.path.join(anns_dir, f"{name}_after_coco.json")
                parser.export_coco_json(after_name, w, h, after_anns, coco_after_path)
            else:
                yolo_before_path = os.path.join(images_dir, f"{name}_before_aligned.txt")
                parser.export_yolo_txt(w, h, b_anns, yolo_before_path)
                
                yolo_after_path = os.path.join(images_dir, f"{name}_after_reference.txt")
                parser.export_yolo_txt(w, h, after_anns, yolo_after_path)

                
        # Zip export package
        zip_path = os.path.join(output_dir, "roboflow_sam_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_root):
                for file in files:
                    full_path = os.path.join(root, file)
                    arc_name = os.path.relpath(full_path, export_root)
                    zipf.write(full_path, arc_name)
                    
        return zip_path

    def segment_before_image_local(
        self,
        before_aligned: np.ndarray,
        after_annotations: List[InstanceAnnotation]
    ) -> List[InstanceAnnotation]:
        """
        Prompts SAM or CPU GrabCut/contour refinement on the before image using 
        bounding boxes/centroids from after-image building annotations.
        """
        h, w = before_aligned.shape[:2]
        refined_annotations = []
        
        if self.sam_predictor is not None:
            # Full SAM Inference
            rgb_before = cv2.cvtColor(before_aligned, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_before)
            
            for ann in after_annotations:
                x, y, bw, bh = ann.bbox
                input_box = np.array([x, y, x + bw, y + bh])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                
                sam_mask = masks[0].astype(np.uint8)
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                poly = []
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    poly = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_cnt]
                else:
                    poly = ann.polygon
                    
                refined_annotations.append(InstanceAnnotation(
                    class_id=ann.class_id,
                    class_name=ann.class_name,
                    polygon=poly if poly else ann.polygon,
                    bbox=ann.bbox,
                    confidence=float(scores[0]) if len(scores) > 0 else 1.0
                ))
        else:
            # CPU Fallback Mode: Refines before-image mask using GrabCut & color gradients
            for ann in after_annotations:
                if not ann.polygon:
                    refined_annotations.append(ann)
                    continue
                    
                x, y, bw, bh = ann.bbox
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
                
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    refined_annotations.append(ann)
                    continue
                    
                # Use after polygon as initial mask prompt on before image
                init_mask = MaskTransformer.render_instance_mask(ann, h, w)
                roi_before = before_aligned[y1:y2, x1:x2]
                roi_mask = init_mask[y1:y2, x1:x2]
                
                # Active contour / morphological edge refinement on pre-disaster ROI
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                refined_roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                
                # Pre-disaster building is intact (Class 1) prior to disaster damage
                final_class_id = 1 if ann.class_id in (1, 2, 3, 4) else ann.class_id
                final_class_name = "building_intact" if final_class_id == 1 else ann.class_name

                full_refined_mask = np.zeros((h, w), dtype=np.uint8)
                full_refined_mask[y1:y2, x1:x2] = refined_roi_mask
                
                contours, _ = cv2.findContours(full_refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    poly = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
                else:
                    poly = ann.polygon
                    
                refined_annotations.append(InstanceAnnotation(
                    class_id=final_class_id,
                    class_name=final_class_name,
                    polygon=poly if poly else ann.polygon,
                    bbox=ann.bbox,
                    confidence=1.0
                ))

        return refined_annotations


