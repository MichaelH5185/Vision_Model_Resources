import os
import json
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class InstanceAnnotation:
    class_id: int
    class_name: str
    polygon: List[Tuple[float, float]]  # Normalized or absolute [(x, y), ...]
    bbox: Tuple[float, float, float, float]  # (x, y, width, height) in pixels
    is_normalized: bool = False
    confidence: float = 1.0

class RoboflowParser:
    """
    Parses annotations exported from Roboflow in COCO JSON, Roboflow Polygon JSON,
    YOLO Segmentation TXT, or PNG mask formats.
    """
    def __init__(self, default_class_map: Optional[Dict[int, str]] = None):
        self.default_class_map = default_class_map or {
            0: "background",
            1: "building_intact",
            2: "building_minor_damage",
            3: "building_major_damage",
            4: "building_destroyed",
            5: "building_unclassified",
            6: "building_obscured"
        }


    def parse_coco_json(self, json_path: str, image_filename: Optional[str] = None) -> Tuple[Dict[str, Any], List[InstanceAnnotation]]:
        """Parses standard COCO JSON format."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        images = {img['id']: img for img in data.get('images', [])}
        
        target_img_id = None
        if image_filename:
            target_base = os.path.splitext(os.path.basename(image_filename))[0].replace('_before', '').replace('_after', '').replace('_aligned', '').replace('_reference', '')
            for img_id, img_info in images.items():
                fn_base = os.path.splitext(os.path.basename(img_info['file_name']))[0].replace('_before', '').replace('_after', '').replace('_aligned', '').replace('_reference', '')
                if fn_base == target_base:
                    target_img_id = img_id
                    break
        elif len(images) > 0:
            target_img_id = list(images.keys())[0]


        annotations = []
        for ann in data.get('annotations', []):
            if target_img_id is not None and ann.get('image_id') != target_img_id:
                continue
                
            class_id = ann.get('category_id', 0)
            class_name = categories.get(class_id, f"class_{class_id}")
            
            # Bounding box [x, y, w, h]
            bbox = tuple(ann.get('bbox', [0, 0, 0, 0]))
            
            # Polygons in segmentation field
            polygons = []
            seg = ann.get('segmentation', [])
            if isinstance(seg, list) and len(seg) > 0:
                for poly_flat in seg:
                    if isinstance(poly_flat, list) and len(poly_flat) >= 6:
                        pts = [(poly_flat[i], poly_flat[i+1]) for i in range(0, len(poly_flat), 2)]
                        polygons.append(pts)
                        
            # If segmentation is empty, fallback to bbox rectangle polygon
            if not polygons and len(bbox) == 4:
                x, y, w, h = bbox
                polygons = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
                
            for poly in polygons:
                annotations.append(InstanceAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    polygon=poly,
                    bbox=bbox,
                    is_normalized=False
                ))
                
        img_info = images.get(target_img_id, {}) if target_img_id else {}
        return img_info, annotations

    def parse_yolo_txt(self, txt_path: str, img_width: int, img_height: int) -> List[InstanceAnnotation]:
        """Parses YOLOv8/v9 Segmentation TXT files (<class_id> x1 y1 x2 y2 ... normalized)."""
        annotations = []
        if not os.path.exists(txt_path):
            return annotations
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                
                # Truncate trailing confidence score if coordinate count is odd
                if len(coords) >= 7 and len(coords) % 2 != 0:
                    coords = coords[:-1]

                # Check if coordinates are normalized (0.0 .. 1.0) or already in pixel space
                is_normalized = all(c <= 1.05 for c in coords)
                
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    if is_normalized:
                        pts_px = [
                            (coords[i] * img_width, coords[i+1] * img_height) 
                            for i in range(0, len(coords), 2)
                        ]
                    else:
                        pts_px = [
                            (coords[i], coords[i+1]) 
                            for i in range(0, len(coords), 2)
                        ]
                elif len(coords) == 4:
                    cx, cy, w, h = coords
                    if is_normalized:
                        x1 = (cx - w / 2) * img_width
                        y1 = (cy - h / 2) * img_height
                        w_px = w * img_width
                        h_px = h * img_height
                    else:
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        w_px = w
                        h_px = h
                    pts_px = [(x1, y1), (x1 + w_px, y1), (x1 + w_px, y1 + h_px), (x1, y1 + h_px)]
                else:
                    continue
                    
                xs = [p[0] for p in pts_px]
                ys = [p[1] for p in pts_px]
                bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                class_name = self.default_class_map.get(class_id, f"class_{class_id}")
                
                annotations.append(InstanceAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    polygon=pts_px,
                    bbox=bbox,
                    is_normalized=False
                ))
        return annotations


    def parse_roboflow_json(self, json_path: str) -> List[InstanceAnnotation]:
        """Parses Roboflow native Polygon JSON export format."""
        annotations = []
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data.get('predictions', data.get('annotations', [])):
            class_id = item.get('class_id', item.get('category_id', 0))
            class_name = item.get('class', item.get('label', f"class_{class_id}"))
            
            points = item.get('points', item.get('polygon', []))
            poly = []
            if points and isinstance(points, list):
                for p in points:
                    if isinstance(p, dict):
                        poly.append((float(p.get('x', 0)), float(p.get('y', 0))))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        poly.append((float(p[0]), float(p[1])))
                        
            if not poly and 'x' in item and 'y' in item and 'width' in item and 'height' in item:
                x, y, w, h = item['x'], item['y'], item['width'], item['height']
                poly = [(x - w/2, y - h/2), (x + w/2, y - h/2), (x + w/2, y + h/2), (x - w/2, y + h/2)]
                
            if poly:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                annotations.append(InstanceAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    polygon=poly,
                    bbox=bbox,
                    is_normalized=False
                ))
        return annotations

    def export_coco_json(self, image_filename: str, width: int, height: int, annotations: List[InstanceAnnotation], output_json_path: str):
        """Exports annotations back into COCO JSON format for Roboflow import."""
        coco_data = {
            "images": [{
                "id": 1,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_filename)
            }],
            "categories": [
                {"id": cid, "name": name, "supercategory": "building"} 
                for cid, name in self.default_class_map.items()
            ],
            "annotations": []
        }
        
        ann_id = 1
        for ann in annotations:
            flat_poly = []
            for pt in ann.polygon:
                flat_poly.extend([float(pt[0]), float(pt[1])])
                
            x, y, w, h = ann.bbox
            coco_ann = {
                "id": ann_id,
                "image_id": 1,
                "category_id": ann.class_id,
                "segmentation": [flat_poly],
                "area": float(w * h),
                "bbox": [float(x), float(y), float(w), float(h)],
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_ann)
            ann_id += 1
            
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
            
    def export_yolo_txt(self, width: int, height: int, annotations: List[InstanceAnnotation], output_txt_path: str):
        """Exports annotations to YOLO segmentation format (<class_id> x1 y1 x2 y2 ... normalized)."""
        lines = []
        for ann in annotations:
            norm_pts = []
            for pt in ann.polygon:
                nx = max(0.0, min(1.0, pt[0] / width))
                ny = max(0.0, min(1.0, pt[1] / height))
                norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
            lines.append(f"{ann.class_id} " + " ".join(norm_pts))
            
        os.makedirs(os.path.dirname(os.path.abspath(output_txt_path)), exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
