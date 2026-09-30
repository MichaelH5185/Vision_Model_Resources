import os
import io
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from core.alignment import ImageAligner
from core.roboflow_parser import RoboflowParser, InstanceAnnotation
from core.mask_transformer import MaskTransformer
from core.sam_integrator import SAMIntegrator
from core.semantic_exporter import SemanticExporter, COLOR_PALETTE_BGR

app = FastAPI(title="Building Damage Alignment & Semantic Masking Studio")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp_export")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

aligner = ImageAligner()
parser = RoboflowParser()
sam_integrator = SAMIntegrator()

def mat_to_base64(img_bgr: np.ndarray, ext: str = ".jpg") -> str:
    _, buf = cv2.imencode(ext, img_bgr)
    return base64.b64encode(buf).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Building Damage Studio Backend Active</h1>"

@app.post("/api/align")
async def align_pair_endpoint(
    before_files: List[UploadFile] = File(...),
    after_file: UploadFile = File(...),
    ann_file: Optional[UploadFile] = File(None),
    run_sam_refine: bool = Form(False)
):
    try:
        # Read after image bytes
        after_bytes = await after_file.read()
        after_np = np.frombuffer(after_bytes, np.uint8)
        after_img = cv2.imdecode(after_np, cv2.IMREAD_COLOR)

        if after_img is None:
            raise HTTPException(status_code=400, detail="Invalid after image file.")

        # Read before image(s) / sequence frames
        candidate_frames = []
        for bf in before_files:
            b_bytes = await bf.read()
            b_np = np.frombuffer(b_bytes, np.uint8)
            b_img = cv2.imdecode(b_np, cv2.IMREAD_COLOR)
            if b_img is not None:
                candidate_frames.append((bf.filename, b_img))

        if not candidate_frames:
            raise HTTPException(status_code=400, detail="No valid before image frames uploaded.")

        # 1. Align best frame or single frame to after image reference frame
        if len(candidate_frames) == 1:
            before_fname, before_img = candidate_frames[0]
            before_aligned, H, metrics = aligner.align_pair(before_img, after_img)
            metrics["selected_frame_name"] = before_fname
            metrics["selected_frame_index"] = 0
        else:
            best_idx, best_fname, before_aligned, H, metrics = aligner.find_best_matching_frame(candidate_frames, after_img)

        h_after, w_after = after_img.shape[:2]

        # 2. Parse annotations if uploaded
        annotations = []
        if ann_file is not None:
            ann_bytes = await ann_file.read()
            filename = ann_file.filename.lower()
            if filename.endswith(".json"):
                try:
                    ann_text = ann_bytes.decode("utf-8")
                    json_data = json.loads(ann_text)
                    if "predictions" in json_data or "annotations" in json_data:
                        # Roboflow native JSON
                        tmp_json_path = os.path.join(TEMP_DIR, "uploaded_rf.json")
                        with open(tmp_json_path, "w", encoding="utf-8") as f:
                            f.write(ann_text)
                        annotations = parser.parse_roboflow_json(tmp_json_path)
                    else:
                        # COCO JSON
                        tmp_coco_path = os.path.join(TEMP_DIR, "uploaded_coco.json")
                        with open(tmp_coco_path, "w", encoding="utf-8") as f:
                            f.write(ann_text)
                        _, annotations = parser.parse_coco_json(tmp_coco_path, after_file.filename)
                except Exception as e:
                    print(f"JSON Parse warning: {e}")
            elif filename.endswith(".txt"):
                # YOLO TXT
                tmp_txt_path = os.path.join(TEMP_DIR, "uploaded_yolo.txt")
                with open(tmp_txt_path, "wb") as f:
                    f.write(ann_bytes)
                annotations = parser.parse_yolo_txt(tmp_txt_path, w_after, h_after)

        # 3. Optional SAM / CPU Refinement
        if run_sam_refine and annotations:
            annotations = sam_integrator.segment_before_image_local(before_aligned, annotations)

        # 4. Render Semantic Mask (0..5)
        semantic_mask = MaskTransformer.render_semantic_mask(annotations, h_after, w_after)

        # 5. Create color visualizations
        vis_mask = np.zeros((h_after, w_after, 3), dtype=np.uint8)
        for cid, color in COLOR_PALETTE_BGR.items():
            vis_mask[semantic_mask == cid] = color

        overlay_after = cv2.addWeighted(after_img, 0.65, vis_mask, 0.35, 0)
        overlay_before = cv2.addWeighted(before_aligned, 0.65, vis_mask, 0.35, 0)

        match_vis = metrics.get("match_vis_bgr")
        if match_vis is None:
            match_vis = np.hstack([before_img, cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))])

        # Scaled 1-channel mask for human viewing (0, 50, 100, 150, 200, 250)
        scaled_mask = (semantic_mask.astype(np.uint16) * 50).clip(0, 255).astype(np.uint8)

        # Prepare base64 images
        return {
            "status": "success",
            "metrics": {
                "inliers_count": metrics.get("inliers_count", 0),
                "good_matches_count": metrics.get("good_matches_count", 0),
                "inlier_ratio": round(metrics.get("inlier_ratio", 0.0), 3),
                "annotations_count": len(annotations),
                "selected_frame_name": metrics.get("selected_frame_name", "before_img"),
                "selected_frame_index": metrics.get("selected_frame_index", 0),
                "total_frames_evaluated": len(candidate_frames),
                "ecc_refined": metrics.get("ecc_refined", False),
                "after_width": w_after,
                "after_height": h_after
            },

            "images": {
                "before_aligned": f"data:image/jpeg;base64,{mat_to_base64(before_aligned)}",
                "after_reference": f"data:image/jpeg;base64,{mat_to_base64(after_img)}",
                "matches_overlay": f"data:image/jpeg;base64,{mat_to_base64(match_vis)}",
                "semantic_overlay_before": f"data:image/jpeg;base64,{mat_to_base64(overlay_before)}",
                "semantic_overlay_after": f"data:image/jpeg;base64,{mat_to_base64(overlay_after)}",
                "semantic_rgb_mask": f"data:image/png;base64,{mat_to_base64(vis_mask, '.png')}",
                "semantic_scaled_mask": f"data:image/png;base64,{mat_to_base64(scaled_mask, '.png')}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-roboflow-sam")
async def export_roboflow_sam_endpoint(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    ann_file: UploadFile = File(...)
):
    try:
        before_bytes = await before_file.read()
        after_bytes = await after_file.read()
        ann_bytes = await ann_file.read()

        before_img = cv2.imdecode(np.frombuffer(before_bytes, np.uint8), cv2.IMREAD_COLOR)
        after_img = cv2.imdecode(np.frombuffer(after_bytes, np.uint8), cv2.IMREAD_COLOR)

        before_aligned, H, _ = aligner.align_pair(before_img, after_img)
        h_after, w_after = after_img.shape[:2]

        tmp_json = os.path.join(TEMP_DIR, "export_tmp.json")
        with open(tmp_json, "wb") as f:
            f.write(ann_bytes)
        
        _, anns = parser.parse_coco_json(tmp_json, after_file.filename)
        if not anns:
            anns = parser.parse_roboflow_json(tmp_json)

        pair_name = os.path.splitext(after_file.filename)[0]
        image_pairs = [(pair_name, before_aligned, after_img, anns)]

        zip_path = sam_integrator.create_roboflow_export_package(TEMP_DIR, image_pairs)
        return FileResponse(zip_path, filename="roboflow_sam_export.zip", media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
