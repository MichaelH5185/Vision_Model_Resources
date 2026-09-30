import os
import json
import argparse
import glob
import cv2
import numpy as np
from typing import List, Tuple
from core.alignment import ImageAligner
from core.roboflow_parser import RoboflowParser, InstanceAnnotation
from core.mask_transformer import MaskTransformer
from core.sam_integrator import SAMIntegrator
from core.semantic_exporter import SemanticExporter

def process_batch(
    before_dir: str,
    after_dir: str,
    anns_dir: str,
    output_dir: str,
    anns_before_dir: Optional[str] = None,
    export_roboflow_sam: bool = False,
    run_local_sam: bool = False,
    enable_sequence_search: bool = False,
    create_splits: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_spatial_iou: float = 0.15,
    min_overlap: float = 0.20
):



    print(f"=== Starting Building Damage Dataset Processing ===")
    print(f"Before Dir: {before_dir}")
    print(f"After Dir:  {after_dir}")
    print(f"Anns Dir:   {anns_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Splits:     Train: {train_ratio:.2f} | Val: {val_ratio:.2f} | Test: {test_ratio:.2f}")

    aligner = ImageAligner()
    parser = RoboflowParser()
    exporter = SemanticExporter(output_dir)
    sam_integrator = SAMIntegrator()

    # Find matching image pairs
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    after_files = [f for f in os.listdir(after_dir) if f.lower().endswith(valid_exts)]

    export_image_pairs = []

    for after_fname in after_files:
        base_name = os.path.splitext(after_fname)[0]
        after_path = os.path.join(after_dir, after_fname)
        after_img = cv2.imread(after_path)
        if after_img is None:
            print(f"[ERROR] Failed to read after image: {after_fname}")
            continue

        print(f"\nProcessing Pair: {base_name}")

        # 1. Search for direct matching 1-to-1 file in before_dir
        candidate_frames = []
        
        # Clean baseline name variants (e.g., img01, img01_png, img01_before)
        clean_base = base_name.replace('_after', '').replace('after_', '')
        base_variants = {
            base_name,
            clean_base,
            f"{clean_base}_png",
            f"{clean_base}_before",
            f"before_{clean_base}",
            base_name.replace('after', 'before'),
            base_name.rstrip('_png')
        }
        
        exact_before_path = None
        exact_before_fname = None
        
        before_dir_files = [f for f in os.listdir(before_dir) if f.lower().endswith(valid_exts)]
        for f in before_dir_files:
            f_base = os.path.splitext(f)[0]
            if f_base in base_variants or f_base.replace('_before', '').replace('_png', '') == clean_base.replace('_before', '').replace('_png', ''):
                exact_before_path = os.path.join(before_dir, f)
                exact_before_fname = f
                break
                
        if exact_before_path:
            img = cv2.imread(exact_before_path)
            if img is not None:
                candidate_frames.append((exact_before_fname, img))
        elif enable_sequence_search:
            # Fallback to subfolder or multi-frame sequence search ONLY if sequence search is explicitly enabled
            seq_subfolder = os.path.join(before_dir, base_name)
            if os.path.isdir(seq_subfolder):
                frame_files = sorted([f for f in os.listdir(seq_subfolder) if f.lower().endswith(valid_exts)])
                for ff in frame_files:
                    img = cv2.imread(os.path.join(seq_subfolder, ff))
                    if img is not None:
                        candidate_frames.append((ff, img))
            else:
                for ff in before_dir_files:
                    img = cv2.imread(os.path.join(before_dir, ff))
                    if img is not None:
                        candidate_frames.append((ff, img))

        if not candidate_frames:
            print(f"[SKIP] No matching 'before' image found for {after_fname}")
            continue


        # 1. Align before image (or best frame in sequence) to after image reference geometry
        if len(candidate_frames) == 1:
            before_fname, before_img = candidate_frames[0]
            before_aligned, H, metrics = aligner.align_pair(before_img, after_img)
            print(f"  Matched Exact File: {before_fname}")
            print(f"  SIFT Inliers: {metrics['inliers_count']} (Ratio: {metrics['inlier_ratio']:.2f}, Alignment: {metrics.get('alignment_type', 'homography')})")
        else:
            best_idx, best_fname, before_aligned, H, metrics = aligner.find_best_matching_frame(candidate_frames, after_img)
            print(f"  [BEST FRAME MATCH] Selected '{best_fname}' (Frame {best_idx} of {len(candidate_frames)})")
            print(f"  SIFT Inliers: {metrics['inliers_count']} (Ratio: {metrics['inlier_ratio']:.2f}, Alignment: {metrics.get('alignment_type', 'homography')})")

        # 2. Parse annotations for after image and optional separate before image annotations
        h_after, w_after = after_img.shape[:2]
        after_anns = []
        coco_json = os.path.join(anns_dir, f"{base_name}.json")
        roboflow_json = os.path.join(anns_dir, "annotations.json")
        yolo_txt = os.path.join(anns_dir, f"{base_name}.txt")

        if os.path.exists(coco_json):
            _, after_anns = parser.parse_coco_json(coco_json, after_fname)
        elif os.path.exists(roboflow_json):
            after_anns = parser.parse_roboflow_json(roboflow_json)
        elif os.path.exists(yolo_txt):
            after_anns = parser.parse_yolo_txt(yolo_txt, w_after, h_after)
        else:
            json_matches = glob.glob(os.path.join(anns_dir, f"*{base_name}*.json"))
            if json_matches:
                _, after_anns = parser.parse_coco_json(json_matches[0], after_fname)

        print(f"  Parsed {len(after_anns)} post-disaster annotations.")

        # Check for optional separate pre-disaster annotations (COCO JSON, YOLO TXT, or Roboflow JSON)
        h_before_orig, w_before_orig = before_img.shape[:2]
        before_anns = None
        if anns_before_dir and os.path.exists(anns_before_dir):
            b_coco = os.path.join(anns_before_dir, f"{base_name}.json")
            b_coco_before = os.path.join(anns_before_dir, f"{base_name}_before.json")
            b_yolo = os.path.join(anns_before_dir, f"{base_name}.txt")
            b_yolo_before = os.path.join(anns_before_dir, f"{base_name}_before.txt")
            b_rf = os.path.join(anns_before_dir, "annotations.json")
            b_coco_single = os.path.join(anns_before_dir, "_annotations.coco.json")

            if os.path.exists(b_coco):
                _, before_anns = parser.parse_coco_json(b_coco, before_fname)
            elif os.path.exists(b_coco_before):
                _, before_anns = parser.parse_coco_json(b_coco_before, before_fname)
            elif os.path.exists(b_yolo):
                before_anns = parser.parse_yolo_txt(b_yolo, w_before_orig, h_before_orig)
            elif os.path.exists(b_yolo_before):
                before_anns = parser.parse_yolo_txt(b_yolo_before, w_before_orig, h_before_orig)
            elif os.path.exists(b_rf):
                before_anns = parser.parse_roboflow_json(b_rf)
            elif os.path.exists(b_coco_single):
                _, before_anns = parser.parse_coco_json(b_coco_single, before_fname)
            else:
                json_matches = glob.glob(os.path.join(anns_before_dir, f"*{base_name}*.json"))
                txt_matches = glob.glob(os.path.join(anns_before_dir, f"*{base_name}*.txt"))
                if json_matches:
                    _, before_anns = parser.parse_coco_json(json_matches[0], before_fname)
                elif txt_matches:
                    before_anns = parser.parse_yolo_txt(txt_matches[0], w_before_orig, h_before_orig)

            if before_anns is not None:
                print(f"  [BEFORE ANNS] Loaded {len(before_anns)} pre-disaster annotations from anns_before_dir.")
                # Transform pre-disaster annotations through matrix H so masks align perfectly with before_aligned image
                if H is not None and not np.array_equal(H, np.eye(3)):
                    transformed_before_anns = []
                    for ann in before_anns:
                        t_ann = MaskTransformer.transform_annotation(ann, H=H, clip_bounds=(w_after, h_after))
                        transformed_before_anns.append(t_ann)
                    before_anns = transformed_before_anns
            else:
                print(f"  [BEFORE ANNS] No matching pre-disaster annotation file found for '{base_name}' in anns_before_dir.")




        # 3. Optional Local SAM / CPU refinement for before image
        if run_local_sam and after_anns and before_anns is None:
            print("  Running SAM / CPU mask refinement on before image...")
            before_anns = sam_integrator.segment_before_image_local(before_aligned, after_anns)

        export_image_pairs.append((base_name, before_aligned, after_img, after_anns, before_anns))

    # 4. Partition dataset into Train / Val / Test splits
    if create_splits and export_image_pairs:
        sample_names = [p[0] for p in export_image_pairs]
        split_map = exporter.partition_dataset_splits(sample_names, train_ratio, val_ratio, test_ratio)
        
        split_counts = {"train": 0, "val": 0, "test": 0}
        
        for name, before_aligned, after_img, final_after_anns, final_before_anns in export_image_pairs:
            split_name = split_map.get(name, "train")
            split_counts[split_name] += 1
            
            # Subdirectories for split
            split_dir = os.path.join(output_dir, split_name)
            split_dirs = {
                "before": os.path.join(split_dir, "before_aligned"),
                "after": os.path.join(split_dir, "after_reference"),
                "masks_before": os.path.join(split_dir, "semantic_masks_before"),
                "masks_after": os.path.join(split_dir, "semantic_masks_after"),
                "masks_before_rgb": os.path.join(split_dir, "semantic_masks_before_rgb"),
                "masks_after_rgb": os.path.join(split_dir, "semantic_masks_after_rgb"),
                "masks": os.path.join(split_dir, "semantic_masks"),
                "masks_rgb": os.path.join(split_dir, "semantic_masks_rgb"),
                "masks_scaled": os.path.join(split_dir, "semantic_masks_scaled"),
                "numpy": os.path.join(split_dir, "numpy_tensors"),
                "vis": os.path.join(split_dir, "visualizations")
            }
            for sd in split_dirs.values():
                os.makedirs(sd, exist_ok=True)
                
            res = exporter.export_sample(
                name, 
                before_aligned, 
                after_img, 
                final_after_anns, 
                before_annotations=final_before_anns, 
                override_dirs=split_dirs,
                min_spatial_iou=min_spatial_iou,
                min_overlap=min_overlap
            )

        print("\n=== Dataset Partition Summary (Sequence-Grouped) ===")
        print(f"  Train Samples: {split_counts['train']}")
        print(f"  Val Samples:   {split_counts['val']}")
        print(f"  Test Samples:  {split_counts['test']}")
        
        summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": len(export_image_pairs),
                "split_counts": split_counts,
                "split_assignments": split_map
            }, f, indent=2)
            
    # 5. Export Roboflow SAM ZIP package if requested
    if export_roboflow_sam and export_image_pairs:
        zip_file = sam_integrator.create_roboflow_export_package(output_dir, export_image_pairs)
        print(f"\n[ROBOFLOW EXPORT] Package created -> {zip_file}")

    print("\n=== Dataset Processing Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building Damage Image Alignment & Semantic Mask Pipeline")
    parser.add_argument("--before_dir", required=True, help="Directory containing pre-disaster images")
    parser.add_argument("--after_dir", required=True, help="Directory containing post-disaster images")
    parser.add_argument("--anns_dir", required=True, help="Directory containing Roboflow JSON/TXT annotations for after images")
    parser.add_argument("--anns_before_dir", required=False, default=None, help="Optional directory containing pre-disaster Roboflow JSON/TXT annotations (from Roboflow or SAM 3)")
    parser.add_argument("--output_dir", required=True, help="Output folder for Siamese Swin-UNet dataset")
    parser.add_argument("--export_roboflow_sam", action="store_true", help="Generate Roboflow SAM export package")
    parser.add_argument("--run_local_sam", action="store_true", help="Run local SAM / CPU mask refinement")
    parser.add_argument("--enable_sequence_search", action="store_true", help="Enable multi-frame video sequence SIFT search (default: False for fast 1-to-1 matching)")
    
    # Split size arguments
    parser.add_argument("--test_size", "--test_ratio", type=float, default=0.15, help="Test split ratio (e.g. 0.20 for 20%% test set)")
    parser.add_argument("--val_size", "--val_ratio", type=float, default=0.15, help="Validation split ratio (e.g. 0.15 for 15%% val set)")
    parser.add_argument("--train_size", "--train_ratio", type=float, default=None, help="Train split ratio (default: 1.0 - val_size - test_size)")

    # Spatial Overlap Pruning arguments
    parser.add_argument("--min_spatial_iou", type=float, default=0.15, help="Minimum spatial IoU between pre and post masks to keep pair (default: 0.15)")
    parser.add_argument("--min_overlap", type=float, default=0.20, help="Minimum pixel overlap ratio between pre and post masks (default: 0.20)")

    args = parser.parse_args()

    test_ratio = args.test_size
    val_ratio = args.val_size
    if args.train_size is not None:
        train_ratio = args.train_size
    else:
        train_ratio = max(0.0, round(1.0 - val_ratio - test_ratio, 4))

    process_batch(
        args.before_dir,
        args.after_dir,
        args.anns_dir,
        args.output_dir,
        anns_before_dir=args.anns_before_dir,
        export_roboflow_sam=args.export_roboflow_sam,
        run_local_sam=args.run_local_sam,
        enable_sequence_search=args.enable_sequence_search,
        create_splits=True,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_spatial_iou=args.min_spatial_iou,
        min_overlap=args.min_overlap
    )





