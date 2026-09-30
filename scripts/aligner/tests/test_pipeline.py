import os
import shutil
import unittest
import numpy as np
import cv2
from core.alignment import ImageAligner
from core.roboflow_parser import RoboflowParser, InstanceAnnotation
from core.mask_transformer import MaskTransformer
from core.sam_integrator import SAMIntegrator
from core.semantic_exporter import SemanticExporter

class TestBuildingDamagePipeline(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "tmp_test_output")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create synthetic 200x200 before and after images with known shift
        self.w, self.h = 200, 200
        self.after_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Draw synthetic buildings on after_img
        cv2.rectangle(self.after_img, (30, 30), (80, 80), (200, 200, 200), -1)
        cv2.rectangle(self.after_img, (110, 40), (170, 120), (150, 150, 150), -1)
        
        # Before img is after_img shifted by dx=10, dy=15
        M_shift = np.float32([[1, 0, 10], [0, 1, 15]])
        self.before_img = cv2.warpAffine(self.after_img, M_shift, (self.w, self.h))

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_alignment(self):
        aligner = ImageAligner(nfeatures=2000, ratio_thresh=0.85, min_matches=5)
        before_aligned, H, metrics = aligner.align_pair(self.before_img, self.after_img)
        
        self.assertEqual(before_aligned.shape, self.after_img.shape)
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (3, 3))

    def test_roboflow_parser_and_transformer(self):
        parser = RoboflowParser()
        coco_json_path = os.path.join(self.test_dir, "sample_coco.json")
        
        anns = [
            InstanceAnnotation(
                class_id=1,
                class_name="building_intact",
                polygon=[(30, 30), (80, 30), (80, 80), (30, 80)],
                bbox=(30, 30, 50, 50)
            ),
            InstanceAnnotation(
                class_id=4,
                class_name="building_destroyed",
                polygon=[(110, 40), (170, 40), (170, 120), (110, 120)],
                bbox=(110, 40, 60, 80)
            )
        ]
        
        parser.export_coco_json("test_after.jpg", self.w, self.h, anns, coco_json_path)
        self.assertTrue(os.path.exists(coco_json_path))
        
        _, parsed_anns = parser.parse_coco_json(coco_json_path, "test_after.jpg")
        self.assertEqual(len(parsed_anns), 2)
        
        # Test semantic mask rendering
        sem_mask = MaskTransformer.render_semantic_mask(parsed_anns, self.h, self.w)
        self.assertEqual(sem_mask.shape, (self.h, self.w))
        self.assertEqual(sem_mask[50, 50], 1)  # Class 1 Intact
        self.assertEqual(sem_mask[60, 140], 4) # Class 4 Destroyed
        self.assertEqual(sem_mask[0, 0], 0)     # Class 0 Background

    def test_exporter_and_sam_export(self):
        exporter = SemanticExporter(self.test_dir)
        anns = [
            InstanceAnnotation(
                class_id=2,
                class_name="building_minor_damage",
                polygon=[(30, 30), (80, 30), (80, 80), (30, 80)],
                bbox=(30, 30, 50, 50)
            )
        ]
        
        res = exporter.export_sample("sample1", self.before_img, self.after_img, anns)
        self.assertTrue(os.path.exists(res["mask_path"]))
        self.assertTrue(os.path.exists(res["npy_path"]))
        
        loaded_npy = np.load(res["npy_path"])
        self.assertEqual(loaded_npy.shape, (2, self.h, self.w))
        self.assertEqual(loaded_npy[1, 50, 50], 2) # Post-disaster channel contains Class 2

        
        # Test Roboflow SAM ZIP Export
        sam_integrator = SAMIntegrator()
        image_pairs = [("sample1", self.before_img, self.after_img, anns)]
        zip_path = sam_integrator.create_roboflow_export_package(self.test_dir, image_pairs)
        self.assertTrue(os.path.exists(zip_path))

    def test_sequence_matching(self):
        aligner = ImageAligner(nfeatures=2000, ratio_thresh=0.85, min_matches=5)
        
        # Create textured image for strong SIFT keypoints
        np.random.seed(42)
        textured_after = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        M_shift = np.float32([[1, 0, 10], [0, 1, 15]])
        textured_before = cv2.warpAffine(textured_after, M_shift, (200, 200))
        
        # Create dummy non-matching candidate frames
        blank1 = np.zeros((200, 200, 3), dtype=np.uint8)
        blank2 = np.ones((200, 200, 3), dtype=np.uint8) * 50
        blank3 = np.ones((200, 200, 3), dtype=np.uint8) * 100
        
        candidates = [
            ("frame_001.png", blank1),
            ("frame_002.png", blank2),
            ("frame_003.png", textured_before),
            ("frame_004.png", blank3)
        ]
        
        best_idx, best_name, best_aligned, H, metrics = aligner.find_best_matching_frame(candidates, textured_after)
        self.assertEqual(best_idx, 2)
        self.assertEqual(best_name, "frame_003.png")
        self.assertIsNotNone(best_aligned)

    def test_spatial_mask_pruning(self):
        # Create 2 overlapping masks and 1 non-overlapping outlier mask
        ann_overlap_before = InstanceAnnotation(
            class_id=1, class_name="building_intact",
            polygon=[(10, 10), (50, 10), (50, 50), (10, 50)],
            bbox=(10, 10, 40, 40)
        )
        ann_outlier_before = InstanceAnnotation(
            class_id=1, class_name="building_intact",
            polygon=[(150, 150), (190, 150), (190, 190), (150, 190)],
            bbox=(150, 150, 40, 40)
        )
        ann_overlap_after = InstanceAnnotation(
            class_id=2, class_name="building_minor_damage",
            polygon=[(15, 15), (55, 15), (55, 55), (15, 55)],
            bbox=(15, 15, 40, 40)
        )

        before_anns = [ann_overlap_before, ann_outlier_before]
        after_anns = [ann_overlap_after]

        pruned_before, pruned_after, stats = MaskTransformer.prune_spatial_mask_pairs(
            before_anns, after_anns, 200, 200, min_iou=0.15, min_overlap=0.20
        )

        # Overlapping building should be kept; non-overlapping outlier should be pruned
        self.assertEqual(len(pruned_before), 1)
        self.assertEqual(len(pruned_after), 1)
        self.assertEqual(stats["before_pruned"], 1)
        self.assertEqual(stats["after_pruned"], 0)


if __name__ == "__main__":
    unittest.main()

