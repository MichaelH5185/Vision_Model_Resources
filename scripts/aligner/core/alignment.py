import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

def is_valid_transform(H: np.ndarray, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> bool:
    """
    Validates a 3x3 transformation matrix H (Homography or Affine) to prevent:
    - Extreme rotations (> 25 degrees)
    - Aspect ratio shear distortion
    - Horizon ray-burst perspective distortion
    - Non-convex warped canvas
    """
    if H is None or H.shape != (3, 3):
        return False
    if np.isnan(H).any() or np.isinf(H).any():
        return False
        
    if abs(H[2, 2]) < 1e-7:
        return False
    H_norm = H / H[2, 2]
    
    # 1. Scale & Rotation check from upper-left 2x2 submatrix
    a, b = H_norm[0, 0], H_norm[0, 1]
    c, d = H_norm[1, 0], H_norm[1, 1]
    
    scale_x = np.sqrt(a * a + c * c)
    scale_y = np.sqrt(b * b + d * d)
    
    # Scale bounds (supports full-frame to sub-region crop scaling down to 0.05x)
    if scale_x < 0.05 or scale_x > 10.0 or scale_y < 0.05 or scale_y > 10.0:
        return False
        
    # Reject aspect ratio shear distortion (relative difference)
    max_scale = max(scale_x, scale_y)
    if max_scale > 0 and abs(scale_x - scale_y) / max_scale > 0.40:
        return False
        
    # Check rotation angle (must be within +/- 25 degrees for pre/post aerial pairs)
    angle_rad = np.arctan2(b, a)
    angle_deg = abs(np.degrees(angle_rad))
    if angle_deg > 25.0 and angle_deg < 335.0:
        return False
        
    # 2. Determinant check
    det = np.linalg.det(H_norm)
    if det <= 0.001 or det > 100.0:
        return False
        
    # 3. Perspective tilt check (prevents division by zero inside canvas)
    if abs(H_norm[2, 0]) > 0.003 or abs(H_norm[2, 1]) > 0.003:
        return False
        
    # 4. Corner projection convexity check
    h_src, w_src = src_shape[:2]
    corners_src = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]).reshape(-1, 1, 2)
    try:
        corners_dst = cv2.perspectiveTransform(corners_src, H_norm)
        pts = corners_dst.reshape(-1, 2)
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return False
    except Exception:
        return False
        
    return True

# Retain is_valid_homography alias for backwards compatibility
is_valid_homography = is_valid_transform

class ImageAligner:
    """
    Aligns a 'before' image to an 'after' image using SIFT feature matching 
    and RANSAC homography. The 'after' image is used as the un-warped reference geometry.
    Supports cropping sub-regions from large drone frames into target reference patches.
    """
    def __init__(self, nfeatures: int = 5000, ratio_thresh: float = 0.75, min_matches: int = 10):
        self.nfeatures = nfeatures
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        
    def align_pair(
        self, 
        before_img: np.ndarray, 
        after_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Aligns before_img to after_img geometry.
        
        Returns:
            - before_aligned: Pre-disaster image cropped and warped to post-disaster coordinate space
            - H: 3x3 Homography Matrix (mapping original before_coords -> after_coords)
            - metrics: Dict with inlier counts, match status, and visual match visualization
        """
        h_after, w_after = after_img.shape[:2]
        h_before, w_before = before_img.shape[:2]
        
        # Scale factor for SIFT feature extraction if before_img is extremely large (e.g. > 2000px)
        max_dim = 2000
        scale_before = 1.0
        if max(h_before, w_before) > max_dim:
            scale_before = max_dim / float(max(h_before, w_before))
            sift_before_img = cv2.resize(before_img, (int(w_before * scale_before), int(h_before * scale_before)), interpolation=cv2.INTER_LINEAR)
        else:
            sift_before_img = before_img

        if len(sift_before_img.shape) == 2:
            before_gray = sift_before_img
        else:
            before_gray = cv2.cvtColor(sift_before_img, cv2.COLOR_BGR2GRAY)
            
        if len(after_img.shape) == 2:
            after_gray = after_img
        else:
            after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
            
        # Detect SIFT keypoints and descriptors
        sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        kp_before, des_before = sift.detectAndCompute(before_gray, None)
        kp_after, des_after = sift.detectAndCompute(after_gray, None)

        metrics = {
            "kp_before_count": len(kp_before) if kp_before else 0,
            "kp_after_count": len(kp_after) if kp_after else 0,
            "good_matches_count": 0,
            "inliers_count": 0,
            "inlier_ratio": 0.0,
            "aligned_successfully": False,
            "match_vis_bgr": None,
            "alignment_type": "none"
        }
        
        if des_before is None or des_after is None or len(kp_before) < 4 or len(kp_after) < 4:
            before_aligned = cv2.resize(before_img, (w_after, h_after), interpolation=cv2.INTER_LINEAR)
            H_identity = np.eye(3, dtype=np.float64)
            metrics["alignment_type"] = "resize_fallback"
            return before_aligned, H_identity, metrics
            
        # Feature Matching using FLANN or BFMatcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des_before, des_after, k=2)
        except Exception:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_before, des_after, k=2)
            
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
                    
        metrics["good_matches_count"] = len(good_matches)
        
        if len(good_matches) < self.min_matches:
            before_aligned = cv2.resize(before_img, (w_after, h_after), interpolation=cv2.INTER_LINEAR)
            H_identity = np.eye(3, dtype=np.float64)
            metrics["alignment_type"] = "resize_fallback"
            return before_aligned, H_identity, metrics
            
        # Extract location of good matches, converting keypoints back to original before_img coordinate space
        pts_before = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_before
        pts_after = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 1. Try Homography using USAC_MAGSAC with strict validity check
        H = None
        mask = None
        alignment_type = "homography"
        
        try:
            H_cand, mask_cand = cv2.findHomography(pts_before, pts_after, cv2.USAC_MAGSAC, 3.0)
            if is_valid_transform(H_cand, before_img.shape, after_img.shape):
                H = H_cand
                mask = mask_cand
                alignment_type = "homography"
        except Exception:
            pass

        # 2. Fallback to Partial Affine (rotation + translation + scale)
        if H is None:
            try:
                M_affine, mask_cand = cv2.estimateAffinePartial2D(pts_before, pts_after, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M_affine is not None:
                    H_cand = np.vstack([M_affine, [0, 0, 1]])
                    if is_valid_transform(H_cand, before_img.shape, after_img.shape):
                        H = H_cand
                        mask = mask_cand
                        alignment_type = "affine_partial"
            except Exception:
                pass

        # 3. Fallback to Full 2D Affine
        if H is None:
            try:
                M_affine, mask_cand = cv2.estimateAffine2D(pts_before, pts_after, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M_affine is not None:
                    H_cand = np.vstack([M_affine, [0, 0, 1]])
                    if is_valid_transform(H_cand, before_img.shape, after_img.shape):
                        H = H_cand
                        mask = mask_cand
                        alignment_type = "affine_full"
            except Exception:
                pass

        if H is None:
            before_aligned = cv2.resize(before_img, (w_after, h_after), interpolation=cv2.INTER_LINEAR)
            H_identity = np.eye(3, dtype=np.float64)
            metrics["alignment_type"] = "resize_fallback"
            return before_aligned, H_identity, metrics


            
        inliers_count = int(np.sum(mask)) if mask is not None else 0
        metrics["inliers_count"] = inliers_count
        metrics["inlier_ratio"] = float(inliers_count / len(good_matches)) if good_matches else 0.0
        metrics["aligned_successfully"] = True
        metrics["alignment_type"] = alignment_type
        
        # Warp before_img to match after_img frame
        before_aligned = cv2.warpPerspective(before_img, H, (w_after, h_after), flags=cv2.INTER_LINEAR)
        
        # ECC (Enhanced Correlation Coefficient) Sub-Pixel Refinement for non-burst alignment
        try:
            warp_mode = cv2.MOTION_AFFINE if alignment_type.startswith("affine") else cv2.MOTION_HOMOGRAPHY
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
            
            before_aligned_gray = cv2.cvtColor(before_aligned, cv2.COLOR_BGR2GRAY)
            H_ecc = np.eye(3, 3, dtype=np.float32)
            
            _, H_ecc = cv2.findTransformECC(after_gray, before_aligned_gray, H_ecc, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
            
            # Combine Homographies
            H_new = np.dot(H_ecc.astype(np.float64), H)
            if is_valid_homography(H_new, before_img.shape, after_img.shape):
                H = H_new
                before_aligned = cv2.warpPerspective(before_img, H, (w_after, h_after), flags=cv2.INTER_LINEAR)
                metrics["ecc_refined"] = True
            else:
                metrics["ecc_refined"] = False
        except Exception:
            metrics["ecc_refined"] = False
        
        # Create visual match image (top 50 matches)
        draw_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
        matches_mask = mask.ravel().tolist() if mask is not None else None
        
        match_vis = cv2.drawMatches(
            before_img, kp_before, 
            after_img, kp_after, 
            draw_matches, None, 
            matchesMask=matches_mask[:len(draw_matches)] if matches_mask else None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        metrics["match_vis_bgr"] = match_vis
        
        return before_aligned, H, metrics



    def find_best_matching_frame(
        self,
        candidate_frames: list,
        after_img: np.ndarray
    ) -> Tuple[int, str, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Evaluates a sequence of candidate 'before' frames against a reference 'after' image.
        Selects the frame with the highest SIFT inlier score.
        
        candidate_frames can be:
          - List of np.ndarray
          - List of tuples: (frame_name_or_id, np.ndarray)
          
        Returns:
          - best_idx: Index of best matching frame
          - best_name: Identifier/filename of best matching frame
          - best_aligned: Warped image of best frame
          - best_H: Homography matrix
          - best_metrics: Alignment metrics dict (includes 'frame_scores' for all candidates)
        """
        if not candidate_frames:
            raise ValueError("candidate_frames list cannot be empty.")
            
        best_score = -1.0
        best_idx = 0
        best_name = "frame_0"
        best_aligned = None
        best_H = None
        best_metrics = None
        
        frame_scores = []
        
        for idx, item in enumerate(candidate_frames):
            if isinstance(item, tuple) and len(item) == 2:
                frame_name, frame_img = item
            else:
                frame_name = f"frame_{idx:04d}"
                frame_img = item
                
            aligned_img, H, metrics = self.align_pair(frame_img, after_img)
            
            inliers = metrics.get("inliers_count", 0)
            ratio = metrics.get("inlier_ratio", 0.0)
            score = float(inliers * (1.0 + ratio))
            
            frame_scores.append({
                "index": idx,
                "name": str(frame_name),
                "inliers": inliers,
                "inlier_ratio": round(ratio, 3),
                "score": round(score, 2)
            })
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_name = str(frame_name)
                best_aligned = aligned_img
                best_H = H
                best_metrics = metrics
                
        if best_metrics is None:
            # Fallback if all scores 0
            frame_img = candidate_frames[0][1] if isinstance(candidate_frames[0], tuple) else candidate_frames[0]
            best_aligned, best_H, best_metrics = self.align_pair(frame_img, after_img)
            
        best_metrics["selected_frame_index"] = best_idx
        best_metrics["selected_frame_name"] = best_name
        best_metrics["best_match_score"] = best_score
        best_metrics["all_frame_scores"] = frame_scores
        
        return best_idx, best_name, best_aligned, best_H, best_metrics

