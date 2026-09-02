"""
Bi-Temporal Change Detection Model for SatQuery AI (Phase 6)
Performs classical computer-vision differencing, morphological noise reduction,
and spatial sector analysis on co-registered before/after satellite rasters.
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from validation.metadata_extractor import extract_metadata
from validation.registration_checker import check_registration

logger = logging.getLogger("satquery.change_detection")


def compute_change_map(
    before_path: str, 
    after_path: str,
    output_dir: Optional[str] = None,
    threshold_val: int = 35
) -> Dict[str, Any]:
    """
    Computes binary change mask and summary statistics between two temporal rasters.
    Returns:
      {
        "change_detected": bool,
        "percentage_changed": float,
        "location_summary": str,
        "mask_image_path": str,
        "mask_overlay_path": str,
        "co_registration": dict,
        "latency_ms": float
      }
    """
    start_time = time.time()
    p_before = Path(before_path)
    p_after = Path(after_path)

    if not p_before.exists() or not p_after.exists():
        raise FileNotFoundError(f"Missing one or both input files: {before_path}, {after_path}")

    # 1. Verify spatial co-registration
    meta_before = extract_metadata(before_path)
    meta_after = extract_metadata(after_path)
    reg_check = check_registration(meta_before, meta_after, overlap_threshold=70.0)

    # 2. Read images using OpenCV
    img_before = cv2.imread(before_path, cv2.IMREAD_COLOR)
    img_after = cv2.imread(after_path, cv2.IMREAD_COLOR)

    if img_before is None or img_after is None:
        # Fallback to PIL
        with Image.open(before_path) as b_img, Image.open(after_path) as a_img:
            img_before = cv2.cvtColor(np.array(b_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            img_after = cv2.cvtColor(np.array(a_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Align dimensions if needed
    h1, w1 = img_before.shape[:2]
    h2, w2 = img_after.shape[:2]
    if (h1, w1) != (h2, w2):
        img_after = cv2.resize(img_after, (w1, h1), interpolation=cv2.INTER_LINEAR)

    # 3. Compute structural differencing
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    # Absolute difference
    diff = cv2.absdiff(gray_before, gray_after)

    # Gaussian blur to filter sensor noise
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Otsu adaptive threshold
    _, thresh = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing & opening to eliminate speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    # 4. Statistical Metrics
    total_pixels = clean_mask.size
    changed_pixels = int(np.count_nonzero(clean_mask))
    pct_changed = round((changed_pixels / total_pixels) * 100.0, 2)
    change_detected = pct_changed > 1.5

    # 5. Spatial quadrant / location description
    h, w = clean_mask.shape
    half_h, half_w = h // 2, w // 2
    nw = np.count_nonzero(clean_mask[:half_h, :half_w])
    ne = np.count_nonzero(clean_mask[:half_h, half_w:])
    sw = np.count_nonzero(clean_mask[half_h:, :half_w])
    se = np.count_nonzero(clean_mask[half_h:, half_w:])

    quadrant_counts = {
        "northwestern": nw,
        "northeastern": ne,
        "southwestern": sw,
        "southeastern": se,
    }
    dominant_sector = max(quadrant_counts, key=quadrant_counts.get) if change_detected else "minimal"

    if change_detected:
        location_desc = f"Changes are concentrated predominantly in the {dominant_sector} sector of the tile (affecting {pct_changed}% of total surface area)."
    else:
        location_desc = "No significant structural land-cover alterations detected between observation epochs (< 1.5% variance)."

    # 6. Generate transparent red change overlay image
    # Red overlay on after image
    overlay = img_after.copy()
    overlay[clean_mask > 0] = [0, 0, 255]  # BGR Red
    blended = cv2.addWeighted(img_after, 0.65, overlay, 0.35, 0)

    # Determine save directory
    out_dir = p_after.parent
    mask_file = out_dir / f"change_mask_{p_after.stem}.png"
    overlay_file = out_dir / f"change_overlay_{p_after.stem}.png"

    cv2.imwrite(str(mask_file), clean_mask)
    cv2.imwrite(str(overlay_file), blended)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"[Change Detection] {pct_changed}% change detected in {latency_ms}ms -> {dominant_sector}")

    return {
        "change_detected": change_detected,
        "percentage_changed": pct_changed,
        "location_summary": location_desc,
        "dominant_sector": dominant_sector,
        "mask_path": str(mask_file),
        "overlay_path": str(overlay_file),
        "co_registration": reg_check,
        "latency_ms": latency_ms,
    }
