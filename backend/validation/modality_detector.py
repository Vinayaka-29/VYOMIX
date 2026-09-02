"""
Modality Detector for SatQuery AI
Classifies satellite imagery into OPTICAL, MULTISPECTRAL, SAR, or UNKNOWN
using band count and pixel statistical distribution (speckle noise / coefficient of variation),
with support for explicit user overrides.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def detect_modality(
    file_path: str, 
    user_override: Optional[str] = None,
    slot_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detects sensor modality from raster properties and statistical cues.
    Returns:
      {
        "modality": "OPTICAL" | "SAR" | "MULTISPECTRAL" | "UNKNOWN",
        "confidence": float,
        "is_override": bool,
        "rationale": str,
        "metrics": dict
      }
    """
    # 1. Check user override first
    if user_override and user_override.strip():
        override_clean = user_override.strip().upper()
        if override_clean in ("OPTICAL", "SAR", "MULTISPECTRAL"):
            return {
                "modality": override_clean,
                "confidence": 1.0,
                "is_override": True,
                "rationale": f"User explicitly designated modality as {override_clean}.",
                "metrics": {},
            }

    # 2. Extract image array sample to evaluate statistics
    band_count = 1
    sample_data = None

    if HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                band_count = src.count
                # Read a downsampled window for fast statistical analysis
                sample_data = src.read(out_shape=(src.count, min(256, src.height), min(256, src.width)))
        except Exception:
            pass

    if sample_data is None:
        try:
            with Image.open(file_path) as img:
                arr = np.array(img)
                if arr.ndim == 2:
                    band_count = 1
                    sample_data = arr[np.newaxis, :, :]
                elif arr.ndim == 3:
                    # (H, W, C) -> (C, H, W)
                    band_count = arr.shape[2]
                    sample_data = np.transpose(arr, (2, 0, 1))
        except Exception:
            pass

    if sample_data is None or sample_data.size == 0:
        # Fallback to slot hint if file couldn't be read
        if slot_hint and "sar" in slot_hint.lower():
            return {
                "modality": "SAR",
                "confidence": 0.6,
                "is_override": False,
                "rationale": "Inferred from SAR upload slot designation.",
                "metrics": {},
            }
        return {
            "modality": "OPTICAL",
            "confidence": 0.5,
            "is_override": False,
            "rationale": "Defaulted to OPTICAL due to unreadable raster array.",
            "metrics": {},
        }

    # 3. Statistical Analysis for Modality Discrimination
    # Filter non-zero and finite values
    valid_mask = np.isfinite(sample_data) & (sample_data > 0)
    if np.any(valid_mask):
        flat_vals = sample_data[valid_mask].astype(np.float64)
    else:
        flat_vals = sample_data.flatten().astype(np.float64)

    mean_val = float(np.mean(flat_vals)) if len(flat_vals) > 0 else 1.0
    std_val = float(np.std(flat_vals)) if len(flat_vals) > 0 else 0.0
    cv = (std_val / mean_val) if mean_val > 0 else 0.0  # Coefficient of Variation

    # High-intensity specular outlier ratio
    p99 = float(np.percentile(flat_vals, 99)) if len(flat_vals) > 0 else 0.0
    p50 = float(np.percentile(flat_vals, 50)) if len(flat_vals) > 0 else 1.0
    dynamic_range_ratio = (p99 / p50) if p50 > 0 else 1.0

    metrics = {
        "band_count": band_count,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "coefficient_of_variation": round(cv, 3),
        "dynamic_range_ratio": round(dynamic_range_ratio, 2),
    }

    # 4. Decision Heuristic
    # Slot designation hint gives prior bias
    slot_is_sar = slot_hint and "sar" in slot_hint.lower()

    if band_count > 4:
        return {
            "modality": "MULTISPECTRAL",
            "confidence": 0.95,
            "is_override": False,
            "rationale": f"High band count ({band_count} spectral bands) indicates multispectral/hyperspectral imagery.",
            "metrics": metrics,
        }

    if band_count in (3, 4) and not slot_is_sar:
        return {
            "modality": "OPTICAL",
            "confidence": 0.92,
            "is_override": False,
            "rationale": f"Standard {band_count}-band RGB/VNIR visual color distribution.",
            "metrics": metrics,
        }

    # Single or dual band analysis: distinguish grayscale Optical vs SAR speckle
    if band_count in (1, 2):
        # SAR typically exhibits significant speckle noise (high CV) and strong corner reflector spikes
        if slot_is_sar or (cv > 0.45 and dynamic_range_ratio > 3.0):
            conf = 0.94 if slot_is_sar else 0.82
            return {
                "modality": "SAR",
                "confidence": conf,
                "is_override": False,
                "rationale": f"Single/dual-band raster with high speckle variance (CV={round(cv, 2)}) and specular reflectivity characteristic of microwave radar.",
                "metrics": metrics,
            }
        else:
            return {
                "modality": "OPTICAL",
                "confidence": 0.78,
                "is_override": False,
                "rationale": f"Panchromatic single-band optical image with smooth radiometric gradients.",
                "metrics": metrics,
            }

    return {
        "modality": "UNKNOWN",
        "confidence": 0.4,
        "is_override": False,
        "rationale": "Ambiguous radiometric distribution.",
        "metrics": metrics,
    }
