"""
Remote-Sensing Image Preprocessor for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Provides a robust, physically grounded image preprocessing pipeline for satellite rasters.
Converts multi-sensor, multi-spectral (Sentinel-2 VNIR/SWIR), single-band SAR (Sentinel-1),
and high-resolution optical rasters into standardized VLM tensors and PIL images.
Zero random-tensor substitutions. Explicit documentation of every transformation.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger("satquery.preprocessor")


class RSImagePreprocessor:
    """
    Standardized remote sensing raster preprocessor.
    Transforms raw GeoTIFF / satellite imagery into calibrated RGB representations
    suitable for Vision-Language Models (GeoChat-7B, BLIP, ViLT, etc.).
    """

    def __init__(self, target_size: Tuple[int, int] = (336, 336)):
        self.target_size = target_size

    def load_and_preprocess(
        self,
        image_path: Union[str, Path],
        return_pil: bool = True
    ) -> Dict[str, Any]:
        """
        Loads and validates a satellite raster, extracts physical metadata,
        applies percentile-based radiometric contrast stretching (2% - 98%),
        and formats the image for VLM consumption.

        Raises:
            FileNotFoundError: If the specified image does not exist.
            ValueError: If the file is corrupt or has invalid dimensions.
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Satellite raster not found at: {image_path}")

        raw_array = None
        nodata_val = None
        crs = None
        transform = None
        band_descriptions = []

        # 1. Attempt GeoTIFF decoding via rasterio
        if HAS_RASTERIO:
            try:
                with rasterio.open(p) as src:
                    raw_array = src.read()  # (C, H, W)
                    nodata_val = src.nodata
                    crs = str(src.crs) if src.crs else None
                    transform = src.transform
                    band_descriptions = [src.descriptions[i] or f"Band_{i+1}" for i in range(src.count)]
            except Exception as e:
                logger.debug(f"Rasterio could not decode {p.name}: {e}. Falling back to PIL.")

        # 2. Fallback to standard image loaders (PNG, JPEG, uncompressed TIFF)
        if raw_array is None:
            try:
                with Image.open(p) as img:
                    img_rgb = img.convert("RGB")
                    np_img = np.array(img_rgb)  # (H, W, 3)
                    raw_array = np.transpose(np_img, (2, 0, 1))  # (3, H, W)
                    band_descriptions = ["Red", "Green", "Blue"]
            except Exception as e:
                raise ValueError(f"Failed to read or decode image file '{p.name}': {str(e)}")

        if raw_array is None or raw_array.size == 0:
            raise ValueError(f"Corrupt or empty image array encountered in: {p.name}")

        channels, orig_h, orig_w = raw_array.shape

        # 3. Clean NoData and NaN/Inf values
        clean_array = raw_array.astype(np.float32)
        if nodata_val is not None:
            clean_array = np.where(clean_array == nodata_val, np.nan, clean_array)
        clean_array = np.nan_to_num(clean_array, nan=0.0, posinf=255.0, neginf=0.0)

        # 4. Modality detection and RGB synthesis
        is_sar = False
        if channels == 1:
            # Single-band SAR or Panchromatic raster
            band = clean_array[0]
            # SAR backscatter typically has high dynamic range (log-normal or speckle)
            mean_val = float(np.mean(band))
            std_val = float(np.std(band))
            cv = (std_val / mean_val) if mean_val > 0 else 0.0
            is_sar = cv > 0.35

            # Apply robust 2% - 98% percentile linear stretch
            p2, p98 = np.percentile(band, (2, 98))
            if p98 > p2:
                stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched = np.clip(band / max(1.0, float(np.max(band))), 0.0, 1.0)

            # Convert single band to 3-channel grayscale for RGB-only VLM backbones
            rgb_float = np.stack([stretched, stretched, stretched], axis=0)  # (3, H, W)

        elif channels == 3:
            # Standard 3-band RGB (e.g. B04, B03, B02 or aerial RGB)
            rgb_float = np.zeros((3, orig_h, orig_w), dtype=np.float32)
            for b in range(3):
                band = clean_array[b]
                p2, p98 = np.percentile(band, (2, 98))
                if p98 > p2:
                    rgb_float[b] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    max_b = float(np.max(band))
                    rgb_float[b] = np.clip(band / (255.0 if max_b <= 255.0 else max_b), 0.0, 1.0)

        elif channels >= 4:
            # Multi-spectral Sentinel-2 or aerial (typically Band 1=Red, 2=Green, 3=Blue, 4=NIR)
            # Map B04 (Red), B03 (Green), B02 (Blue) to RGB channels
            rgb_float = np.zeros((3, orig_h, orig_w), dtype=np.float32)
            for b in range(3):
                band = clean_array[b]
                p2, p98 = np.percentile(band, (2, 98))
                if p98 > p2:
                    rgb_float[b] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    max_b = float(np.max(band))
                    rgb_float[b] = np.clip(band / (255.0 if max_b <= 255.0 else max_b), 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported channel count {channels} in: {p.name}")

        # 5. Convert to PIL Image with aspect-ratio preserving letterboxing
        rgb_uint8 = (np.transpose(rgb_float, (1, 2, 0)) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(rgb_uint8, mode="RGB")

        # Resize preserving aspect ratio with black padding to self.target_size
        target_w, target_h = self.target_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        resized_content = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded_img.paste(resized_content, (pad_x, pad_y))

        return {
            "pil_image": padded_img if return_pil else None,
            "raw_pil_unpadded": pil_img,
            "original_dimensions": {"width": orig_w, "height": orig_h, "channels": channels},
            "padded_dimensions": {"width": target_w, "height": target_h},
            "padding_offset": {"pad_x": pad_x, "pad_y": pad_y, "scale": scale},
            "is_sar": is_sar,
            "band_descriptions": band_descriptions,
            "crs": crs,
            "filename": p.name,
            "image_path": str(p),
        }


# Global preprocessor instance
rs_preprocessor = RSImagePreprocessor(target_size=(336, 336))
