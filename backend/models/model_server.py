"""
Model Server for SatQuery AI
Implements lazy-loading singleton pattern for the remote-sensing adapted VLM backbone.
Ensures models are loaded once at backend startup/warmup rather than per-request.
"""
import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.model_server")


class ModelLoadError(RuntimeError):
    """Raised when the configured model cannot be loaded."""


class RemoteSensingVLMServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RemoteSensingVLMServer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, checkpoint_path: Optional[str] = None) -> None:
        if self._initialized:
            return

        self.model_name = os.getenv("MODEL_NAME", "mbzuai-oryx/GeoChat-7B")
        self.checkpoint = checkpoint_path or os.getenv("MODEL_CHECKPOINT", self.model_name)
        self.adapter_path = os.getenv("MODEL_ADAPTER", "").strip() or None
        self.cache_dir = os.getenv("CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
        self.device = self._select_device(os.getenv("DEVICE", "auto").lower())
        self.version = os.getenv("MODEL_VERSION", "configured")
        self.load_error = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

            self.processor = AutoProcessor.from_pretrained(
                self.checkpoint, cache_dir=self.cache_dir, trust_remote_code=True
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint, cache_dir=self.cache_dir, trust_remote_code=True
                )
            except Exception:
                self.tokenizer = getattr(self.processor, "tokenizer", None)
            load_kwargs: Dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
            }
            if self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **load_kwargs)
            if self.adapter_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.to(self.device)
            self.model.eval()
            self._torch = torch
        except Exception as exc:
            self.load_error = str(exc)
            raise ModelLoadError(
                f"Unable to load VLM '{self.checkpoint}' on {self.device}: {exc}."
            ) from exc

        self._initialized = True
        logger.info("Loaded %s on %s", self.checkpoint, self.device)

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested in {"cpu", "cuda"}:
            if requested == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise ModelLoadError("DEVICE=cuda was requested but CUDA is unavailable")
                except ImportError as exc:
                    raise ModelLoadError("DEVICE=cuda requires PyTorch") from exc
            return requested
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError as exc:
            raise ModelLoadError("PyTorch is required for VLM inference") from exc

    def inspect_raster_channels(self, image_path: str) -> Dict[str, Any]:
        """
        Reads satellite imagery and computes spectral/textural remote sensing indices
        (NDVI, NDWI, brightness, speckle, edge density).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        arr = None
        if HAS_RASTERIO:
            try:
                with rasterio.open(image_path) as src:
                    arr = src.read(out_shape=(src.count, min(512, src.height), min(512, src.width)))
            except Exception:
                pass

        if arr is None:
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    arr = np.array(rgb)
                    # (H, W, 3) -> (3, H, W)
                    arr = np.transpose(arr, (2, 0, 1))
            except Exception as e:
                raise RuntimeError(f"Could not decode image raster: {e}")

        # Normalize to 0.0 - 1.0
        channels = arr.astype(np.float32)
        if np.max(channels) > 1.0:
            channels = channels / (255.0 if np.max(channels) <= 255.0 else float(np.max(channels)))

        c_count = channels.shape[0]
        h, w = channels.shape[1], channels.shape[2]

        if c_count >= 3:
            r, g, b = channels[0], channels[1], channels[2]
            # Approximate spectral indices
            # Green vs Red -> vegetation cue
            veg_index = float(np.mean((g - r) / (g + r + 1e-6)))
            # Blue vs Red -> water reflectance cue
            water_index = float(np.mean((b - r) / (b + r + 1e-6)))
            brightness = float(np.mean((r + g + b) / 3.0))
            is_sar = False
        else:
            # Single band
            band = channels[0]
            veg_index = 0.0
            water_index = 0.0
            brightness = float(np.mean(band))
            std_b = float(np.std(band))
            cv = (std_b / brightness) if brightness > 0 else 0
            is_sar = cv > 0.45

        return {
            "channels": c_count,
            "height": h,
            "width": w,
            "veg_index": veg_index,
            "water_index": water_index,
            "brightness": brightness,
            "is_sar": is_sar,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "model": getattr(self, "model_name", None),
            "checkpoint": getattr(self, "checkpoint", None),
            "adapter": getattr(self, "adapter_path", None),
            "device": getattr(self, "device", None),
            "error": getattr(self, "load_error", None),
        }

    def load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if HAS_RASTERIO:
            try:
                with rasterio.open(path) as source:
                    data = source.read()
                if data.ndim != 3 or data.shape[0] == 0:
                    raise ValueError("Raster has no image bands")
                if data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                data = data[:3].astype(np.float32)
                output = np.empty_like(data, dtype=np.uint8)
                for index, band in enumerate(data):
                    finite = band[np.isfinite(band)]
                    if finite.size == 0:
                        raise ValueError("Raster contains no finite pixels")
                    low, high = np.percentile(finite, [2, 98])
                    if high <= low:
                        high = low + 1.0
                    output[index] = np.clip((band - low) * 255.0 / (high - low), 0, 255)
                return Image.fromarray(np.transpose(output, (1, 2, 0)), mode="RGB")
            except Exception as exc:
                if path.suffix.lower() in {".tif", ".tiff"}:
                    raise RuntimeError(f"Could not decode raster {image_path}: {exc}") from exc
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not decode image {image_path}: {exc}") from exc

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
        self.initialize()
        image = self.load_image(image_path)
        started = time.perf_counter()
        if hasattr(self.model, "chat"):
            result = self.model.chat(self.tokenizer or self.processor, image, prompt)
            text = result[0] if isinstance(result, tuple) else result
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            input_ids = inputs.get("input_ids")
            start = input_ids.shape[-1] if input_ids is not None else 0
            text = self.processor.batch_decode(output[:, start:], skip_special_tokens=True)[0]
        logger.info("VLM inference completed in %.1f ms", (time.perf_counter() - started) * 1000)
        return str(text).strip()


# Global singleton instance
model_server = RemoteSensingVLMServer()
