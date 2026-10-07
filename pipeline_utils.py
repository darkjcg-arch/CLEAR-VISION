"""Shared CLEAR-VISION image-to-model preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


MODEL_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



def extract_roi(
    frame_bgr: np.ndarray,
    roi_extractor: Callable[[np.ndarray], np.ndarray | None],
) -> np.ndarray:
    """Extract and validate the BGR eye ROI supplied by the runtime detector."""
    roi_bgr = roi_extractor(frame_bgr)
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("ROI extraction returned no image")
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
        raise ValueError("ROI must be a non-empty BGR image with three channels")
    return roi_bgr



def preprocess_roi_to_tensor(roi_bgr: np.ndarray) -> np.ndarray:
    """Convert one extracted BGR ROI into the model tensor [1, 3, 224, 224]."""
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("ROI must contain image data")
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] != 3:
        raise ValueError("ROI must be a BGR image with three channels")

    resized = cv2.resize(roi_bgr, MODEL_SIZE, interpolation=cv2.INTER_AREA)
    lab_image = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced_bgr = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    image_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (image_rgb - MEAN) / STD
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)



def preprocess_image_path(image_path: str | Path) -> np.ndarray:
    """Load a calibration image and apply the exact runtime tensor pipeline."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return preprocess_roi_to_tensor(image_bgr)
