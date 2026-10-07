"""Image preprocessing utilities for the CLEAR-VISION pipeline."""

import cv2
import numpy as np
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    mp = None

FACE_LANDMARKER = None
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
if mp is not None and MODEL_PATH.exists():
    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        FACE_LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options)
    except Exception:
        FACE_LANDMARKER = None

LEFT_EYE_LANDMARKS = (33, 133, 159, 145)
RIGHT_EYE_LANDMARKS = (362, 263, 386, 374)


def detect_eye_roi(
    image_bgr: np.ndarray,
    padding: float = 0.25,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Find the largest visible eye and return a square, padded crop and box."""
    if (
        image_bgr is None
        or image_bgr.size == 0
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        return None, None
    if FACE_LANDMARKER is None or mp is None:
        return None, None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = FACE_LANDMARKER.detect(media_image)
    if not detection_result.face_landmarks:
        return None, None

    landmarks = detection_result.face_landmarks[0]
    eye_boxes = []
    for eye_landmarks in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
        points = [landmarks[index] for index in eye_landmarks]
        min_x = max(0.0, min(point.x for point in points) - padding * 0.5 * (max(point.x for point in points) - min(point.x for point in points)))
        max_x = min(1.0, max(point.x for point in points) + padding * 0.5 * (max(point.x for point in points) - min(point.x for point in points)))
        min_y = max(0.0, min(point.y for point in points) - padding * 0.5 * (max(point.y for point in points) - min(point.y for point in points)))
        max_y = min(1.0, max(point.y for point in points) + padding * 0.5 * (max(point.y for point in points) - min(point.y for point in points)))
        eye_boxes.append((min_x, min_y, max_x, max_y))

    min_x, min_y, max_x, max_y = max(
        eye_boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
    )
    image_height, image_width = image_bgr.shape[:2]
    raw_left = min_x * image_width
    raw_top = min_y * image_height
    raw_right = max_x * image_width
    raw_bottom = max_y * image_height
    center_x = (raw_left + raw_right) / 2.0
    center_y = (raw_top + raw_bottom) / 2.0
    side = max(raw_right - raw_left, raw_bottom - raw_top) * (1.0 + 2.0 * padding)
    side = max(2.0, side)
    left = max(0, int(round(center_x - side / 2.0)))
    top = max(0, int(round(center_y - side / 2.0)))
    right = min(image_width, left + int(round(side)))
    bottom = min(image_height, top + int(round(side)))
    left = max(0, right - int(round(side)))
    top = max(0, bottom - int(round(side)))

    return image_bgr[top:bottom, left:right].copy(), (left, top, right - left, bottom - top)


def apply_clahe_and_resize(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Denoise luminance, apply CLAHE, and upscale without changing aspect ratio."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must contain image data")

    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    denoised_lightness = cv2.fastNlMeansDenoising(
        lightness, None, h=5, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(denoised_lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return cv2.resize(enhanced_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
