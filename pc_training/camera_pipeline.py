"""High-resolution camera capture and ocular preprocessing for CLEAR-VISION."""

from dataclasses import dataclass
import sys

import cv2
import numpy as np

from preprocess import apply_clahe_and_resize, detect_eye_roi


def _default_camera_backend() -> int:
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


@dataclass(frozen=True)
class CameraConfig:
    """Capture settings chosen to preserve detail before the eye crop is made."""

    device_index: int = 0
    width: int = 1920
    height: int = 1080
    backend: int = _default_camera_backend()


class CameraPipeline:
    """Own camera lifecycle, landmark ROI extraction, and high-resolution output."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.release()
        capture = cv2.VideoCapture(self.config.device_index, self.config.backend)
        if not capture.isOpened():
            capture.release()
            if self.config.backend == cv2.CAP_ANY:
                raise RuntimeError("Could not open the webcam.")
            capture = cv2.VideoCapture(self.config.device_index, cv2.CAP_ANY)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("Could not open the webcam with the selected or automatic backend.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture = capture

    def read(self) -> np.ndarray:
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError("The camera is not open.")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("The webcam returned an invalid frame.")
        return frame

    def reconnect(self) -> None:
        """Reopen the configured device after a transient driver disconnect."""
        self.open()

    @staticmethod
    def extract_eye(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
        return detect_eye_roi(frame_bgr, padding=0.25)

    @staticmethod
    def preprocess_eye(eye_crop: np.ndarray, size: int = 512) -> np.ndarray:
        return apply_clahe_and_resize(eye_crop, target_size=(size, size))

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class OcularCapturePipeline(CameraPipeline):
    """Production-facing name for the high-resolution ocular capture pipeline."""