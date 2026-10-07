"""Quality and screening biometrics for high-resolution ocular crops."""

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityTelemetry:
    blur_variance: float
    glare_percentage: float
    centering_error: float | None
    mean_brightness: float
    blur_passed: bool
    glare_passed: bool
    centering_passed: bool
    lighting_passed: bool
    corneal_opacity: float | None
    pterygium_coverage_pct: float | None
    invasion_depth_px: float | None
    pupil_circularity: float | None

    @property
    def passed(self) -> bool:
        return all((self.blur_passed, self.glare_passed, self.centering_passed, self.lighting_passed))

    def as_dict(self) -> dict[str, float | bool | None]:
        values = asdict(self)
        values["blur_score"] = values["blur_variance"]
        values["gate_passed"] = self.passed
        return values


class BiometricEngine:
    """Evaluate quality gates and explainable ocular screening biometrics."""

    def __init__(
        self,
        blur_threshold: float = 150.0,
        glare_threshold: float = 5.0,
        centering_tolerance: float = 0.30,
        low_light_threshold: float = 35.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.glare_threshold = glare_threshold
        self.centering_tolerance = centering_tolerance
        self.low_light_threshold = low_light_threshold

    @staticmethod
    def _validate(image_bgr: np.ndarray) -> None:
        if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("The ocular crop must be a non-empty BGR image.")

    @staticmethod
    def _largest_relevant_contour(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        height, width = mask.shape
        image_center = np.array([width / 2.0, height / 2.0])
        candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(16.0, height * width * 0.002) or area > height * width * 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
            if np.linalg.norm(center - image_center) > max(width, height) * 0.45:
                continue
            candidates.append((area, contour))
        if not candidates:
            return None, mask
        contour = max(candidates, key=lambda item: item[0])[1]
        selected_mask = np.zeros_like(mask)
        cv2.drawContours(selected_mask, [contour], -1, 255, thickness=-1)
        return contour, selected_mask

    @classmethod
    def _segment_pupil(cls, image_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        dark_pixels = cv2.bitwise_and(otsu_mask, adaptive_mask)
        kernel = np.ones((3, 3), np.uint8)
        dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel)
        dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_CLOSE, kernel)
        return cls._largest_relevant_contour(dark_pixels)

    @staticmethod
    def disease_prediction_contradicted(
        predicted_label: str,
        telemetry: dict[str, float | bool | None],
        poi_threshold: float = 25.0,
        pti_threshold: float = 8.0,
    ) -> bool:
        """Return true when disease evidence is absent from both structural indices."""
        if predicted_label.lower() not in {"cataracts", "cataract", "pterygium"}:
            return False
        opacity = telemetry.get("corneal_opacity")
        pterygium = telemetry.get("pterygium_coverage_pct")
        return (
            opacity is not None
            and pterygium is not None
            and float(opacity) < poi_threshold
            and float(pterygium) < pti_threshold
        )

    @staticmethod
    def _segment_limbus(image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        red_low = cv2.inRange(hsv, np.array([0, 35, 25], np.uint8), np.array([18, 255, 255], np.uint8))
        red_high = cv2.inRange(hsv, np.array([160, 35, 25], np.uint8), np.array([180, 255, 255], np.uint8))
        tissue_mask = cv2.bitwise_or(red_low, red_high)
        return cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def calculate_corneal_opacity(roi_crop: np.ndarray, pupil_mask: np.ndarray | None) -> float | None:
        """Estimate pupil/lens cloudiness from masked LAB luminance (0-100%)."""
        if roi_crop is None or roi_crop.size == 0 or pupil_mask is None:
            return None
        if roi_crop.ndim != 3 or roi_crop.shape[2] != 3 or pupil_mask.shape != roi_crop.shape[:2]:
            return None
        pixels = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2LAB)[:, :, 0][pupil_mask > 0]
        if pixels.size < 16:
            return None
        luminance = float(np.percentile(pixels, 50) * 0.7 + np.percentile(pixels, 75) * 0.3)
        return float(np.clip((luminance - 20.0) / 180.0 * 100.0, 0.0, 100.0))

    @staticmethod
    def calculate_pterygium_index(
        roi_crop: np.ndarray, limbus_mask: np.ndarray | None
    ) -> tuple[float | None, float | None]:
        """Return fibrovascular area percentage and maximum horizontal invasion depth."""
        if roi_crop is None or roi_crop.size == 0 or limbus_mask is None:
            return None, None
        if limbus_mask.shape != roi_crop.shape[:2]:
            return None, None
        pixel_count = cv2.countNonZero(limbus_mask)
        if pixel_count < 8:
            return None, None
        height, width = limbus_mask.shape
        coordinates = np.column_stack(np.where(limbus_mask > 0))
        x_coordinates = coordinates[:, 1]
        distance_from_edge = np.minimum(x_coordinates, width - 1 - x_coordinates)
        coverage_pct = float(pixel_count * 100.0 / (height * width))
        return coverage_pct, float(distance_from_edge.max())

    @staticmethod
    def calculate_circularity(contour: np.ndarray | None) -> float | None:
        """Calculate contour circularity, where 1.0 is a perfect circle."""
        if contour is None or len(contour) < 3:
            return None
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0.0 or perimeter <= 0.0:
            return None
        return float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0.0, 1.0))

    def _biometric_metrics(self, image_bgr: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
        pupil_contour, pupil_mask = self._segment_pupil(image_bgr)
        opacity = self.calculate_corneal_opacity(image_bgr, pupil_mask if pupil_contour is not None else None)
        coverage, depth = self.calculate_pterygium_index(image_bgr, self._segment_limbus(image_bgr))
        return opacity, coverage, depth, self.calculate_circularity(pupil_contour)

    def segment_regions(self, image_bgr: np.ndarray) -> dict[str, np.ndarray | None]:
        """Return independently evaluable pupil and fibrovascular ROI masks."""
        self._validate(image_bgr)
        pupil_contour, pupil_mask = self._segment_pupil(image_bgr)
        limbus_mask = self._segment_limbus(image_bgr)
        return {
            "pupil_mask": pupil_mask if pupil_contour is not None else None,
            "pterygium_mask": (limbus_mask > 0).astype(np.uint8),
        }

    def evaluate_validated_indices(self, image_bgr: np.ndarray, pupil_iou: float | None, pterygium_iou: float | None, minimum_iou: float = 0.50) -> dict[str, float | bool | None]:
        """Calculate indices only for ROIs that meet the configured validation IoU."""
        telemetry = self.evaluate(image_bgr)[2]
        pupil_valid = pupil_iou is not None and pupil_iou >= minimum_iou
        pterygium_valid = pterygium_iou is not None and pterygium_iou >= minimum_iou
        return {
            "corneal_opacity": telemetry.get("corneal_opacity") if pupil_valid else None,
            "pterygium_coverage_pct": telemetry.get("pterygium_coverage_pct") if pterygium_valid else None,
            "pupil_circularity": telemetry.get("pupil_circularity") if pupil_valid else None,
            "pupil_iou": pupil_iou,
            "pterygium_iou": pterygium_iou,
            "segmentation_valid": bool(pupil_valid and pterygium_valid),
            "override_safe": bool(pupil_valid and pterygium_valid),
        }

    def evaluate(self, image_bgr: np.ndarray) -> tuple[bool, str, dict[str, float | bool | None]]:
        self._validate(image_bgr)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        glare_mask = cv2.inRange(hsv, np.array([0, 0, 240], np.uint8), np.array([180, 45, 255], np.uint8))
        glare_percentage = float(cv2.countNonZero(glare_mask) * 100.0 / gray.size)
        mean_brightness = float(gray.mean())

        dark_mask = cv2.inRange(gray, 20, 90)
        moments = cv2.moments(dark_mask)
        if moments["m00"] > 0:
            centroid_x = moments["m10"] / moments["m00"] / gray.shape[1]
            centroid_y = moments["m01"] / moments["m00"] / gray.shape[0]
            centering_error = float(np.hypot(centroid_x - 0.5, centroid_y - 0.5))
        else:
            centering_error = None
        centering_passed = centering_error is not None and centering_error <= self.centering_tolerance
        opacity, pterygium_coverage, invasion_depth, pupil_circularity = self._biometric_metrics(image_bgr)
        telemetry = QualityTelemetry(
            blur_variance=blur_variance,
            glare_percentage=glare_percentage,
            centering_error=centering_error,
            mean_brightness=mean_brightness,
            blur_passed=blur_variance >= self.blur_threshold,
            glare_passed=glare_percentage <= self.glare_threshold,
            centering_passed=centering_passed,
            lighting_passed=mean_brightness >= self.low_light_threshold,
            corneal_opacity=opacity,
            pterygium_coverage_pct=pterygium_coverage,
            invasion_depth_px=invasion_depth,
            pupil_circularity=pupil_circularity,
        )
        if not telemetry.lighting_passed:
            message = "Increase illumination - scene is too dark"
        elif not telemetry.blur_passed:
            message = "Hold still - ocular image is blurry"
        elif not telemetry.glare_passed:
            message = "Adjust light angle - excessive corneal glare"
        elif not telemetry.centering_passed:
            message = "Center the pupil inside the tracking box"
        else:
            message = "Quality check passed"
        return telemetry.passed, message, telemetry.as_dict()


def calculate_blur_score(image_bgr: np.ndarray) -> tuple[float, bool]:
    telemetry = BiometricEngine().evaluate(image_bgr)[2]
    return float(telemetry["blur_variance"]), bool(telemetry["blur_passed"])


def calculate_glare_percentage(image_bgr: np.ndarray) -> tuple[float, bool]:
    telemetry = BiometricEngine().evaluate(image_bgr)[2]
    return float(telemetry["glare_percentage"]), bool(telemetry["glare_passed"])


def evaluate_frame_quality(image_bgr: np.ndarray):
    """Backward-compatible function API used by older callers."""
    try:
        return BiometricEngine().evaluate(image_bgr)
    except ValueError:
        return False, "Error: Invalid image frame", {}


class BiometricQualityEngine(BiometricEngine):
    """Production-facing name for the ocular biometric quality gate."""