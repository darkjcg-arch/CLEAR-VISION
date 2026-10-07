"""PyQt6 desktop interface for the CLEAR-VISION screening pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from biometric_engine import BiometricQualityEngine
from camera_pipeline import OcularCapturePipeline
from heatmap_generator import GradCAMGenerator
from predict_onnx import EyeClassifierONNX
from preprocess import FACE_LANDMARKER

MODEL_DIR = PROJECT_DIR / "models" / "ood_aware2"
MODEL_PATH = MODEL_DIR / "clear_vision_ood_aware.onnx"
CHECKPOINT_PATH = MODEL_DIR / "clear_vision_ood_aware.pth"
MINIMUM_CONFIDENCE = 70.0
FAIL_SAFE_MESSAGE = "Inconclusive: Professional Referral Required"


class CameraViewport(QWidget):
    """Aspect-preserving camera canvas with a thin cyan corner reticle."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()
        self._frame_size = (1, 1)
        self._bbox: tuple[int, int, int, int] | None = None

    def set_frame(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int] | None) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        image = QImage(frame_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._frame_size = (width, height)
        self._bbox = bbox
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080B11"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#687386"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Camera preview will appear here")
            return

        target = self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        left = (self.width() - target.width()) / 2.0
        top = (self.height() - target.height()) / 2.0
        painter.drawPixmap(int(left), int(top), target)
        if self._bbox is None:
            return

        frame_width, frame_height = self._frame_size
        x, y, width, height = self._bbox
        scale_x = target.width() / frame_width
        scale_y = target.height() / frame_height
        rect = QRectF(left + x * scale_x, top + y * scale_y, width * scale_x, height * scale_y)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#58E6FF"), 2.0))
        bracket = min(28.0, rect.width() * 0.22, rect.height() * 0.22)
        lines = (
            (rect.topLeft(), QPointF(rect.left() + bracket, rect.top())),
            (rect.topLeft(), QPointF(rect.left(), rect.top() + bracket)),
            (rect.topRight(), QPointF(rect.right() - bracket, rect.top())),
            (rect.topRight(), QPointF(rect.right(), rect.top() + bracket)),
            (rect.bottomLeft(), QPointF(rect.left() + bracket, rect.bottom())),
            (rect.bottomLeft(), QPointF(rect.left(), rect.bottom() - bracket)),
            (rect.bottomRight(), QPointF(rect.right() - bracket, rect.bottom())),
            (rect.bottomRight(), QPointF(rect.right(), rect.bottom() - bracket)),
        )
        for start, end in lines:
            painter.drawLine(start, end)


class ImagePanel(QLabel):
    """Aspect-preserving image container for the processed eye result."""

    def __init__(self) -> None:
        super().__init__("Awaiting capture")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setObjectName("imagePanel")
        self._pixmap = QPixmap()

    def set_bgr_image(self, image_bgr: np.ndarray) -> None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        image = QImage(image_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class ClearVisionWindow(QMainWindow):
    """Apple-inspired diagnostic workstation backed by the live ocular pipeline."""

    CLASS_COLORS = {"healthy": "#38D996", "pterygium": "#F4B860", "cataracts": "#F27D7D"}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CLEAR-VISION | Anterior Ocular Screening System")
        self.resize(1420, 900)
        self.setMinimumSize(1120, 720)
        self.pipeline = OcularCapturePipeline()
        self.quality_engine = BiometricQualityEngine(blur_threshold=150.0)
        self.classifier: EyeClassifierONNX | None = None
        self.gradcam: GradCAMGenerator | None = None
        self.frame_failures = 0
        self.stability_started: float | None = None
        self.last_crop: np.ndarray | None = None
        self.last_processed: np.ndarray | None = None
        self.last_heatmap: np.ndarray | None = None
        self.last_hotspot: tuple[int, int, float, float] | None = None
        self.last_label = ""
        self.last_confidence = 0.0
        self.last_telemetry: dict[str, Any] = {}
        self.show_heatmap = False
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._update_frame)
        self._build_ui()
        self._load_classifier()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        header = QHBoxLayout()
        brand = QLabel("CLEAR-VISION")
        brand.setObjectName("brand")
        subtitle = QLabel("Anterior Ocular Screening System")
        subtitle.setObjectName("subtitle")
        header.addWidget(brand)
        header.addWidget(subtitle)
        header.addStretch()
        self.status_pill = QLabel("●  SYSTEM READY")
        self.status_pill.setObjectName("statusReady")
        header.addWidget(self.status_pill)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(12)
        left_title = QLabel("LIVE CAMERA VIEWPORT")
        left_title.setObjectName("eyebrow")
        left.addWidget(left_title)
        self.viewport = CameraViewport()
        left.addWidget(self.viewport, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        left.addWidget(self.progress)
        self.guidance = QLabel("Position one eye inside the tracking area")
        self.guidance.setObjectName("muted")
        left.addWidget(self.guidance)
        body.addLayout(left, 7)

        right = QVBoxLayout()
        right.setSpacing(14)
        right.addWidget(self._build_telemetry_card())
        right.addWidget(self._build_biometric_card())
        right.addWidget(self._build_result_card(), 1)
        body.addLayout(right, 4)
        root.addLayout(body, 1)

        self.start_button = QPushButton("START NEW SCAN")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._toggle_scan)
        root.addWidget(self.start_button)

    def _build_telemetry_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.addWidget(self._section_title("QUALITY GATEKEEPER"))
        self.blur_value = self._metric("Laplacian blur variance", "--")
        self.glare_value = self._metric("Specular glare", "--")
        self.center_value = self._metric("Pupil centering", "--")
        for row in (self.blur_value, self.glare_value, self.center_value):
            layout.addLayout(row)
        self.gate_label = QLabel("QUALITY GATE  --")
        self.gate_label.setObjectName("gateIdle")
        layout.addWidget(self.gate_label)
        return card

    def _build_result_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.addWidget(self._section_title("DIAGNOSTIC RESULT"))
        toggle_row = QHBoxLayout()
        self.rgb_button = QPushButton("SHOW HIGH-RES RGB")
        self.heatmap_button = QPushButton("SHOW GRAD-CAM")
        for button in (self.rgb_button, self.heatmap_button):
            button.setCheckable(True)
            button.setObjectName("segmentButton")
            toggle_row.addWidget(button)
        self.rgb_button.setChecked(True)
        self.rgb_button.clicked.connect(lambda: self._set_display(False))
        self.heatmap_button.clicked.connect(lambda: self._set_display(True))
        layout.addLayout(toggle_row)
        self.result_image = ImagePanel()
        layout.addWidget(self.result_image, 1)
        self.diagnosis = QLabel("Awaiting diagnostic capture")
        self.diagnosis.setObjectName("diagnosis")
        self.confidence = QLabel("Confidence --")
        self.confidence.setObjectName("muted")
        layout.addWidget(self.diagnosis)
        layout.addWidget(self.confidence)
        return card

    def _build_biometric_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.addWidget(self._section_title("BIOMETRIC ANALYTICAL METRICS"))
        self.opacity_value = self._metric("Corneal opacity", "--")
        self.pterygium_value = self._metric("Pterygium invasion", "--")
        self.circularity_value = self._metric("Pupil circularity", "--")
        for row in (self.opacity_value, self.pterygium_value, self.circularity_value):
            layout.addLayout(row)
        return card

    @staticmethod
    def _card() -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        return card

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _metric(name: str, value: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        label = QLabel(name)
        label.setObjectName("muted")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value_label)
        layout.value_label = value_label  # type: ignore[attr-defined]
        return layout

    def _load_classifier(self) -> None:
        try:
            self.classifier = EyeClassifierONNX(MODEL_PATH)
        except Exception as error:
            self.start_button.setEnabled(False)
            self.start_button.setToolTip("The OOD-aware ONNX model could not be loaded.")
            self._set_status(f"MODEL UNAVAILABLE: {error}", "#F4B860")
            return

        if FACE_LANDMARKER is None:
            self.start_button.setEnabled(False)
            self.start_button.setToolTip(
                "Install MediaPipe and provide models/face_landmarker.task to enable eye tracking."
            )
            self._set_status("EYE TRACKER UNAVAILABLE: check MediaPipe and face_landmarker.task", "#F4B860")

    def _toggle_scan(self) -> None:
        if self.timer.isActive():
            self._stop_scan()
            return
        if self.classifier is None or FACE_LANDMARKER is None:
            return
        try:
            self.pipeline.open()
        except Exception as error:
            self._set_status(f"CAMERA ERROR: {error}", "#F27D7D")
            return
        self.frame_failures = 0
        self.stability_started = None
        self.start_button.setText("STOP SCAN")
        self._set_status("●  LIVE SCAN", "#58E6FF")
        self.timer.start()

    def _stop_scan(self) -> None:
        self.timer.stop()
        self.pipeline.release()
        self.stability_started = None
        self.progress.setValue(0)
        self.start_button.setText("START NEW SCAN")
        self._set_status("●  SYSTEM READY", "#38D996")

    def _update_frame(self) -> None:
        try:
            frame = self.pipeline.read()
            self.frame_failures = 0
        except Exception as error:
            self.frame_failures += 1
            if self.frame_failures >= 3:
                try:
                    self.pipeline.reconnect()
                    self.frame_failures = 0
                    self.guidance.setText("Camera reconnected")
                except Exception:
                    self._set_status(f"CAMERA DISCONNECTED: {error}", "#F27D7D")
            return

        crop, bbox = self.pipeline.extract_eye(frame)
        self.viewport.set_frame(frame, bbox)
        if crop is None or bbox is None:
            self._reset_quality("Face or eye landmarks not detected")
            return
        passed, message, telemetry = self.quality_engine.evaluate(crop)
        self.last_telemetry = telemetry
        self._update_quality(passed, telemetry)
        self.guidance.setText(message)
        if passed:
            if self.stability_started is None:
                self.stability_started = time.monotonic()
            progress = min(1.0, (time.monotonic() - self.stability_started) / 2.0)
            self.progress.setValue(round(progress * 100))
            if progress >= 1.0:
                self._capture_result(crop, telemetry)
        else:
            self.stability_started = None
            self.progress.setValue(0)

    def _capture_result(self, crop: np.ndarray, telemetry: dict[str, Any]) -> None:
        self.timer.stop()
        self.pipeline.release()
        self.start_button.setText("START NEW SCAN")
        self.last_crop = crop.copy()
        try:
            processed = self.pipeline.preprocess_eye(crop, size=512)
            self.last_processed = processed
            self.last_heatmap = None
            self._set_display(False)
            if self.classifier is None:
                raise RuntimeError("ONNX classifier is unavailable")
            label, confidence = self.classifier.predict(crop)
            self.last_label = label
            self.last_confidence = confidence
            referral_reason = self._referral_reason(label, confidence, telemetry)
            if referral_reason:
                self.diagnosis.setText("INCONCLUSIVE")
                self.diagnosis.setToolTip(FAIL_SAFE_MESSAGE)
                self.confidence.setText(f"Referral required | {label.replace('_', ' ').title()} ({confidence:.1f}%)")
                self.diagnosis.setStyleSheet("color: #F4B860;")
                self._set_status("●  REFERRAL REQUIRED", "#F4B860")
                self.guidance.setText(referral_reason)
            else:
                self.diagnosis.setText(label.replace("_", " ").title())
                self.confidence.setText(f"Confidence {confidence:.1f}%")
                self.diagnosis.setStyleSheet(f"color: {self.CLASS_COLORS.get(label, '#E8ECF4')};")
                self._set_status("●  SCAN COMPLETE", "#38D996")
                self.guidance.setText("Capture complete. Review the result or start a new scan.")
        except Exception as error:
            self._set_status(f"PROCESSING ERROR: {error}", "#F27D7D")
            self.guidance.setText("The capture could not be processed.")

    def _set_display(self, heatmap: bool) -> None:
        self.show_heatmap = heatmap
        self.rgb_button.setChecked(not heatmap)
        self.heatmap_button.setChecked(heatmap)
        if self.last_processed is None:
            return
        if heatmap:
            try:
                if self.gradcam is None:
                    self.gradcam = GradCAMGenerator(
                        CHECKPOINT_PATH,
                        class_count=len(self.classifier.class_names) if self.classifier else 4,
                    )
                class_index = self.classifier.class_names.index(self.last_label) if self.classifier else None
                source_crop = self.last_crop if self.last_crop is not None else self.last_processed
                self.last_heatmap, self.last_hotspot = self.gradcam.generate_grid_heatmap(source_crop, class_index)
                self.result_image.set_bgr_image(self.last_heatmap)
                column, row, center_x, center_y = self.last_hotspot
                self.guidance.setText(
                    f"Grad-CAM hotspot: C{column}, R{row} | normalized center ({center_x:.2f}, {center_y:.2f})"
                )
            except Exception as error:
                self.show_heatmap = False
                self.rgb_button.setChecked(True)
                self.heatmap_button.setChecked(False)
                self.guidance.setText(f"Grad-CAM unavailable: {error}")
        else:
            self.result_image.set_bgr_image(self.last_processed)

    def _update_quality(self, passed: bool, telemetry: dict[str, Any]) -> None:
        self.blur_value.value_label.setText(f"{telemetry['blur_variance']:.1f}")
        self.glare_value.value_label.setText(f"{telemetry['glare_percentage']:.2f}%")
        centering = telemetry.get("centering_error")
        self.center_value.value_label.setText("PASS" if centering is not None and telemetry.get("centering_passed") else "FAIL")
        self.gate_label.setText(f"QUALITY GATE  {'PASS' if passed else 'FAIL'}")
        self.gate_label.setObjectName("gatePass" if passed else "gateFail")
        self.gate_label.style().unpolish(self.gate_label)
        self.gate_label.style().polish(self.gate_label)
        self._update_biometrics(telemetry)

    @staticmethod
    def _referral_reason(label: str, confidence: float, telemetry: dict[str, Any]) -> str:
        """Apply the Module 4 confidence and biometric fail-safe policy."""
        if label.lower() == "unknown":
            return "The model classified this image as unknown."
        if confidence < MINIMUM_CONFIDENCE:
            return f"Model confidence is below the {MINIMUM_CONFIDENCE:.0f}% acceptance threshold."

        opacity = telemetry.get("corneal_opacity")
        pterygium = telemetry.get("pterygium_coverage_pct")
        circularity = telemetry.get("pupil_circularity")
        if opacity is None or pterygium is None or circularity is None:
            return "A required biometric metric is unavailable."
        if label == "healthy" and (opacity >= 25.0 or pterygium >= 12.0 or circularity <= 0.60):
            return "Biometric measurements contradict the healthy prediction."
        if label == "cataracts" and (opacity < 25.0 or circularity > 0.95):
            return "Biometric measurements contradict the cataracts prediction."
        if label == "pterygium" and (pterygium < 8.0 or circularity <= 0.45):
            return "Biometric measurements contradict the pterygium prediction."
        return ""

    def _update_biometrics(self, telemetry: dict[str, Any]) -> None:
        opacity = telemetry.get("corneal_opacity")
        coverage = telemetry.get("pterygium_coverage_pct")
        circularity = telemetry.get("pupil_circularity")
        self._set_biometric_value(
            self.opacity_value,
            "--" if opacity is None else f"{opacity:.1f}%  {self._opacity_status(opacity)}",
            "#7E8A9E" if opacity is None else "#F27D7D" if opacity >= 60.0 else "#F4B860" if opacity >= 35.0 else "#38D996",
        )
        self._set_biometric_value(
            self.pterygium_value,
            "--" if coverage is None else f"{coverage:.1f}% area  {self._pterygium_status(coverage)}",
            "#7E8A9E" if coverage is None else "#F27D7D" if coverage >= 3.0 else "#F4B860" if coverage >= 1.0 else "#38D996",
        )
        self._set_biometric_value(
            self.circularity_value,
            "--" if circularity is None else f"{circularity:.2f}  {self._circularity_status(circularity)}",
            "#7E8A9E" if circularity is None else "#F27D7D" if circularity < 0.75 else "#F4B860" if circularity < 0.85 else "#38D996",
        )

    @staticmethod
    def _set_biometric_value(row: QHBoxLayout, text: str, color: str) -> None:
        row.value_label.setText(text)
        row.value_label.setStyleSheet(f"color: {color};")

    @staticmethod
    def _opacity_status(value: float) -> str:
        return "(HIGH DENSITY)" if value >= 60.0 else "(MILD ELEVATION)" if value >= 35.0 else "(NORMAL)"

    @staticmethod
    def _pterygium_status(value: float) -> str:
        return "(INVASION DETECTED)" if value >= 1.0 else "(NORMAL)"

    @staticmethod
    def _circularity_status(value: float) -> str:
        return "(IRREGULAR)" if value < 0.85 else "(REGULAR)"

    def _reset_quality(self, message: str) -> None:
        for row in (self.blur_value, self.glare_value, self.center_value):
            row.value_label.setText("--")
        for row in (self.opacity_value, self.pterygium_value, self.circularity_value):
            self._set_biometric_value(row, "--", "#7E8A9E")
        self.gate_label.setText("QUALITY GATE  --")
        self.gate_label.setObjectName("gateIdle")
        self.gate_label.style().unpolish(self.gate_label)
        self.gate_label.style().polish(self.gate_label)
        self.guidance.setText(message)
        self.progress.setValue(0)
        self.stability_started = None

    def _set_status(self, text: str, color: str) -> None:
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"color: {color};")

    def export_report(self) -> None:
        """Export the current diagnosis and telemetry as a PDF report."""
        if self.last_processed is None or not self.last_label:
            QMessageBox.information(self, "No result", "Complete a scan before exporting a report.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export diagnostic report", "clear_vision_report.pdf", "PDF files (*.pdf)")
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen.canvas import Canvas
            report = Canvas(path, pagesize=letter)
            report.setFont("Helvetica-Bold", 18)
            report.drawString(54, 740, "CLEAR-VISION Diagnostic Report")
            report.setFont("Helvetica", 12)
            report.drawString(54, 710, f"Diagnosis: {self.last_label.title()}")
            report.drawString(54, 690, f"Confidence: {self.last_confidence:.1f}%")
            report.drawString(54, 670, "Quality gate: PASS")
            report.save()
            self.guidance.setText("PDF report exported.")
        except ImportError:
            QMessageBox.warning(self, "PDF unavailable", "Install reportlab in the active environment to export reports.")

    def closeEvent(self, event) -> None:
        self._stop_scan()
        event.accept()


STYLE = """
QWidget#root { background: #0A0D14; color: #E8ECF4; font-family: Inter, SF Pro Display, Segoe UI; font-size: 13px; }
QFrame#card { background: #131823; border: 1px solid #232D3F; border-radius: 16px; }
QLabel#brand { color: #F4F7FC; font-size: 25px; font-weight: 700; letter-spacing: 1px; }
QLabel#subtitle, QLabel#muted { color: #7E8A9E; }
QLabel#statusReady { background: #15261F; border: 1px solid #285841; border-radius: 14px; padding: 7px 13px; color: #38D996; font-weight: 600; }
QLabel#eyebrow, QLabel#sectionTitle { color: #8F9BAD; font-size: 11px; font-weight: 700; letter-spacing: 1.4px; }
QLabel#imagePanel { background: #080B11; border: 1px solid #232D3F; border-radius: 12px; color: #687386; }
QLabel#metricValue { color: #F4F7FC; font-size: 14px; font-weight: 600; }
QLabel#diagnosis { color: #E8ECF4; font-size: 25px; font-weight: 700; padding-top: 4px; }
QLabel#gateIdle, QLabel#gatePass, QLabel#gateFail { border-radius: 8px; padding: 9px; font-weight: 700; margin-top: 8px; }
QLabel#gateIdle { background: #1B2230; color: #8F9BAD; }
QLabel#gatePass { background: #15362A; color: #38D996; }
QLabel#gateFail { background: #3A2025; color: #F27D7D; }
QPushButton#segmentButton { background: #1B2230; border: 1px solid #2B3649; border-radius: 8px; color: #AEB8C8; padding: 9px; font-weight: 600; }
QPushButton#segmentButton:checked { background: #173C49; border-color: #58E6FF; color: #58E6FF; }
QPushButton#primaryButton { background: #167C9B; border: 1px solid #58E6FF; border-radius: 14px; color: white; padding: 14px; font-size: 14px; font-weight: 700; }
QPushButton#primaryButton:hover { background: #2099B8; }
QProgressBar { background: #1B2230; border: none; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #58E6FF; border-radius: 3px; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = ClearVisionWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
