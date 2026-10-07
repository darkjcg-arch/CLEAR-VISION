"""ONNX inference helper for the ClearVision eye classifier."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from pipeline_utils import preprocess_roi_to_tensor


class EyeClassifierONNX:
    """Run the exported MobileNet classifier on BGR OpenCV images."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path,
        mapping_path: str | Path | None = None,
    ) -> None:
        resolved_model_path = self._resolve_project_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {resolved_model_path}")

        if mapping_path is None:
            resolved_mapping_path = resolved_model_path.parent / "class_mapping.json"
        else:
            resolved_mapping_path = self._resolve_project_path(mapping_path)
        self.class_names = self._load_class_names(resolved_mapping_path)

        available_providers = ort.get_available_providers()
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available_providers
        ]
        if not providers:
            raise RuntimeError(
                "ONNX Runtime has no usable execution providers. "
                f"Available providers: {available_providers}"
            )

        self.session = ort.InferenceSession(str(resolved_model_path), providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224]:
            raise RuntimeError(
                "Unexpected classifier input contract; expected [1, 3, 224, 224]."
            )
        if len(outputs) != 1 or outputs[0].shape[-1] != len(self.class_names):
            raise RuntimeError(
                "Classifier output count does not match class_mapping.json."
            )
        self.input_name = inputs[0].name

    @staticmethod
    def _resolve_project_path(path: str | Path) -> Path:
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).resolve().parent.parent / resolved_path
        return resolved_path

    @staticmethod
    def _load_class_names(mapping_path: Path) -> list[str]:
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Class mapping not found: {mapping_path}")
        try:
            payload: dict[str, Any] = json.loads(mapping_path.read_text(encoding="utf-8"))
            index_to_class = payload["idx_to_class"]
            class_names = [index_to_class[str(index)] for index in range(len(index_to_class))]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid class mapping file: {mapping_path}") from error
        if not class_names or any(not isinstance(name, str) for name in class_names):
            raise RuntimeError(f"Class mapping contains no valid labels: {mapping_path}")
        return class_names

    def predict(self, eye_crop: np.ndarray) -> tuple[str, float]:
        """Return the predicted label and confidence percentage for a BGR crop."""
        if eye_crop is None or eye_crop.size == 0:
            raise ValueError("The eye crop is empty")
        if eye_crop.ndim != 3 or eye_crop.shape[2] != 3:
            raise ValueError("The eye crop must be a BGR image with three channels")

        input_tensor = preprocess_roi_to_tensor(eye_crop)

        output = self.session.run(None, {self.input_name: input_tensor})[0]
        logits = np.asarray(output[0], dtype=np.float32)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        class_index = int(np.argmax(probabilities))
        return self.class_names[class_index], float(probabilities[class_index] * 100.0)