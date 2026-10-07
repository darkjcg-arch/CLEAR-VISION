"""Optional PyTorch Grad-CAM generation for the exported MobileNetV3 model."""

from pathlib import Path

import cv2
import numpy as np

from pipeline_utils import preprocess_roi_to_tensor


class HeatmapGenerator:
    """Generate a Grad-CAM overlay from MobileNetV3 ``features[-1]``."""

    def __init__(self, checkpoint_path: str | Path, class_count: int = 3) -> None:
        try:
            import torch
            from torchvision import models
        except ImportError as error:
            raise RuntimeError("Grad-CAM requires torch and torchvision.") from error
        self.torch = torch
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, class_count)
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self.model.features[-1].register_forward_hook(self._capture_activations)
        self.model.features[-1].register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output) -> None:
        self.activations = output

    def _capture_gradients(self, _module, _inputs, output) -> None:
        self.gradients = output[0]

    def _compute_activation_map(self, image_bgr: np.ndarray, class_index: int | None = None) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot generate a heatmap from an empty image.")
        tensor = self.torch.from_numpy(preprocess_roi_to_tensor(image_bgr)).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        index = int(logits.argmax(1).item()) if class_index is None else class_index
        logits[0, index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((weights * self.activations).sum(dim=1)).squeeze().detach().numpy()
        cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def generate(self, image_bgr: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """Return a standalone colorized Grad-CAM image, without the source photo."""
        cam = self._compute_activation_map(image_bgr, class_index)
        return cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)

    def generate_grid_heatmap(
        self,
        image_bgr: np.ndarray,
        class_index: int | None = None,
        grid_size: int = 8,
    ) -> tuple[np.ndarray, tuple[int, int, float, float]]:
        """Return a heatmap with coordinates and the hottest grid cell.

        The returned location is ``(column, row, center_x, center_y)`` where
        the center coordinates are normalized to the range 0.0 to 1.0.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        cam = self._compute_activation_map(image_bgr, class_index)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)
        height, width = cam.shape
        cell_height = height / grid_size
        cell_width = width / grid_size
        scores = np.zeros((grid_size, grid_size), dtype=np.float32)
        for row in range(grid_size):
            for column in range(grid_size):
                top = int(row * cell_height)
                bottom = int((row + 1) * cell_height)
                left = int(column * cell_width)
                right = int((column + 1) * cell_width)
                scores[row, column] = float(cam[top:bottom, left:right].mean())
        hottest_row, hottest_column = np.unravel_index(int(np.argmax(scores)), scores.shape)
        center_x = (hottest_column + 0.5) / grid_size
        center_y = (hottest_row + 0.5) / grid_size

        for index in range(1, grid_size):
            x = int(index * cell_width)
            y = int(index * cell_height)
            cv2.line(heatmap, (x, 0), (x, height - 1), (235, 245, 255), 1, cv2.LINE_AA)
            cv2.line(heatmap, (0, y), (width - 1, y), (235, 245, 255), 1, cv2.LINE_AA)
        left = int(hottest_column * cell_width)
        top = int(hottest_row * cell_height)
        right = int((hottest_column + 1) * cell_width) - 1
        bottom = int((hottest_row + 1) * cell_height) - 1
        cv2.rectangle(heatmap, (left, top), (right, bottom), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            heatmap,
            f"HOTSPOT C{hottest_column + 1},R{hottest_row + 1}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return heatmap, (hottest_column + 1, hottest_row + 1, center_x, center_y)

    def generate_overlay(self, image_bgr: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """Return the optional source-image overlay representation."""
        heatmap = self.generate(image_bgr, class_index)
        return cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)


class GradCAMGenerator(HeatmapGenerator):
    """Production-facing name for the MobileNetV3 Grad-CAM generator."""