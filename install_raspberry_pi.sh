#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-pi"
MODEL_DIR="$PROJECT_DIR/models/ood_aware2"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Required project file is missing: $1"
}

[[ "$(uname -m)" == "aarch64" ]] || fail "Use 64-bit Raspberry Pi OS (aarch64); detected $(uname -m)."
command -v apt-get >/dev/null || fail "This installer requires Raspberry Pi OS or another apt-based OS."

python3 - <<'PY'
import platform
import sys

if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python 3.11 or newer is required for the ARM64 ONNX Runtime wheel.")

libc_name, libc_version = platform.libc_ver()
try:
    libc_parts = tuple(int(part) for part in libc_version.split(".")[:2])
except ValueError:
    libc_parts = ()
if libc_name != "glibc" or libc_parts < (2, 28):
    raise SystemExit(
        f"ERROR: glibc 2.28 or newer is required for ARM64 wheels; detected {libc_name} {libc_version}."
    )
PY

require_file "$PROJECT_DIR/pipeline_utils.py"
require_file "$PROJECT_DIR/pc_training/app_gui.py"
require_file "$MODEL_DIR/clear_vision_ood_aware.onnx"
require_file "$MODEL_DIR/clear_vision_ood_aware.onnx.data"
require_file "$MODEL_DIR/class_mapping.json"
require_file "$PROJECT_DIR/models/face_landmarker.task"

run_root() {
    if [[ "$EUID" -eq 0 ]]; then
        "$@"
    else
        command -v sudo >/dev/null || fail "Run as root or install sudo first."
        sudo "$@"
    fi
}

printf 'Installing Raspberry Pi OS packages...\n'
run_root apt-get update
run_root apt-get install -y \
    python3-full \
    python3-pyqt6 \
    python3-venv \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    libportaudio2

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

printf 'Installing Python inference and eye-tracking packages...\n'
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install onnxruntime==1.30.0 mediapipe==1.1.0

printf 'Checking runtime imports, model files, and camera backend...\n'
PYTHONPATH="$PROJECT_DIR/pc_training:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_DIR/bin/python" - <<'PY'
import cv2
import mediapipe
import numpy
import onnxruntime
from PyQt6 import QtCore

from camera_pipeline import CameraConfig
from predict_onnx import EyeClassifierONNX
from preprocess import FACE_LANDMARKER

if CameraConfig().backend != cv2.CAP_V4L2:
    raise SystemExit("Linux camera backend is not V4L2.")
if FACE_LANDMARKER is None:
    raise SystemExit("MediaPipe could not initialize models/face_landmarker.task.")

EyeClassifierONNX("models/ood_aware2/clear_vision_ood_aware.onnx")
print(f"OpenCV {cv2.__version__}; NumPy {numpy.__version__}; MediaPipe {mediapipe.__version__}")
print(f"ONNX Runtime {onnxruntime.__version__}; camera backend V4L2; PyQt6 {QtCore.PYQT_VERSION_STR}")
PY

printf '\nSetup complete. Start the GUI from the project directory with:\n'
printf '  %q %q\n' "$VENV_DIR/bin/python" "$PROJECT_DIR/pc_training/app_gui.py"
printf '\nTo inspect attached USB cameras, run: v4l2-ctl --list-devices\n'