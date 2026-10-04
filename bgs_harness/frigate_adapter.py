"""Runs Frigate NVR's motion detectors (unmodified, vendored at ../frigate)
as benchmark algorithms for the harness.

Frigate is not a pip package; this module puts the pinned source tree on
sys.path and imports the detector classes exactly the way Frigate's own
unit test (frigate/test/test_motion_detector.py) does.
"""

import logging
import sys
from pathlib import Path

import numpy as np

FRIGATE_ROOT = Path(__file__).resolve().parent.parent / "frigate"
FRIGATE_VERSION = "0.18.0-77a66e7"

VARIANTS = {
    "FrigateImprovedMotion": "improved",
}


class _PtzState:
    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return bool(self.value)


class _PtzStub:
    """PTZ metrics stub mirroring frigate/test/test_motion_detector.py."""

    def __init__(self):
        self.autotracker_enabled = _PtzState(False)
        self.motor_stopped = _PtzState(False)
        self.stop_time = _PtzState(0)


def _ensure_frigate_importable() -> None:
    root = str(FRIGATE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    version_file = FRIGATE_ROOT / "frigate" / "version.py"
    if not version_file.exists():
        version_file.write_text(f'VERSION = "{FRIGATE_VERSION}"\n')
    # Frigate logs an error per missing optional detector runtime
    # (onnxruntime, zmq, edgetpu, ...) at import time; expected here.
    logging.getLogger("frigate").setLevel(logging.CRITICAL)


class FrigateMotionAlgorithm:
    """Wraps a Frigate motion detector behind the harness algorithm interface."""

    def __init__(self, variant: str, frame_shape: tuple[int, int], fps: float,
                 params: dict):
        _ensure_frigate_importable()
        from frigate.config.camera.motion import MotionConfig

        mc_kwargs = {
            "threshold": params.get("threshold", 30),
            "improve_contrast": params.get("improve_contrast", True),
            "delta_alpha": params.get("delta_alpha", 0.2),
            "frame_alpha": params.get("frame_alpha", 0.01),
            "lightning_threshold": params.get("lightning_threshold", 0.8),
        }
        contour_area = params.get("contour_area")
        if contour_area is not None:
            mc_kwargs["contour_area"] = int(contour_area)
        # null -> None -> detector uses full processing height (fair
        # comparison); MotionConfig's own default is 100 (Frigate stock).
        mc_kwargs["frame_height"] = (
            int(params["frame_height"])
            if params.get("frame_height")
            else None
        )
        skip_motion = params.get("skip_motion_threshold")
        if skip_motion is not None:
            mc_kwargs["skip_motion_threshold"] = float(skip_motion)
        config = MotionConfig(**mc_kwargs)
        # Full-frame rasterized mask (no exclusion zones); same bypass as
        # Frigate's own unit test.
        object.__setattr__(
            config,
            "rasterized_mask",
            np.ones(frame_shape, dtype=np.uint8),
        )

        # The legacy FrigateMotionDetector is dead code in v0.18: it does not
        # implement the MotionDetector ABC (stop/update_mask), so only the
        # improved variant is exposed.
        from frigate.motion.improved_motion import ImprovedMotionDetector

        self.detector = ImprovedMotionDetector(
            frame_shape, config, int(fps), name=variant
        )
        self.variant = variant
        self.kind = VARIANTS[variant]

    def detect_boxes(self, gray_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Returns ROI boxes as (x, y, w, h) in full-frame pixel coordinates.

        Boxes produced while the detector calibrates its background model are
        dropped, matching how frigate/video/detect.py gates motion regions.
        """
        raw_boxes = self.detector.detect(gray_frame)
        if self.detector.is_calibrating():
            return []
        frame_h, frame_w = gray_frame.shape[:2]
        boxes = []
        for x1, y1, x2, y2 in raw_boxes:
            x1 = max(0, min(int(x1), frame_w))
            y1 = max(0, min(int(y1), frame_h))
            x2 = max(0, min(int(x2), frame_w))
            y2 = max(0, min(int(y2), frame_h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
