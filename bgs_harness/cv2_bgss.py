"""Thin OpenCV background-subtractor wrappers for the benchmark harness.

Each wrapper exposes the same `.apply(frame) -> mask` interface as the pybgs
algorithms so they can be benchmarked side-by-side through the identical
downscale -> morphology -> contour -> metrics pipeline in harness.py.

Only the standard OpenCV `video` module is used here
(cv2.createBackgroundSubtractorMOG2). The opencv-contrib `bgsegm` variants
(MOG / GMG) are intentionally NOT included.

`OpenCVMOG2` wraps cv2 MOG2 and makes the handling of the 125 "shadow" pixels
explicit so we can benchmark both policies:
  - shadow_mode="keep"       : shadow pixels (125) count as foreground
  - shadow_mode="background" : shadow pixels (125) are treated as background
`apply()` always returns a binary 0/255 mask (matching what the pybgs MOG2 path
produces after its own threshold step).
"""

import cv2


class OpenCVMOG2:
    """cv2.createBackgroundSubtractorMOG2 with an explicit shadow policy."""

    def __init__(
        self,
        shadow_mode: str = "keep",
        history: int = 100,
        var_threshold: int = 24,
        learning_rate: float = 0.05,
    ) -> None:
        if shadow_mode not in ("keep", "background"):
            raise ValueError(
                f"shadow_mode must be 'keep' or 'background', got {shadow_mode!r}"
            )
        self.shadow_mode = shadow_mode
        # learning_rate defaults to 0.05 to match the pybgs MOG2 alpha used by
        # the harness, so the two MOG2 implementations are directly comparable.
        self._learning_rate = learning_rate
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )

    def apply(self, frame):
        fg = self._sub.apply(frame, self._learning_rate)
        # MOG2 emits 0=background, 127=shadow (OpenCV 5.x; historically 125),
        # 255=foreground. Separate foreground from shadow by value, not by a
        # magic threshold, so it is robust to the exact shadow constant.
        if self.shadow_mode == "keep":
            # any non-zero (255 fg or 127 shadow) -> foreground
            return cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY)[1]
        # background: keep only definite foreground (255); shadows (125/127) -> 0
        return cv2.inRange(fg, 255, 255)


# Registry: algorithm name (as used in config.json) -> factory.
# Names here are neither pybgs classes nor Frigate variants; harness.py checks
# this registry before falling back to pybgs.
CV2_ALGORITHMS = {
    "OpenCVMOG2": lambda: OpenCVMOG2(shadow_mode="keep"),
    "OpenCVMOG2_noshadow": lambda: OpenCVMOG2(shadow_mode="background"),
}
