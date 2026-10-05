"""Frigate-inspired preprocessing for the benchmark harness.

Ports the OpenCV-optimized versions of Frigate's "improved" motion
preprocessing (from the max/optimize-motion branch) so the same techniques
can be applied to the pybgs / cv2 BGS algorithms in this harness:

  * percentile contrast normalization  -> percentile_via_histogram + cv2.LUT
  * gaussian blur                      -> cv2.GaussianBlur (ksize = 2*radius+1)
  * ROI persistence gate               -> grid-streak temporal filter

The first two are applied pre-BGS on the (downscaled) BGR frame; the
persistence gate is applied post-contour extraction. State (contrast
history, streak grid) is per process_video call, i.e. per (video, algorithm),
matching how the BGS model itself is managed.

These run only on the non-Frigate BGS path (the Frigate variant performs its
own internal preprocessing).
"""

import cv2
import numpy as np


class FrigatePreprocessor:
    def __init__(self, cfg: dict, width: int, height: int) -> None:
        self.do_blur = bool(cfg.get("gaussian_blur", True))
        self.blur_sigma = float(cfg.get("gaussian_sigma", 1.0))
        self.blur_ksize = int(cfg.get("gaussian_ksize", 3))

        self.do_contrast = bool(cfg.get("contrast_norm", True))
        self.contrast_history = int(cfg.get("contrast_history", 50))
        self.contrast_min_pct = float(cfg.get("contrast_min_pct", 4.0))
        self.contrast_max_pct = float(cfg.get("contrast_max_pct", 96.0))

        self.min_persist = int(cfg.get("persistence_frames", 0))

        # Frigate keeps a moving window of (min, max) percentiles; column 1
        # (max) is initialized to 255 so the first frames behave sanely.
        self._contrast_values = np.zeros((self.contrast_history, 2), np.uint8)
        self._contrast_values[:, 1:2] = 255
        self._contrast_index = 0
        self._lut = np.zeros((256,), dtype=np.uint8)

        # persistence state: list of (box, consecutive_frame_streak) from the
        # previous frame, matched across frames by center distance so a moving
        # object keeps its streak while a flickering patch does not.
        self._tracked = None

    # Frigate's histogram-accelerated percentile (replaces np.percentile).
    def _percentile_via_histogram(self, gray: np.ndarray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        cum_hist = np.cumsum(hist)
        total = gray.size
        min_thresh = total * (self.contrast_min_pct / 100.0)
        max_thresh = total * (self.contrast_max_pct / 100.0)
        min_value = np.searchsorted(cum_hist, min_thresh).astype(np.uint8)
        max_value = np.searchsorted(cum_hist, max_thresh).astype(np.uint8)
        return min_value, max_value

    # Frigate's cv2 gaussian (replaces scipy.ndimage.gaussian_filter).
    def _gaussian_via_cv2(self, frame):
        k = self.blur_ksize
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), sigmaX=self.blur_sigma)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if self.do_contrast:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            min_value, max_value = self._percentile_via_histogram(gray)
            if min_value < max_value:
                self._contrast_values[self._contrast_index] = [
                    min_value,
                    max_value,
                ]
                self._contrast_index += 1
                if self._contrast_index == self.contrast_history:
                    self._contrast_index = 0

                avg_min, avg_max = np.mean(self._contrast_values, axis=0)

                # LUT rescale replaces Frigate's np.clip + (v-min)/(max-min)
                # per-pixel math (optimization 2 in the branch).
                bins = np.arange(256)
                lut_values = np.clip(
                    (bins - avg_min) * (255.0 / (avg_max - avg_min + 1e-6)),
                    0,
                    255,
                )
                self._lut = lut_values.astype(np.uint8)
                bgr = cv2.LUT(bgr, self._lut)

        if self.do_blur:
            bgr = self._gaussian_via_cv2(bgr)

        return bgr

    def filter_persistent(self, boxes, width: int, height: int):
        """Keep only boxes that have been present (tracking across frames by
        center distance) for min_persist consecutive frames. A moving object
        keeps accumulating its streak; a flickering dappled patch appears in a
        new place each frame, so its streak never reaches the threshold."""
        if self.min_persist <= 0:
            return boxes
        if self._tracked is None:
            self._tracked = []

        prev_boxes = [b for (b, _s) in self._tracked]
        prev_streaks = [s for (_b, s) in self._tracked]

        def center(box):
            return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

        # Greedy nearest-center matching, most confident (closest) pairs first.
        pairs = []
        for i, cb in enumerate(boxes):
            cx1, cy1 = center(cb)
            for j, pb in enumerate(prev_boxes):
                cx2, cy2 = center(pb)
                pairs.append(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2, i, j))
        pairs.sort(key=lambda p: p[0])

        matched = [None] * len(boxes)
        used = [False] * len(prev_boxes)
        for dist2, i, j in pairs:
            if matched[i] is not None or used[j]:
                continue
            cb = boxes[i]
            pb = prev_boxes[j]
            tol = 0.5 * (max(cb[2], cb[3]) + max(pb[2], pb[3]))
            if dist2 <= tol * tol:
                matched[i] = j
                used[j] = True

        out = []
        new_tracked = []
        for i, cb in enumerate(boxes):
            streak = prev_streaks[matched[i]] + 1 if matched[i] is not None else 1
            new_tracked.append((cb, streak))
            if streak >= self.min_persist:
                out.append(cb)
        self._tracked = new_tracked
        return out
