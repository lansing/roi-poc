# Plan: Replace Frigate Motion Detection with a pybgs + Morphology Pipeline

Goal: swap Frigate's `ImprovedMotionDetector` for a `pybgs` background-subtraction
detector (with morphology), while **reusing Frigate's existing clustering/region
building** (see `FRIGATE_CLUSTERING.md`) untouched. Keep the change as a clean
drop-in: one new class + a tiny factory + config fields. All target files are in
the vendored `../frigate` submodule (used unmodified at `v0.18.0`); this plan is the
spec for that patch.

---

## 1. Design principles

- **One class swap.** Frigate's only detector touch-points are the `MotionDetector`
  ABC methods and a `.config` attribute. Replace the class at the single
  instantiation site; do **not** touch `process_frames`, clustering, the region
  grid, or the object detector.
- **Reuse Frigate's data path.** Operate on the luma plane, downscale to
  `frame_height`, honor `rasterized_mask`, and emit boxes in the exact
  `(x1, y1, x2, y2)` full-frame format the clustering expects.
- **Reuse Frigate's YAML.** Add optional fields to the existing `motion:` config so
  everything is selectable/tunable from `config.yml`.
- **No dependency unless selected.** `pybgs` is imported lazily, so a
  `detector: improved` deployment never needs it.

---

## 2. The detector contract (verified)

`frigate/video/detect.py` uses the detector **only** via:

| Call | Site | Notes |
|---|---|---|
| `motion_detector.detect(frame)` | `detect.py:293` | returns `list[(x1,y1,x2,y2)]`, full-frame px |
| `motion_detector.is_calibrating()` | `detect.py:322,361` | gates whether motion builds regions |
| `motion_detector.config = camera_config.motion` | `detect.py:235` | hot-reload reassigns the config |
| `motion_detector.update_mask()` | `detect.py:236` | called after a motion config change |
| `motion_detector.stop()` | `detect.py:554` | shutdown |

ABC at `frigate/motion/__init__.py:8-39` (the abstract `__init__` signature is
aspirational; concrete classes define their own). We mirror
`ImprovedMotionDetector.__init__` (`improved_motion.py:16-26`) for drop-in parity:
`(frame_shape, config, fps, name=..., ptz_metrics=...)`.

**Frame format:** frames are YUV420p (I420) shared-memory buffers
(`frigate/util/image.py:895-899`). `detect()` receives the full frame; we take the
luma plane `frame[0:H, 0:W]` (2D uint8), exactly as `improved_motion.py:74` does.

---

## 3. New class: `frigate/motion/pybgs_motion.py`

`class PybgsMotionDetector(MotionDetector)`. Per-frame `detect()` pipeline
(adapted from `bgs_harness/harness.py`, the validated pipeline):

```
detect(frame):
  if not self.config.enabled:                 return []
  # PTZ guard (mirror improved_motion.py:61-72): while autotrack motor moves,
  # return a single 80%-frame box; on stop, reset the model + return [].
  if self._ptz_moving():                      return [full_frame_80pct_box]

  H, W = self.config.frame_shape[0], self.config.frame_shape[1]
  gray = frame[0:H, 0:W]                      # luma plane (2D)

  # downscale to frame_height (reuse Frigate's knob; mirrors improved_motion.py:30-35)
  proc_h = self.config.frame_height or H
  resize_factor = H / proc_h
  proc_w = round(proc_h * W / H)
  small = cv2.resize(gray, (proc_w, proc_h), cv2.INTER_NEAREST)

  # apply mask (reuse config.rasterized_mask; mirrors improved_motion.py:115)
  small[self._mask] = 0

  # optional light blur to match improved's noise handling (improved_motion.py:117)
  if self._blur: small = gaussian_filter(small, sigma=1)

  # feed pybgs (expects 3ch). Replicate luma to BGR (POC: motion is in luma).
  rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
  if self.calibrating:
      self._warmup()                          # apply to model, count down warmup
      return []

  fg = self.algo.apply(rgb)
  if fg is None:                              return []
  if fg.ndim == 3:   fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
  if fg.dtype != np.uint8: fg = cv2.normalize(fg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  if cv2.countNonZero(fg) == 0 or fg.max() < 255:
      fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)[1]

  # morphology (our pipeline; from config.pybgs.morphology)
  if morph.enabled:
      k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph.kernel_size,)*2)
      fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=morph.iterations)
      fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=morph.iterations)

  # mask out excluded regions again (safety: no ROI in masked area)
  fg[self._mask] = 0

  # contours -> boxes (same area gates as the harness)
  min_area  = contours.min_area                # default -> self.config.contour_area
  max_area  = proc_w * proc_h * contours.max_area_ratio
  boxes = []
  for c in cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]:
      a = cv2.contourArea(c)
      if min_area <= a <= max_area:
          x, y, w, h = cv2.boundingRect(c)
          boxes.append((int(x*rf), int(y*rf), int((x+w)*rf), int((y+h)*rf)))   # (x1,y1,x2,y2)

  # calibration heuristic (mirror improved_motion.py:157-207)
  pct = cv2.countNonZero(fg) / (proc_w * proc_h)
  if not self.calibrating and pct < 0.05 and len(boxes) <= 4:
      ...  # (already calibrated)
  if pct > self.config.lightning_threshold:
      self.calibrating = True
  return boxes
```

Other methods:
- `is_calibrating()` -> `self.calibrating`.
- `update_mask()` -> re-read `config.rasterized_mask` (downscaled to `proc_w×proc_h`
  via `INTER_AREA`), **re-instantiate `self.algo`** (relearn background), set
  `calibrating = True`. (Mirror `improved_motion.py:256-268`.)
- `stop()` -> no-op.
- `_ptz_moving()` -> mirror `improved_motion.py:61-72 / 163-177` so PTZ cameras
  behave identically.

**Model construction + per-algo params** (the mechanism our `sweep.py` already
proves): pybgs reads `./config/<Algo>.xml` at construction. On `__init__` and on
`update_mask()` we write `config.pybgs.params` into that FileStorage via
`cv2.FileStorage`, then `self.algo = getattr(pybgs, algorithm)()`.

---

## 4. The drop-in wiring (minimal diff)

**Add a factory** to `frigate/motion/__init__.py` (lazy imports so `improved`
needs no `pybgs`):

```python
def create_motion_detector(frame_shape, config, fps, name, ptz_metrics):
    if getattr(config, "detector", "improved") == "pybgs":
        from frigate.motion.pybgs_motion import PybgsMotionDetector
        return PybgsMotionDetector(frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics)
    from frigate.motion.improved_motion import ImprovedMotionDetector
    return ImprovedMotionDetector(frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics)
```

**Replace the hard-coded construction** at `frigate/video/detect.py:91-97` with:

```python
from frigate.motion import create_motion_detector          # (replace import at line 26)
...
motion_detector = create_motion_detector(
    frame_shape, self.config.motion, self.config.detect.fps,
    name=self.config.name, ptz_metrics=self.ptz_metrics,
)
```

That is the **only** behavioral change to the Frigate detect process. Clustering,
region building, the 8×8 grid, stationary throttling, and NMS are all reused as-is.

*(Minimal alternative if we don't want a factory: a 4-line `if/else` directly in
`detect.py`. The factory is preferred for testability.)*

---

## 5. Config YAML (`config/camera/motion.py`)

Extend `MotionConfig` with optional fields (backward compatible;
`RuntimeMotionConfig` has `extra="ignore"` so unknowns are already safe). Add a
nested `PybgsMotionConfig` sub-model for grouping:

```python
class PybgsMorphologyConfig(FrigateBaseModel):
    enabled: bool = True
    kernel_size: int = 3
    iterations: int = 1

class PybgsContoursConfig(FrigateBaseModel):
    min_area: int | None = None        # None -> fall back to MotionConfig.contour_area
    max_area_ratio: float = 0.9

class PybgsMotionConfig(FrigateBaseModel):
    algorithm: str = "MixtureOfGaussianV2"
    params: dict[str, Any] = Field(default_factory=dict)   # e.g. {"alpha":0.05,"threshold":15}
    morphology: PybgsMorphologyConfig = PybgsMorphologyConfig()
    contours: PybgsContoursConfig = PybgsContoursConfig()
    warmup_frames: int = 30

# on MotionConfig:
    detector: Literal["improved", "pybgs"] = "improved"
    pybgs: PybgsMotionConfig = PybgsMotionConfig()
```

Example `config.yml`:

```yaml
cameras:
  - name: front
    motion:
      detector: pybgs          # "improved" to fall back to stock Frigate
      contour_area: 40         # reused as pybgs min_area default (same px scale)
      pybgs:
        algorithm: MixtureOfGaussianV2
        params: { alpha: 0.05, threshold: 15 }   # written to pybgs config XML
        morphology: { enabled: true, kernel_size: 3, iterations: 1 }
        contours: { min_area: null, max_area_ratio: 0.9 }
        warmup_frames: 30
```

All knobs are per-camera, hot-reloadable (a `motion` config change already triggers
`config = ...` + `update_mask()` at `detect.py:235-236`).

---

## 6. Contours: pick up from Frigate YAML, or our own block?

**Answer: have our own small `pybgs.contours` block, but default `min_area` to
`contour_area`.** Rationale:

- **Semantically aligned.** Frigate's `contour_area` (default `10`) is exactly a
  *minimum contour area* gate (`improved_motion.py:146`) — the same concept as our
  `min_area`. So reusing it as the default is meaningful, not a stretch.
- **Scale is the catch.** `contour_area` is measured at Frigate's downscale
  (`frame_height`, default 100). Our harness `min_area: 200` was tuned at 720-wide.
  The numbers are only directly comparable **if our detector downscales to the same
  `frame_height`** (which it does in this plan, reusing the knob). At that scale,
  `contour_area` is directly usable; `max_area_ratio` has no Frigate equivalent, so
  it stays ours.
- **We need two gates, Frigate has one.** Our pipeline needs `min_area` **and**
  `max_area_ratio`; Frigate only exposes a min. A dedicated sub-config is cleaner
  than overloading the single `contour_area` int.

So: `pybgs.contours.min_area` defaults to `self.config.contour_area` (picks up the
existing YAML), `max_area_ratio` is new, both overridable. If we later run at a
larger downscale (e.g., to match our 720 tuning), we set an explicit `min_area`
instead of relying on the inherited value.

---

## 7. Caveats / constraints

- **pybgs config XML is single-writer.** pybgs reads/writes `./config/<Algo>.xml`
  (CWD-relative, shared). Multiple cameras using the same algo with *different*
  `params` will fight over the file (last ctor/dtor wins). v1: treat algo `params`
  as effectively shared (fine for a single-camera POC). Morphology/contours are
  implemented in our code (no XML), so they are safely per-camera. Multi-camera
  per-algo params would need per-instance CWD isolation (deferred).
- **Re-tune at `frame_height` scale.** Our tuned `alpha`/`threshold`/morphology
  were measured at 720-wide. At `frame_height` (100) the feature scale differs, so
  run one MOG2 sweep at the Frigate downscale before finalizing defaults.
- **Build/deploy.** `pybgs` is a compiled extension (no arm64 wheel; built from
  sdist with `CMAKE_PREFIX_PATH` to OpenCV). Add `pybgs` to Frigate's requirements
  and ensure the Docker image can build it. The POC reuses the existing
  `bgs_harness/.venv` (which already has pybgs).
- **PTZ.** Port the autotrack guard (section 3) for parity, or mark v1 as
  "PTZ autotrack not handled by the pybgs detector." Recommend porting (small).
- **Shadow semantics.** At `frame_height`, MOG2's `threshold` is still a
  keep/drop-shadows toggle (see the MOG2 sweep notes). Defaults are a starting
  point, not the tuned optimum.

---

## 8. Testing & rollout

1. **Unit test** `frigate/test/test_pybgs_motion.py` mirroring
   `test_motion_detector.py`: assert `detect()` returns `(x1,y1,x2,y2)` boxes, that
   a moving object yields a box, that a static scene yields none after warmup, and
   that `update_mask()`/`stop()`/`is_calibrating()` behave.
2. **End-to-end:** run Frigate against the three `processed_videos/`
   (`motion_target`, `wind_trees`, `dappled_shadows`) with `detector: pybgs`, and
   compare resulting regions/inferences against (a) stock `improved` and (b) the
   `bgs_harness` metrics. Confirm clustering still receives sane box counts.
3. **Regression:** with `detector: improved` (default), verify Frigate is
   byte-identical in behavior (no pybgs import occurs).

### File change list

| Action | File | Change |
|---|---|---|
| NEW | `frigate/motion/pybgs_motion.py` | `PybgsMotionDetector` (section 3) |
| EDIT | `frigate/motion/__init__.py` | add `create_motion_detector` factory |
| EDIT | `frigate/video/detect.py` | import factory; replace lines 91-97; drop line-26 import |
| EDIT | `frigate/config/camera/motion.py` | add `detector`, `pybgs` + the three sub-models |
| EDIT | (build) requirements / Dockerfile | add `pybgs` + OpenCV build path |
| NEW | `frigate/test/test_pybgs_motion.py` | unit tests |

### Phases

- **P1 (POC):** P1 class + factory + `MixtureOfGaussianV2` + morphology + contours
  (min from `contour_area`, max_ratio). Single camera, 3 videos.
- **P2:** arbitrary `pybgs.algorithm` + `params` passthrough; PTZ guard; unit tests green.
- **P3:** re-tune MOG2 at `frame_height`; multi-camera params isolation; Docker build.
