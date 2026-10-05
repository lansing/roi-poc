# Plan: Add a cv2 MOG2 Motion Detector to Frigate

Goal: add a **pure-OpenCV `MOG2` background-subtraction detector** as a drop-in
alternative to Frigate's `ImprovedMotionDetector`, selectable from `config.yml`,
and expose a **temporal ROI persistence gate** as a first-class setting (Frigate
has no per-box temporal gate today). Unlike the pybgs plan
(`FRIGATE_PYBGS.md`), this needs **no compiled extension, no `./config/*.xml`
file, and no build step** — `cv2.createBackgroundSubtractorMOG2` is already in
Frigate's pinned OpenCV 4.11 (headless + contrib, `docker/main/requirements-wheels.txt:41-42`).
Clustering / region building is reused untouched (see `FRIGATE_CLUSTERING.md`).
All target files are in the vendored `../frigate` submodule (`v0.18.0`).

---

## 1. Why cv2 MOG2 (vs the pybgs plan)

- **Zero new dependencies.** `pybgs` is a compiled C++ extension with no arm64
  wheel and an XML config side-channel. `cv2` MOG2 is in the image Frigate already
  ships; the detector is pure `cv2` + `numpy` (both already required).
- **Faster.** Our harness measured `cv2` MOG2 at ~1.3 ms/frame vs ~3 ms for
  `pybgs` MOG2 (≈2×) at the same resolution.
- **Dappled suppression is tunable.** With the Frigate-inspired **persistence
  gate + percentile contrast norm** (both pure `cv2`/`numpy`, ported from
  `bgs_harness/frigate_preproc.py`), `cv2` MOG2 (keep-shadows) drops dappled
  trigger from 59% → ~36% while holding wind at ~0% and motion recall at ~76%.
- **Honest tradeoff.** `cv2` MOG2 is still noisier on dappled than `pybgs` MOG2
  (different algorithm implementation). Pick this when you want a fast,
  dependency-free detector; pick the pybgs plan for maximum dappled suppression.

This doc is the spec for that patch. It is intentionally independent of the
benchmark harness: **everything in §10 is testable inside the Frigate codebase
using only `cv2` + `numpy`.**

---

## 2. Design principles

- **One class + a tiny factory + config fields.** The only detector touch-points
  are the `MotionDetector` ABC methods and a `.config` attribute. Replace the
  class at the single instantiation site; do **not** touch `process_frames`,
  clustering, the region grid, stationary throttling, or the object detector.
- **Pure `cv2`/`numpy`.** No `pybgs`, no `line_profiler`, no file I/O for model
  state. MOG2 keeps its background model in memory, per instance.
- **Reuse Frigate's data path.** Operate on the luma plane, downscale to
  `frame_height`, honor `rasterized_mask`, and emit boxes in the exact
  `(x1, y1, x2, y2)` full-frame format clustering expects.
- **Reuse Frigate's YAML + semantics.** New optional fields under the existing
  `motion:` block; mirror `ImprovedMotionDetector`'s calibration / lightning /
  `skip_motion_threshold` / PTZ behavior so region-building behaves identically.
- **Persistence is a new, exposed setting.** Frigate's existing "persistence"
  (the background-update gating in `improved_motion.py:278-294`) is *not* a
  per-box temporal gate. We add one and make it configurable.
- **No change to the default.** `detector: improved` (default) keeps
  `ImprovedMotionDetector` byte-for-byte; the new code path is only taken when
  `detector: mog2` is set.

---

## 3. The detector contract (verified)

`frigate/video/detect.py` uses the detector **only** via:

| Call | Site | Notes |
|---|---|---|
| `motion_detector.detect(frame)` | `detect.py:293` | returns `list[(x1,y1,x2,y2)]`, full-frame px |
| `motion_detector.is_calibrating()` | `detect.py:322,361` | gates whether motion builds regions |
| `motion_detector.config = camera_config.motion` | `detect.py:235` | hot-reload reassigns the config |
| `motion_detector.update_mask()` | `detect.py:236` | called after a motion config change |
| `motion_detector.stop()` | `detect.py:554` | shutdown |

ABC at `frigate/motion/__init__.py:8-39` (the abstract `__init__` is
aspirational; concrete classes define their own). Mirror
`ImprovedMotionDetector.__init__` (`improved_motion.py:16-26`) for drop-in
parity: `(frame_shape, config, fps, name=..., ptz_metrics=...)`.

**Frame format.** Frames are YUV420p (I420) shared-memory buffers; `detect()`
takes the full frame and we read the luma plane `frame[0:H, 0:W]` (2D uint8),
exactly as `improved_motion.py:74` does. **The luma plane is 2D grayscale**, so
MOG2 is fed grayscale directly (no BGR replication, unlike the pybgs plan).
This also means unit tests can pass a plain 2D array
(`frame[0:H,0:W]` on a 2D array is the whole array), matching
`frigate/test/test_motion_detector.py`.

**Config base is `extra="forbid"`.** `FrigateBaseModel` sets
`model_config = ConfigDict(extra="forbid", ...)` (`frigate/config/base.py:7`), so
the static `motion:` block is validated as `MotionConfig` and **rejects unknown
fields**. `RuntimeMotionConfig` overrides with `extra="ignore"`
(`frigate/config/config.py:170`), but the *static* parse still runs first.
**Therefore the new fields MUST be added to `MotionConfig`** (not just the
runtime subclass) or `config.yml` validation fails.

**Other verified facts:** `frame_shape` is `(H, W)`
(`camera.py:231`); `rasterized_mask` is a `(H,W)` uint8 (255 = included,
0 = excluded), built in `RuntimeMotionConfig.__init__`
(`config/config.py:163-168`). There is **no existing detector-selection
mechanism** — `FrigateMotionDetector` (`frigate/motion/frigate_motion.py`) is
defined but never instantiated; only `ImprovedMotionDetector` is used
(`detect.py:91`). So we introduce the selector.

---

## 4. New class: `frigate/motion/cv2_mog2_motion.py`

`class Cv2Mog2MotionDetector(MotionDetector)`. Per-frame `detect()` pipeline
(adapted from the validated `bgs_harness/harness.py` +
`bgs_harness/frigate_preproc.py`):

```
__init__(frame_shape, config, fps, name="mog2", ptz_metrics=None):
    self.name = name
    self.config = config
    self.frame_shape = frame_shape
    m = config.mog2                                  # Mog2MotionConfig (section 6)
    self._history       = m.history                  # default 100
    self._var_threshold = m.var_threshold            # default 24
    self._learning_rate = m.learning_rate            # default 0.05
    self._shadow_mode   = m.shadow_mode              # "keep" | "background"
    self._contrast      = m.contrast_norm            # default True
    self._persist_frames= m.persistence_frames       # default 2 (0 = off)
    self._persist_tol   = m.persistence_match_tolerance  # default 0.5
    self._morph         = m.morphology
    self._warmup        = m.warmup_frames            # default 30

    # downscale to frame_height (reuse Frigate's knob; mirrors improved_motion.py:30-35)
    H, W = frame_shape
    proc_h = config.frame_height or H
    self._resize_factor = H / proc_h
    self._proc_w = round(proc_h * W / H)
    self._proc_h = proc_h

    # contrast state (Frigate's 50-frame moving min/max window)
    self._contrast_values = np.zeros((m.contrast_history, 2), np.uint8); self._contrast_values[:,1:2]=255
    self._contrast_index = 0; self._lut = np.zeros(256, np.uint8)

    self._frame_idx = 0
    self._prev_boxes = []                            # persistence state
    self.calibrating = True
    self._sub = None
    self._build_model()                              # create MOG2
    self.update_mask()                               # build downscaled mask
    self.ptz_metrics = ptz_metrics

_build_model():
    # detectShadows=True always (we need the 127 plane to split keep/background)
    self._sub = cv2.createBackgroundSubtractorMOG2(
        history=self._history, varThreshold=self._var_threshold, detectShadows=True)

detect(frame):
    boxes = []
    if not self.config.enabled:                      return []

    # PTZ guard (mirror improved_motion.py:61-72 / 163-177)
    if self._ptz_moving(): return [full_frame_80pct_box]

    H, W = self.frame_shape
    gray = frame[0:H, 0:W]                           # luma plane (2D); works on 2D test frames
    small = cv2.resize(gray, (self._proc_w, self._proc_h), cv2.INTER_NEAREST)
    small = cv2.bitwise_and(small, small, mask=self._inv_mask)     # masked pixels -> 0

    # optional percentile contrast norm (pure cv2; see _normalize_contrast)
    if self._contrast:
        small = self._normalize_contrast(small)

    # feed the model and grab the foreground plane (0=bg, ~127=shadow, 255=fg)
    fg_model = self._sub.apply(small, self._learning_rate)
    self._frame_idx += 1
    if self.calibrating and self._frame_idx < self._warmup:
        return []                                                             # warmup: learn, emit nothing

    # shadow handling: value-robust (OpenCV 4.11 shadow may be 127, historically 125)
    if self._shadow_mode == "keep":
        fg = cv2.threshold(fg_model, 0, 255, cv2.THRESH_BINARY)[1]            # any non-zero
    else:
        fg = cv2.inRange(fg_model, 255, 255)                                  # only definite fg

    # optional morphology open/close (scrub speckle)
    if self._morph.enabled:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (self._morph.kernel_size,)*2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=self._morph.iterations)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=self._morph.iterations)

    fg[self._mask] = 0                               # re-mask (safety)

    # contours -> boxes in PROC space (area gates; min defaults to contour_area)
    min_area = self._contours.min_area or self.config.contour_area
    max_area = self._proc_w * self._proc_h * self._contours.max_area_ratio
    proc_boxes = []
    for c in cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            x, y, w, h = cv2.boundingRect(c); proc_boxes.append((x, y, w, h))

    # persistence gate (NEW; proc space, tracked by center distance)
    proc_boxes = self._filter_persistent(proc_boxes)

    # scale to full frame -> (x1, y1, x2, y2)
    rf = self._resize_factor
    boxes = [(int(x*rf), int(y*rf), int((x+w)*rf), int((y+h)*rf)) for (x,y,w,h) in proc_boxes]

    # calibration / lightning / skip (mirror improved_motion.py:157-207)
    pct = cv2.countNonZero(fg) / (self._proc_w * self._proc_h)
    if self.config.skip_motion_threshold is not None and pct > self.config.skip_motion_threshold:
        self.calibrating = True; self._reset_state(); return []
    if pct < 0.05 and len(boxes) <= 4:
        self.calibrating = False
    if pct > self.config.lightning_threshold:
        self.calibrating = True; self._reset_state()   # relearn on scene change
    return boxes

is_calibrating():   return self.calibrating
update_mask():      # re-read config.rasterized_mask (downscale to proc, INTER_AREA),
                    # rebuild self._mask/self._inv_mask, _build_model() (relearn bg),
                    # self.calibrating = True, self._frame_idx = 0, _reset_state()
stop():             pass
_reset_state():     self._prev_boxes = []; self._frame_idx = 0
_ptz_moving():      mirror improved_motion.py:61-72
```

Ported subroutines (all `cv2`/`numpy`; sources in `bgs_harness/frigate_preproc.py`
and the `max/optimize-motion` branch of `lansing/frigate`):

- `_normalize_contrast(small)` — Frigate's optimized percentile contrast:
  `cv2.calcHist` → `np.cumsum` → `np.searchsorted` for the 4th/96th percentile,
  a 50-frame moving (min,max) window, then a `cv2.LUT` rescale (replaces
  `np.percentile` + per-pixel `np.clip` math).
- `_filter_persistent(boxes)` — the **new** temporal gate. Track boxes across
  frames by **center distance** (a box inherits the previous frame's streak if
  its center is within `persist_tol * (maxdim_cur + maxdim_prev)` px); a box is
  emitted only when its streak `>= persistence_frames`. A moving object keeps
  its streak; a flickering dappled patch never reaches the threshold. State
  (`self._prev_boxes`) resets on `update_mask` / recalibration.

Model construction is **in-memory** (`cv2.createBackgroundSubtractorMOG2`), so
there is no shared-config-file problem (unlike the pybgs plan). Each camera gets
its own MOG2 instance.

---

## 5. The drop-in wiring (minimal diff)

**Add a factory** to `frigate/motion/__init__.py` (no lazy import needed — `cv2`
is already a hard dependency of the improved detector):

```python
def create_motion_detector(frame_shape, config, fps, name, ptz_metrics):
    if getattr(config, "detector", "improved") == "mog2":
        from frigate.motion.cv2_mog2_motion import Cv2Mog2MotionDetector
        return Cv2Mog2MotionDetector(frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics)
    from frigate.motion.improved_motion import ImprovedMotionDetector
    return ImprovedMotionDetector(frame_shape, config, fps, name=name, ptz_metrics=ptz_metrics)
```

**Replace the hard-coded construction** at `frigate/video/detect.py:91-97`
(import `create_motion_detector` instead of `ImprovedMotionDetector`, line 26):

```python
motion_detector = create_motion_detector(
    frame_shape, self.config.motion, self.config.detect.fps,
    name=self.config.name, ptz_metrics=self.ptz_metrics,
)
```

That is the **only** behavioral change to the detect process. Clustering, region
building, the 8×8 grid, stationary throttling, and NMS are reused as-is.

---

## 6. Config YAML (`config/camera/motion.py`)

Add to **`MotionConfig`** (the `extra="forbid"` base, so static validation
passes). New optional fields, all backward-compatible (defaults preserve stock
behavior when `detector: improved`):

```python
class Mog2MorphologyConfig(FrigateBaseModel):
    enabled: bool = True
    kernel_size: int = 3
    iterations: int = 3

class Mog2ContoursConfig(FrigateBaseModel):
    min_area: int | None = None          # None -> fall back to MotionConfig.contour_area
    max_area_ratio: float = 0.9

class Mog2MotionConfig(FrigateBaseModel):
    history: int = 100
    var_threshold: int = 24              # cv2 MOG2 variance threshold (tuned up for dappled)
    learning_rate: float = 0.05
    detect_shadows: bool = True
    shadow_mode: Literal["keep", "background"] = "keep"
    contrast_norm: bool = True
    contrast_history: int = 50
    contrast_min_pct: float = 4.0
    contrast_max_pct: float = 96.0
    persistence_frames: int = 2          # 0 disables the temporal gate
    persistence_match_tolerance: float = 0.5
    morphology: Mog2MorphologyConfig = Mog2MorphologyConfig()
    contours: Mog2ContoursConfig = Mog2ContoursConfig()
    warmup_frames: int = 30

# on MotionConfig:
    detector: Literal["improved", "mog2"] = "improved"
    mog2: Mog2MotionConfig = Mog2MotionConfig()
```

Example `config.yml`:

```yaml
cameras:
  - name: front
    motion:
      detector: mog2          # "improved" = stock Frigate (default)
      contour_area: 10        # reused as mog2 min_area default (same px scale)
      frame_height: 100       # reused downscale knob
      lightning_threshold: 0.8
      mog2:
        history: 100
        var_threshold: 24
        learning_rate: 0.05
        shadow_mode: keep
        contrast_norm: true
        persistence_frames: 2          # <- the new temporal gate
        persistence_match_tolerance: 0.5
        morphology: { enabled: true, kernel_size: 3, iterations: 3 }
        contours: { min_area: null, max_area_ratio: 0.9 }
        warmup_frames: 30
```

All knobs are per-camera and hot-reloadable: a `motion` config change already
triggers `config = ...` + `update_mask()` at `detect.py:235-236`, which
re-instantiates the MOG2 and resets persistence/warmup.

---

## 7. The persistence setting (the new knob)

**What Frigate has today:** `ImprovedMotionDetector` gates the *background
update* (only bakes a frame into the running average after 10 consecutive
motion frames, `improved_motion.py:278-294`). It does **not** gate the *output*
boxes — any contour over `contour_area` is emitted the frame it appears. There is
no per-box temporal persistence setting.

**What we add:** `persistence_frames` (and `persistence_match_tolerance`). A box
is emitted only if the same region has been present for `persistence_frames`
consecutive frames, where "the same region" is matched by center distance with a
tolerance proportional to box size (so a moving object tracks cleanly). This is
what made dappled suppression work in the harness ablation:

| cv2 MOG2 (keep), dappled trigger % (default `persistence_frames: 2`) | none | persist only | persist + contrast |
|---|---|---|---|
| value | 59.05 | 44.76 | **36.19** |

Motion recall cost is ~2 pts trigger (noshadow ~5). `persistence_frames: 0`
disables the gate entirely (stock-like behavior). State resets on
`update_mask`/recalibration. It is cheap (a small center-distance match per
frame) and resolution-independent (streaks are in frames; tolerance is relative
to box size), so it transfers across `frame_height`.

---

## 8. Contours: reuse `contour_area`, add `max_area_ratio`

Same reasoning as `FRIGATE_PYBGS.md` §6: `contour_area` **is** a minimum-area
gate (`improved_motion.py:146`), so default `mog2.contours.min_area` to
`config.contour_area`. We need a second gate (`max_area_ratio`) Frigate doesn't
expose, hence the small `mog2.contours` sub-config. **Scale caveat:**
`contour_area` is measured at `frame_height` (default 100); our harness
`min_area: 200` was tuned at 720-wide, so the numbers are only directly
comparable at the same `frame_height`. At the stock `frame_height: 100`, inherit
`contour_area`; if we raise `frame_height` for better dappled separation, set an
explicit `min_area`.

---

## 9. Caveats / constraints

- **Different algorithm than pybgs MOG2.** `cv2` MOG2 and `pybgs`
  `MixtureOfGaussianV2` are distinct implementations; `cv2` is faster but noisier
   on dappled even with persistence + contrast. Expect the ~36% dappled-trigger
   number (keep, `persistence_frames=2`), not pybgs's ~1%.
- **Shadow-value robustness.** OpenCV 4.11 may emit shadow as `127`
  (historically `125`). Never threshold at a magic `125`; split by value as in
  §4 (`inRange(255,255)` for background, `threshold(0)` for keep).
- **Re-tune at `frame_height`.** `var_threshold`, `history`, and
  `persistence_frames` are scale-robust; `contour_area`/`min_area` and the
  morphology kernel are scale-dependent. Run one clip-eval (§10) at the chosen
  `frame_height` before locking defaults.
- **Calibration / recalibration.** MOG2 needs a short warmup (`warmup_frames`);
  on a large scene change (`pct > lightning_threshold`) re-instantiate the model
  and re-warmup (mirrors the PTZ-stop baseline reset in `improved_motion.py:176`).
- **PTZ.** Port the autotrack guard (§4 `_ptz_moving`), or mark v1 as
  "PTZ autotrack not handled by the mog2 detector." Recommend porting (small).
- **No shared-file hazard.** Unlike pybgs, MOG2 state is in-memory per instance,
  so multiple cameras with different `mog2.*` params are safe.

---

## 10. Testing & rollout — **within the Frigate codebase only**

The benchmark harness is **not** available during this work. Everything below
uses only `cv2` + `numpy` (already Frigate deps) and lives in the Frigate repo.
Follow `frigate/AGENTS.md`: unittest, `python3 -u -m unittest`.

**Tier 1 — Unit tests: `frigate/test/test_mog2_motion.py`** (mirror
`frigate/test/test_motion_detector.py`, which drives `detect()` with plain 2D
grayscale frames — the luma slice handles that). Synthesize frames in `numpy`:

- **Static scene** (constant background) → after `warmup_frames`, `detect()`
  returns `[]`; `is_calibrating()` is `False`.
- **Moving object** — a white square translating across the frame → a box
   appears and its center tracks the object; with the default `persistence_frames=2`
   the box first appears no earlier than frame 2 of the motion (assert streak
   gating), and is continuous once established.
- **Dappled-like flicker** — per-pixel random noise that changes every frame with
   no stable object → with `persistence_frames=2`, box count/coverage stays ~0;
  with `persistence_frames=0` it rises (proves the gate is what suppresses it).
- **Shadow modes** — a shadow-like intensity drop: `shadow_mode=keep` yields a
  box, `shadow_mode=background` does not.
- **Config surface** — `persistence_frames=0` disables the gate; changing
  `config.mog2.*` + `detector.config = ...` + `update_mask()` re-instantiates the
  model and resets state; `stop()` is safe; a 2D grayscale frame **and** a 3D
  YUV-shaped frame both produce the same luma result.
- **Box format** — every box is `(x1,y1,x2,y2)` with `x2>x1, y2>y1` and within
  `[0, frame_shape]`; boxes are in **full-frame** pixels (assert a proc-space
  box scales by `resize_factor`).
- **Factory** — `create_motion_detector(..., config.detector="mog2")` returns a
  `Cv2Mog2MotionDetector`; `detector="improved"` returns `ImprovedMotionDetector`
  and does **not** import `cv2_mog2_motion`.

**Tier 2 — In-repo clip eval: `frigate/test/eval_mog2_motion.py`.** A small
standalone script (run with `python3`, not unittest) that reads local clips via
`cv2.VideoCapture` and drives the detector directly, reproducing the two metrics
we optimize for **without the harness**:

- **trigger %** = (frames with ≥ 1 box) / (total frames) × 100
- **coverage %** = mean over frames of (ROI union area / frame area) × 100 (ROI
  union via a `np.zeros` canvas + `cv2.countNonZero`, as in the harness)

It accepts `--detector mog2|improved --persistence N --contrast on|off --video
<path>` and prints a per-clip table. Point it at three clips
(`motion_target`, `wind_trees`, `dappled_shadows`) and confirm: `dappled` drops
with persistence, `wind` stays ~0, `motion` recall holds. This is the
substitute for the harness comparison and runs inside the Frigate environment.

**Tier 3 — Regression.** With `detector: improved` (default) the detect process
is behaviorally identical and `cv2_mog2_motion` is never imported.

### File change list

| Action | File | Change |
|---|---|---|
| NEW | `frigate/motion/cv2_mog2_motion.py` | `Cv2Mog2MotionDetector` (§4) |
| EDIT | `frigate/motion/__init__.py` | add `create_motion_detector` factory |
| EDIT | `frigate/video/detect.py` | import factory; replace lines 91-97; drop line-26 import |
| EDIT | `frigate/config/camera/motion.py` | add `detector`, `mog2` + the three sub-models |
| NEW | `frigate/test/test_mog2_motion.py` | unit tests (Tier 1) |
| NEW | `frigate/test/eval_mog2_motion.py` | in-repo clip eval (Tier 2) |
| — | (no build/requirements changes) | `cv2` is already a Frigate dependency |

### Phases

- **P1 (POC):** class + factory + MOG2 tuned defaults (`var_threshold=24`,
  `history=100`, `shadow_mode=keep`) + **persistence** + contours
  (`min_area` ← `contour_area`, `max_area_ratio`). Unit tests (Tier 1) green on
  synthetic frames.
- **P2:** `contrast_norm` (cv2 histogram + LUT) + morphology; wire the clip-eval
  script; re-tune `contour_area`/morphology at the chosen `frame_height`.
- **P3:** PTZ guard; calibration/recalibration parity; multi-camera; E2E in a
  Frigate container against a real/file source.

---

## Appendix: Settings reference

**Reused `motion:` knobs** (on `MotionConfig`; the `mog2` detector reads these):

| Setting | Default | Effect |
|---|---|---|
| `detector` | `improved` | Selects the detector: `improved` (stock) or `mog2` (this one). |
| `contour_area` | `10` | Min contour area (px); reused as `mog2.contours.min_area` default. |
| `frame_height` | `100` | Height frames are downscaled to for motion (sets feature scale). |
| `lightning_threshold` | `0.8` | Frame-change fraction that forces recalibration. |
| `skip_motion_threshold` | `null` | Frame-change fraction that skips the frame + recalibrates. |

(`motion.improve_contrast` and `motion.threshold` drive the *stock* detector only;
`mog2` uses `mog2.contrast_norm` and `mog2.var_threshold` instead.)

**New `motion.mog2.*` knobs** (on `Mog2MotionConfig`):

| Setting | Default | Effect |
|---|---|---|
| `history` | `100` | MOG2 background-model history in frames (memory/robustness). |
| `var_threshold` | `24` | MOG2 variance threshold; higher → fewer pixels flagged (tuned up for dappled). |
| `learning_rate` | `0.05` | Per-frame background adaptation rate. |
| `detect_shadows` | `true` | Always `true`; keeps the shadow plane so `shadow_mode` can split it. |
| `shadow_mode` | `keep` | Whether MOG2 shadow (127) pixels count as foreground (`keep` / `background`). |
| `contrast_norm` | `true` | Percentile contrast normalization (cv2 `calcHist` + `LUT`). |
| `contrast_history` | `50` | Moving-window length for the min/max percentile baseline. |
| `contrast_min_pct` | `4.0` | Lower percentile bound for normalization. |
| `contrast_max_pct` | `96.0` | Upper percentile bound for normalization. |
| `persistence_frames` | `2` | Min consecutive frames a region must persist before a box is emitted (`0` = off). |
| `persistence_match_tolerance` | `0.5` | Center-distance match factor (× box size) for tracking a region across frames. |
| `morphology.enabled` | `true` | Apply open/close morphology to the foreground mask. |
| `morphology.kernel_size` | `3` | Morphology kernel size. |
| `morphology.iterations` | `3` | Morphology iterations. |
| `contours.min_area` | `null` | Min ROI area (proc px); `null` → fall back to `contour_area`. |
| `contours.max_area_ratio` | `0.9` | Max ROI area as a fraction of the proc frame. |
| `warmup_frames` | `30` | Frames the model learns the background before emitting boxes. |
