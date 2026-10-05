# Frigate: Motion-Box Clustering & Region Building

How Frigate turns a list of small motion boxes into a small number of
detector-sized regions, so that it does not send every motion box to the
object detector.

All references are to the vendored submodule pinned at `v0.18.0`
(`../frigate`), used unmodified.

## TL;DR

- The motion detector emits **one small box per contour** and never merges.
- A separate **region-building** stage groups nearby motion boxes into
  **clusters** and builds **one square region per cluster**.
- A *region* (not a box) is what gets cropped, resized to the model input,
  and run through the detector. **Fewer regions = fewer inferences.**
- Several small motion blobs that fit inside one feasible model-sized square
  are collapsed into a **single** `>=320px` square and sent as **one** crop.
- Additional gates (stationary throttle, calibration, global-motion, NMS)
  further cut the number of frames/regions actually run through the model.

---

## Two-stage architecture

### Stage 1 — Motion detector (raw boxes, no merging)

`frigate/motion/improved_motion.py`

`detect()` (`improved_motion.py:53`) returns `motion_boxes`, where **each
contour** above `config.contour_area` becomes its own box
(`improved_motion.py:142-155`). Box format is `(x1, y1, x2, y2)`.

It also computes a global `pct_motion`
(`improved_motion.py:157-159`) = total contour area / frame area.

Gates that run *inside* the detector and suppress output:

- **PTZ autotracking** (`improved_motion.py:61-72`): while the motor is
  moving, return a single box that is 80% of the frame.
- **Scene-change skip** (`improved_motion.py:187-193`): if
  `pct_motion > config.skip_motion_threshold` (IR switch, lightning), drop
  the frame and recalibrate.
- **Calibration / lightning** (`improved_motion.py:195-207`): if still
  calibrating, or `pct_motion > config.lightning_threshold`, stay in
  calibration and (for lightning) halt further processing for the frame.
  Calibration ends when `pct_motion < 0.05` and `len(motion_boxes) <= 4`.

### Stage 2 — Region building (`frigate/video/detect.py`)

The per-frame loop (`process_frames`, `detect.py:176`) is where boxes become
regions. Key steps:

1. `motion_boxes = motion_detector.detect(frame)` (`detect.py:293`).
2. Build regions for **tracked objects** (`detect.py:327-348`).
3. Build regions for **standalone motion** (see below) (`detect.py:361-383`).
4. On the first frame, add **startup regions** (`detect.py:386-391`).
5. For **each region**, crop the frame, resize to the model input, and run
   the detector (`detect.py:408-419` -> `detect()` at `detect.py:139`).

`detect()` (`detect.py:139-151`) calls `create_tensor_input(frame,
model_config, region)` which crops the region out of the frame and resizes it
to the model's input dimensions. **One region == one model inference.**

---

## The grouping algorithm (the core)

Only motion **not already covered by a tracked object** is turned into new
regions. This is the `standalone_motion_boxes` filter (`detect.py:363-365`):

```python
standalone_motion_boxes = [b for b in motion_boxes if not inside_any(b, regions)]
```

These standalone boxes are then clustered and one region is built per cluster
(`detect.py:367-383`):

```python
motion_clusters = get_cluster_candidates(frame_shape, region_min_size, standalone_motion_boxes)
motion_regions  = [get_cluster_region_from_grid(...) for candidate in motion_clusters]
regions += motion_regions
```

### Clustering: `get_cluster_candidates` (`frigate/util/object.py:392`)

Greedy, order-dependent clustering:

- For each unused box `b`, start a new cluster `cluster = [b]`.
- Compute a `cluster_boundary` for `b` (below).
- For every other unused box, if it is **inside** `cluster_boundary`,
  tentatively add it and compute the resulting cluster region.
  - **5% rule** (`object.py:423-430`): if the merged region is larger than
    `min_region` and *any* member box would be `< 5%` of the merged region's
    area, **reject the merge** (`should_cluster = False`). This stops a large
    blob from dragging in a far-away speck into an oversized region where the
    speck would be sub-pixel and undetectable.
  - Otherwise merge.
- Deduplicate clusters (set of sorted box-index tuples) and return.

### Cluster boundary: `get_cluster_boundary` (`frigate/util/object.py:372`)

A square centered on the box, sized from the **"the box is ~10% of the
region"** rule:

```python
max_region_area   = abs(box_w * box_h) / 0.1          # box = 10% of region
max_region_size   = max(min_region, int(sqrt(max_region_area)))
max_x_dist        = int(max_region_size - box_w / 2 * 1.1)
max_y_dist        = int(max_region_size - box_h / 2 * 1.1)
# boundary = square centered on the box with half-extents (max_x_dist, max_y_dist)
```

Interpretation: two boxes merge if the second can fit in a square that can
still be covered by one feasible detector region. The `* 1.1` makes the
boundary slightly tighter than the full region size.

### Region sizing: `get_cluster_region` (`frigate/util/object.py:442`) + `calculate_region`

`get_cluster_region` takes the min/max of the cluster's boxes and calls
`calculate_region` with a **`1.35` padding multiplier**
(`frigate/util/image.py:506`):

```python
size = int(longest_edge * multiplier // 4 * 4)   # multiple of 4
if size < model_size: size = model_size          # floor at min_region
x_offset = cluster_center_x - size / 2           # clamped to frame bounds
y_offset = cluster_center_y - size / 2           # clamped to frame bounds
return (x_offset, y_offset, x_offset + size, y_offset + size)   # a SQUARE
```

So every region is a **square**, at least `min_region` on a side, centered on
the cluster, and clamped to the frame.

`min_region` = `get_min_region_size(model_config)`
(`frigate/util/object.py:271`):

- model max dim `< 320` -> the model max dim (rounded up to a multiple of 4)
- model max dim `>= 320` -> **320**

---

## What actually gets sent to the detector

Number of detector invocations per frame ≈

```
(# tracked-object regions) + (# standalone-motion clusters) + (startup regions, frame 1 only)
```

- Tracked objects reuse their own box/region (`detect.py:327-348`).
- Nearby standalone motion boxes collapse into **one** region each
  (`detect.py:367-383`).
- **Overlap handling:** tracked boxes that already intersect current motion
  are excluded from the "stationary" set (`detect.py:320-323`); standalone
  motion inside a tracked region is skipped (`detect.py:363-365`).

After the model runs, `reduce_detections` (`frigate/util/object.py:491`)
applies per-label NMS to suppress weak overlapping detections.

---

## The learned 8x8 region-size grid (optional refinement)

A per-camera **8x8 grid** (`GRID_SIZE = 8`, `frigate/util/object.py:35`)
stores, for each screen location, the historically observed object region
size (mean + std dev), learned from past tracked objects via the timeline
(`get_camera_regions_grid`, `object.py:43`, rebuilt nightly at 02:00 and
fetched in `process_frames`, `detect.py:266-268`).

`get_region_from_grid` (`object.py:167`) consults the cell under the region
centroid:

- If the computed region size is within `[mean - std, mean + std]` for that
  cell, keep it.
- If it is *smaller* than expected there, **grow the region to the learned
  size** (so it is not too small to contain the object).
- If it is *larger* than expected, keep it (noted as a TODO in the source).

On startup, `get_startup_regions` (`object.py:457`) runs the **8 most
populated cells** so common object locations are scanned immediately.

---

## Key tunable parameters

| Knob | Where | Default | Effect |
|---|---|---|---|
| `min_region` | `get_min_region_size` (`object.py:271`) | 320 (models >= 320) | Floor on region side length; small blobs padded up to this |
| Region padding multiplier | `get_cluster_region` (`object.py:442`) | 1.35 | How much the cluster is inflated before squaring |
| "box = 10% of region" | `get_cluster_boundary` (`object.py:372`) | `/0.1` | Max cluster extent; controls how far apart boxes may be to still merge |
| "box >= 5% of merged region" | `get_cluster_candidates` (`object.py:428`) | 0.05 | Rejects merges that would make a member sub-scale |
| `GRID_SIZE` | `object.py:35` | 8 | Resolution of the learned size grid |
| `config.contour_area` | motion config | e.g. 100 | Min contour area to emit a motion box at all |
| `config.lightning_threshold` | motion config | e.g. 0.8 | Global-motion fraction that forces recalibration |
| `config.skip_motion_threshold` | motion config | `None` | Global-motion fraction that drops the frame entirely |
| `detect.stationary.interval` / `.threshold` | detect config | - | How often a motionless track is re-run through the model, and when it stops |

---

## Worked example (the "several small regions" case)

Several small motion blobs near the center of frame, each `~40x40px`:

1. Each blob passes `config.contour_area` and becomes a box
   (`improved_motion.py:142-155`).
2. `get_cluster_candidates` seeds a cluster on the first blob. The
   `get_cluster_boundary` for a 40x40 blob is `max(320, sqrt(40*40/0.1)) =
   max(320, 126) = 320` px, so any other blob within ~`320 - 0.55*40` px of
   the center joins the cluster.
3. The 5% rule passes (all blobs are comparable and `>= 5%` of a 320px
   region).
4. `get_cluster_region` squares the union: `longest_edge * 1.35`, floored at
   320 -> one **320x320** square centered on the group.
5. That single square is cropped, resized to the model input, and run **once**.

Result: N small blobs -> **1** inference, not N.

---

## Implications for the harness

`bgs_harness/harness.py` currently sends **one ROI per contour** to YOLO.
The Frigate approach to the "many small regions" problem is:

1. **Cluster** nearby boxes (`get_cluster_candidates` + `get_cluster_boundary`
   + 5% rule).
2. Build **one padded square per cluster** (`calculate_region`, `>= min_region`,
   `x1.35`).
3. **Send one crop per cluster** to the detector.
4. **Throttle** stationary targets and **skip** global-motion frames.

The direct, high-value change would be adding a Frigate-style
cluster/merge step after contour extraction (before the ROI list is finalized),
gated behind a config flag, so each YOLO call receives a merged square region
rather than every contour.
