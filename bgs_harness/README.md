# bgs-harness

Configurable Python benchmark harness that evaluates background subtraction
algorithms (via [pybgs](https://github.com/alexander-faber/pybgs)) on MP4 video
streams, measures ROI generation performance for a downstream YOLO
pre-processing stage, and renders side-by-side annotated MP4 outputs.


```
input_videos/  --make convert-->  processed_videos/  --make run-->  outputs/
```

## Setup

```bash
make setup
```

## Convert

```bash
make convert
```

Re-encodes every `*.mp4` / `*.mkv` in `input_videos/` to the stream format
defined by the `stream` section of `config.json` (default: h264, 1000 kbps,
1280x720, 5 fps) into `processed_videos/`. 

## Run

```bash
make run
```

Processes every clip in `processed_videos/` with every algorithm in
`config.json`. In `outputs/`:

- `<video>_<algorithm>_annotated.mp4` side-by-side: ROI boxes and cleaned binary foreground mask
- `metrics_summary.csv` / `metrics_summary.json` per (video, algorithm) run

## Metrics

| Metric | Meaning |
| --- | --- |
| `trigger_rate_pct` | % of frames producing at least one valid ROI |
| `avg_rois_per_frame` | Total bounding boxes generated per frame |
| `avg_frame_coverage_pct` | Mean share of frame pixels covered by the union of ROI boxes |
| `max_frame_coverage_pct` | Worst-case single-frame ROI coverage |
| `avg_bgs_latency_ms` | pybgs `apply()` time per frame |
| `avg_preprocess_ms` | Full preprocessing stage per frame (resize + BGS + morphology + contours) |

ROI boxes are filtered by contour area (`min_area` px) and rejected if they
cover more than `max_area_ratio` of the frame.

## Frigate motion detector

An extra "algorithm" benchmarks Frigate NVR's own motion/ROI pass.

### `frigate` config section

| Key | Meaning |
| --- | --- |
| `threshold` | pixel-difference threshold (Frigate default 30); higher = fewer, larger ROIs |
| `contour_area` | minimum motion blob area (Frigate default 10 at its stock 100px motion height); at native height use ~100-200 |
| `frame_alpha` | background learning rate; lower = slower adaptation, fewer false triggers |
| `improve_contrast` | percentile contrast stretch before differencing |
| `frame_height` | `null` = run motion at full processing height (apples-to-apples with pybgs); an integer (e.g. `100`) = Frigate's stock downscale-to-100px mode |
| `lightning_threshold` | fraction of frame that must change to trigger recalibration (improved variant) |

## Configuration (`config.json`)

### `stream`

Target format for the `make convert` step.

### `algorithms`

List of pybgs algorithm class names to benchmark.

### `downscale_width`

Target width (px) the frames are resized to before BGS.

### `max_frames`

Cap on frames processed per (video, algorithm) run. `0` = process the whole
video. 

### `morphology`

Applies a morphological OPEN (remove small speckles) then CLOSE (fill small
holes inside solid targets) to the raw BGS mask, before contour extraction.

- `enabled`: `false` skips both steps (saves ~1-3 ms/frame) but passes raw
  algorithm speckle straight to the contour filter. Expect more small ROIs
  unless `min_area` is raised. `true` is recommended for noisy outdoor scenes.
- `kernel_size` (square kernel side in px):
  - **Larger** (e.g. 9-13): scrubs bigger noise blobs and thins out small
    motion. Fewer false ROIs in wind/foliage scenes.
  - **Smaller** (e.g. 3): preserves fine/small motion but lets more salt-and-
    pepper noise survive to the contour stage; pairs well with a higher
    `min_area`.
- `iterations`: repeats open+close N times; each pass strengthens the effect.

### `contours`

Bounding-box acceptance filter applied to each detected contour.

- `min_area` (min contour area in processed px)
- `max_area_ratio` (max ROI box area as a fraction of the frame)
