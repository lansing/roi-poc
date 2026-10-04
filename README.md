# roi-poc

Benchmarks background subtraction algorithms (pybgs + Frigate NVR motion detection) for ROI generation before a YOLO object detector.

## Usage

1. Put MP4 clips in `bgs_harness/input_videos/`.
2. To set up the project: `make -C bgs_harness setup`
3. To convert the clips to the eval stream format: `make -C bgs_harness convert`
4. To run the eval: `make -C bgs_harness run`

Results (annotated MP4s + metrics CSV/JSON) are written to `bgs_harness/outputs/`.
Algorithm and post-processing settings live in `bgs_harness/config.json`; see `bgs_harness/README.md` for details.
