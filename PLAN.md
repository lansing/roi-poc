Executive Summary Plan: pybgs Evaluation Harness
This plan outlines the architecture for a configurable Python-based benchmarking harness using pybgs (the Python wrapper for BGSLibrary). The harness evaluates background subtraction algorithms across your challenging MP4 video streams, measures ROI generation performance, and renders visual overlay outputs.
Key Goals & Metrics
Candidate Filtering: Evaluate 3–4 high-performing pybgs algorithms (e.g., SuBSENSE, LOBSTER, MOG2, and KDE) against baseline Naive Frame Difference.
Noise Suppression Efficiency: Quantify how effectively each algorithm suppresses dynamic background noise (foliage/wind, dappled shadows) compared to valid target detection (human/animal motion).
Trigger Reduction Metric: Measure the potential computation saved for the downstream YOLO model by tracking:
Trigger Rate: Percentage of total frames that produce at least one valid ROI.
ROI Multiplicity: Total bounding boxes generated per frame.
Frame Coverage %: Total bounding box pixel area sent to YOLO versus full frame area.
Processing Overhead: Milliseconds per frame for the pre-processing stage.


Project Structure & Setup

bgs_harness/
├── pyproject.toml         # Dependencies managed via uv
├── Makefile               # Automated commands for setup, pipeline execution, and cleanup
├── config.json            # Configuration file for algorithms & post-processing
├── harness.py             # Main pipeline and benchmarking harness
├── input_videos/          # Folder containing input MP4 video clips
│   ├── motion_target.mp4
│   ├── wind_trees.mp4
│   └── dappled_shadows.mp4
└── outputs/               # Auto-generated annotated MP4s and CSV/JSON metrics


[project]
name = "bgs-harness"
version = "0.1.0"
description = "Background subtraction benchmark harness for YOLO pre-processing using pybgs"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "opencv-python-headless>=4.8.0",
    "pybgs>=3.3.0",
    "numpy>=1.24.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"


config.json:

{
  "algorithms": [
    "StaticFrameDifference",
    "SuBSENSE",
    "LOBSTER",
    "WeightedMovingMean",
    "KDE"
  ],
  "downscale_width": 854,
  "morphology": {
    "enabled": true,
    "kernel_size": 5,
    "iterations": 1
  },
  "contours": {
    "min_area": 800,
    "max_area_ratio": 0.6
  }
}


example harness.py (just a starting point, not verified)

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pybgs


def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration settings for BGS algorithms and post-processing."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_bgs_algorithm(algo_name: str):
    """Instantiates a pybgs algorithm object dynamically by name."""
    try:
        # Get the class dynamically from the pybgs package
        algo_class = getattr(pybgs, algo_name)
        return algo_class()
    except AttributeError:
        raise ValueError(
            f"Algorithm '{algo_name}' is not supported by pybgs."
        )


def process_video(
    video_path: Path, algo_name: str, config: dict, output_dir: Path
) -> dict:
    """Processes a single video stream using a pybgs algorithm and returns evaluation metrics.

    Steps:
    1. Read and optionally downscale input frame.
    2. Compute foreground mask using pybgs.
    3. Apply morphological filtering (erode/dilate) to scrub high-frequency noise.
    4. Extract contours and filter bounding boxes by area threshold.
    5. Render side-by-side composite frame (Original + Bounding Boxes vs. Clean Mask).
    6. Collect execution stats (latency, frame trigger rate, ROI density).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return {}

    # Extract video parameters
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate target dimensions
    target_w = config.get("downscale_width", orig_w)
    aspect_ratio = orig_h / orig_w
    target_h = int(target_w * aspect_ratio)

    # Initialize pybgs instance
    bgs_algo = get_bgs_algorithm(algo_name)

    # Setup video writer (Side-by-side: original frame left, binary mask right -> width = target_w * 2)
    output_filename = (
        output_dir / f"{video_path.stem}_{algo_name}_annotated.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_filename), fourcc, fps, (target_w * 2, target_h)
    )

    # Prepare morphological kernel
    kernel_size = config["morphology"]["kernel_size"]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    # Metrics counters
    frames_processed = 0
    frames_triggered = 0
    total_rois = 0
    total_bgs_time_ms = 0.0
    total_roi_area_px = 0

    min_area = config["contours"]["min_area"]
    max_area = (target_w * target_h) * config["contours"]["max_area_ratio"]

    print(f"  -> Running {algo_name} on {video_path.name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1

        # Step 1: Rescale frame for speed and noise reduction
        resized_frame = cv2.resize(
            frame, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        # Step 2: Compute foreground mask with time tracking
        start_time = time.perf_counter()
        fg_mask = bgs_algo.apply(resized_frame)
        bgs_time = (time.perf_counter() - start_time) * 1000.0
        total_bgs_time_ms += bgs_time

        # Ensure mask is single channel uint8
        if fg_mask is None:
            continue
        if len(fg_mask.shape) == 3:
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)

        # Step 3: Morphological noise filtering
        if config["morphology"]["enabled"]:
            # Morphological OPENING: removes small noise dots
            cleaned_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=config["morphology"]["iterations"],
            )
            # Morphological CLOSING: closes gaps inside solid moving targets
            cleaned_mask = cv2.morphologyEx(
                cleaned_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=config["morphology"]["iterations"],
            )
        else:
            cleaned_mask = fg_mask

        # Step 4: Extract contours and bounding boxes
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_rois = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_rois.append((x, y, w, h))
                total_roi_area_px += w * h

        num_rois = len(frame_rois)
        if num_rois > 0:
            frames_triggered += 1
            total_rois += num_rois

        # Step 5: Render annotated visuals
        annotated_frame = resized_frame.copy()
        for x, y, w, h in frame_rois:
            # Draw green box for valid ROI
            cv2.rectangle(
                annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.putText(
                annotated_frame,
                "ROI -> YOLO",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        # Convert grayscale mask to 3-channel for side-by-side concatenation
        mask_3ch = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

        # Draw HUD info on frame
        hud_text = f"{algo_name} | Frame: {frames_processed} | ROIs: {num_rois} | Latency: {bgs_time:.1f}ms"
        cv2.putText(
            annotated_frame,
            hud_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        # Concatenate side-by-side: Left=Original+ROIs, Right=Binary BGS Mask
        composite_view = np.hstack((annotated_frame, mask_3ch))
        writer.write(composite_view)

    cap.release()
    writer.release()

    # Calculate final telemetry statistics
    avg_latency = (
        total_bgs_time_ms / frames_processed if frames_processed > 0 else 0
    )
    trigger_rate = (
        (frames_triggered / frames_processed) * 100
        if frames_processed > 0
        else 0
    )
    avg_rois_per_frame = (
        total_rois / frames_processed if frames_processed > 0 else 0
    )

    return {
        "video": video_path.name,
        "algorithm": algo_name,
        "total_frames": frames_processed,
        "frames_triggered": frames_triggered,
        "trigger_rate_pct": round(trigger_rate, 2),
        "total_rois_generated": total_rois,
        "avg_rois_per_frame": round(avg_rois_per_frame, 2),
        "avg_bgs_latency_ms": round(avg_latency, 2),
        "output_video": output_filename.name,
    }


def main():
    config = load_config("config.json")
    input_dir = Path("input_videos")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    video_files = list(input_dir.glob("*.mp4")) + list(
        input_dir.glob("*.mkv")
    )

    if not video_files:
        print(
            f"No video files found in '{input_dir}'. Please place test MP4 clips there."
        )
        return

    all_metrics = []

    print(
        f"Starting pybgs evaluation across {len(video_files)} video file(s)..."
    )

    for video_file in video_files:
        print(f"\nProcessing Video: {video_file.name}")
        for algo in config["algorithms"]:
            try:
                metrics = process_video(video_file, algo, config, output_dir)
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"  [ERROR] Failed running {algo} on {video_file.name}: {e}")

    # Export CSV summary report
    csv_path = output_dir / "metrics_summary.csv"
    if all_metrics:
        keys = all_metrics[0].keys()
        import csv

        with open(csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_metrics)

        print(f"\n[DONE] Benchmark complete! Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

----


sample makefile


.PHONY: setup run clean help

# Environment configuration
VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

help:
	@echo "Available commands:"
	@echo "  make setup   - Create virtualenv and install dependencies using uv"
	@echo "  make run     - Run the pybgs benchmark pipeline"
	@echo "  make clean   - Remove generated output videos, logs, and venv"

# 1. Setup virtual environment and dependencies using uv
setup:
	@echo "Setting up environment using uv..."
	$(UV) venv $(VENV)
	$(UV) pip install -e .

# 2. Run the main evaluation harness
run:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtualenv not found. Running 'make setup' first..."; \
		make setup; \
	fi
	@echo "Running background subtraction harness..."
	$(PYTHON) harness.py

# 3. Clean generated artifacts
clean:
	@echo "Cleaning up generated files and virtual environment..."
	rm -rf outputs/*
	rm -rf $(VENV)
	rm -rf *.egg-info
	@echo "Cleanup complete."
