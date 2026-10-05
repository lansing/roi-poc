import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pybgs

from cv2_bgss import CV2_ALGORITHMS
from frigate_adapter import FrigateMotionAlgorithm, VARIANTS
from frigate_preproc import FrigatePreprocessor


def load_config(config_path: str = "config.json") -> dict:
    """Loads configuration settings for BGS algorithms and post-processing."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_bgs_algorithm(algo_name: str):
    """Instantiates a pybgs algorithm object dynamically by name."""
    try:
        algo_class = getattr(pybgs, algo_name)
        return algo_class()
    except AttributeError:
        raise ValueError(
            f"Algorithm '{algo_name}' is not supported by pybgs."
        )


def roi_union_area_px(boxes: list, frame_w: int, frame_h: int) -> int:
    """Pixel area covered by the union of ROI boxes (overlaps counted once)."""
    if not boxes:
        return 0
    canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for x, y, w, h in boxes:
        x2 = min(x + w, frame_w)
        y2 = min(y + h, frame_h)
        canvas[y:y2, x:x2] = 1
    return int(cv2.countNonZero(canvas))


def process_video(
    video_path: Path,
    algo_name: str,
    config: dict,
    output_dir: Path,
    bgs_algo=None,
    write_video: bool = True,
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

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not fps or fps <= 0:
        fps = 30.0
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_w = config.get("downscale_width", orig_w)
    if target_w >= orig_w:
        target_w, target_h = orig_w, orig_h
    else:
        target_h = int(target_w * (orig_h / orig_w))
    target_h -= target_h % 2

    max_frames = int(config.get("max_frames", 0))

    is_frigate = algo_name in VARIANTS
    if is_frigate:
        bgs_algo = FrigateMotionAlgorithm(
            algo_name,
            (target_h, target_w),
            fps,
            config.get("frigate", {}),
        )
    elif bgs_algo is None:
        if algo_name in CV2_ALGORITHMS:
            bgs_algo = CV2_ALGORITHMS[algo_name]()
        else:
            bgs_algo = get_bgs_algorithm(algo_name)

    preproc = None
    if not is_frigate and config.get("frigate_preproc", {}).get("enabled"):
        preproc = FrigatePreprocessor(
            config["frigate_preproc"], target_w, target_h
        )

    output_filename = (
        output_dir / f"{video_path.stem}_{algo_name}_annotated.mp4"
    )
    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_filename), fourcc, fps, (target_w * 2, target_h)
        )

    morphology_cfg = config["morphology"]
    kernel = None
    if morphology_cfg["enabled"]:
        kernel_size = morphology_cfg["kernel_size"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

    frames_processed = 0
    frames_triggered = 0
    total_rois = 0
    total_bgs_time_ms = 0.0
    total_preprocess_time_ms = 0.0
    total_roi_area_px = 0
    max_frame_roi_area_px = 0

    frame_area_px = target_w * target_h
    min_area = config["contours"]["min_area"]
    max_area = frame_area_px * config["contours"]["max_area_ratio"]

    run_start = time.perf_counter()
    print(f"  -> Running {algo_name} on {video_path.name}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frames_processed >= max_frames:
            break

        frames_processed += 1

        preprocess_start = time.perf_counter()

        resized_frame = frame
        if (target_w, target_h) != (orig_w, orig_h):
            resized_frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

        if preproc is not None:
            resized_frame = preproc.preprocess(resized_frame)

        bgs_start = time.perf_counter()

        if is_frigate:
            frame_rois = bgs_algo.detect_boxes(
                cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            )
            frame_rois = [
                (x, y, w, h)
                for (x, y, w, h) in frame_rois
                if min_area <= w * h <= max_area
            ]
            cleaned_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            for x, y, w, h in frame_rois:
                cleaned_mask[y : y + h, x : x + w] = 255
        else:
            fg_mask = bgs_algo.apply(resized_frame)
            if fg_mask is None:
                bgs_time_ms = (time.perf_counter() - bgs_start) * 1000.0
                total_bgs_time_ms += bgs_time_ms
                continue
            if len(fg_mask.shape) == 3:
                fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
            if fg_mask.dtype != np.uint8:
                fg_mask = cv2.normalize(
                    fg_mask, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            if cv2.countNonZero(fg_mask) == 0 or fg_mask.max() < 255:
                fg_mask = cv2.threshold(
                    fg_mask, 127, 255, cv2.THRESH_BINARY
                )[1]

            if morphology_cfg["enabled"]:
                cleaned_mask = cv2.morphologyEx(
                    fg_mask,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=morphology_cfg["iterations"],
                )
                cleaned_mask = cv2.morphologyEx(
                    cleaned_mask,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=morphology_cfg["iterations"],
                )
            else:
                cleaned_mask = fg_mask

            contours, _ = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            frame_rois = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    frame_rois.append((x, y, w, h))

            if preproc is not None:
                frame_rois = preproc.filter_persistent(
                    frame_rois, target_w, target_h
                )

        bgs_time_ms = (time.perf_counter() - bgs_start) * 1000.0
        total_bgs_time_ms += bgs_time_ms

        frame_roi_area = roi_union_area_px(
            frame_rois, target_w, target_h
        )
        total_roi_area_px += frame_roi_area
        max_frame_roi_area_px = max(max_frame_roi_area_px, frame_roi_area)

        preprocess_time_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000.0
        total_preprocess_time_ms += preprocess_time_ms

        num_rois = len(frame_rois)
        if num_rois > 0:
            frames_triggered += 1
            total_rois += num_rois

        if write_video:
            annotated_frame = resized_frame.copy()
            for x, y, w, h in frame_rois:
                cv2.rectangle(
                    annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                )
                if h > 24:
                    cv2.putText(
                        annotated_frame,
                        "ROI -> YOLO",
                        (x + 2, max(y + 16, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                    )

            mask_3ch = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

            hud_text = (
                f"{algo_name} | F{frames_processed} | "
                f"ROIs: {num_rois} | {bgs_time_ms:.1f}ms"
            )
            cv2.putText(
                annotated_frame,
                hud_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            composite_view = np.hstack((annotated_frame, mask_3ch))
            writer.write(composite_view)

        if frames_processed % 100 == 0:
            total_hint = f"/{total_frames_hint}" if total_frames_hint else ""
            print(
                f"\r    {algo_name} {video_path.name}: "
                f"{frames_processed}{total_hint} frames, "
                f"avg bgs {total_bgs_time_ms / frames_processed:.1f}ms/frame",
                end="",
                flush=True,
            )

    cap.release()
    if writer is not None:
        writer.release()
    run_time_s = time.perf_counter() - run_start

    if frames_processed > 0:
        avg_bgs_ms = total_bgs_time_ms / frames_processed
        avg_preprocess_ms = total_preprocess_time_ms / frames_processed
        trigger_rate = (frames_triggered / frames_processed) * 100
        avg_rois_per_frame = total_rois / frames_processed
        avg_frame_coverage_pct = (
            total_roi_area_px / (frames_processed * frame_area_px) * 100
        )
        max_frame_coverage_pct = (
            max_frame_roi_area_px / frame_area_px * 100
        )
    else:
        avg_bgs_ms = avg_preprocess_ms = 0.0
        trigger_rate = 0.0
        avg_rois_per_frame = 0.0
        avg_frame_coverage_pct = 0.0
        max_frame_coverage_pct = 0.0

    print(
        f"\r    {algo_name} {video_path.name}: done "
        f"({frames_processed} frames, {run_time_s:.0f}s wall)\n"
    )

    return {
        "video": video_path.name,
        "algorithm": algo_name,
        "processed_resolution": f"{target_w}x{target_h}",
        "total_frames": frames_processed,
        "frames_triggered": frames_triggered,
        "trigger_rate_pct": round(trigger_rate, 2),
        "total_rois_generated": total_rois,
        "avg_rois_per_frame": round(avg_rois_per_frame, 2),
        "total_roi_union_area_px": total_roi_area_px,
        "avg_frame_coverage_pct": round(avg_frame_coverage_pct, 2),
        "max_frame_coverage_pct": round(max_frame_coverage_pct, 2),
        "avg_bgs_latency_ms": round(avg_bgs_ms, 2),
        "avg_preprocess_ms": round(avg_preprocess_ms, 2),
        "run_wall_time_s": round(run_time_s, 1),
        "output_video": output_filename.name if write_video else "",
    }


def print_summary_table(all_metrics: list) -> None:
    if not all_metrics:
        return
    cols = [
        "video",
        "algorithm",
        "trigger_rate_pct",
        "avg_rois_per_frame",
        "avg_frame_coverage_pct",
        "max_frame_coverage_pct",
        "avg_bgs_latency_ms",
        "avg_preprocess_ms",
    ]
    widths = {c: max(len(c), *(len(str(m[c])) for m in all_metrics)) for c in cols}
    header = " | ".join(c.rjust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for m in all_metrics:
        row = " | ".join(
            str(m[c]).rjust(widths[c]) for c in cols
        )
        print(row)


def main():
    ap = argparse.ArgumentParser(description="BGS benchmark harness.")
    ap.add_argument("--config", default="config.json", help="Config file.")
    ap.add_argument("--output-dir", default="outputs", help="Output directory.")
    ap.add_argument(
        "--preproc",
        nargs="*",
        default=None,
        help="Enable Frigate-inspired preprocessing for the non-Frigate BGS "
        "path. No value = all three (contrast, blur, persist). Pass a subset "
        "to ablate: 'contrast', 'blur', 'persist'.",
    )
    args = ap.parse_args()

    config = load_config(args.config)
    if args.preproc is not None:
        fpp = config.setdefault("frigate_preproc", {})
        fpp["enabled"] = True
        if args.preproc:
            techniques = set(args.preproc)
            fpp["contrast_norm"] = "contrast" in techniques
            fpp["gaussian_blur"] = "blur" in techniques
            if "persist" in techniques:
                if int(fpp.get("persistence_frames", 0)) < 1:
                    fpp["persistence_frames"] = 3
            else:
                fpp["persistence_frames"] = 0
        # else: no value -> run the recipe as configured in config.json
    input_dir = Path("processed_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(exist_ok=True)

    video_files = sorted(
        set(input_dir.glob("*.mp4")) | set(input_dir.glob("*.mkv"))
    )

    if not video_files:
        print(
            f"No video files found in '{input_dir}'. "
            "Place source clips in 'input_videos/' and run "
            "'make convert' first."
        )
        return

    all_metrics = []

    print(
        f"Starting pybgs evaluation across {len(video_files)} video "
        f"file(s) and {len(config['algorithms'])} algorithm(s)..."
    )

    for video_file in video_files:
        print(f"\nProcessing Video: {video_file.name}")
        for algo in config["algorithms"]:
            try:
                metrics = process_video(
                    video_file, algo, config, output_dir
                )
                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                print(
                    f"  [ERROR] Failed running {algo} on "
                    f"{video_file.name}: {e}"
                )

    if all_metrics:
        csv_path = output_dir / "metrics_summary.csv"
        json_path = output_dir / "metrics_summary.json"

        keys = all_metrics[0].keys()
        with open(csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_metrics)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "runs": all_metrics,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        print_summary_table(all_metrics)
        print(f"\n[DONE] Benchmark complete!")
        print(f"  CSV:  {csv_path}")
        print(f"  JSON: {json_path}")
    else:
        print("\n[WARN] No metrics collected.")


if __name__ == "__main__":
    main()
