"""Generic parameter-sweep driver for pybgs background-subtraction algorithms.

Reuses the exact per-frame pipeline from harness.py (downscale -> apply ->
morphology -> contours -> metrics) so results are directly comparable to the
main benchmark. Each parameter combination is run by:
  1. writing the algorithm's pybgs config XML (./config/<Algo>.xml),
  2. letting harness.process_video construct a FRESH algorithm object, which
     reads that file at construction (the pybgs ctor/dtor own the config file),
  3. running it over every input video and recording the metrics.

The sweep definition is a JSON file describing one algorithm and the parameter
space to explore. See sweeps/MixtureOfGaussianV2.json for an example.

Usage:
  python sweep.py --sweep sweeps/MixtureOfGaussianV2.json              # full sweep
  python sweep.py --sweep sweeps/MixtureOfGaussianV2.json --max-frames 30   # smoke test
"""

import argparse
import csv
import itertools
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pybgs

from harness import load_config, process_video


def write_algo_config(algo_name: str, params: dict, config_dir: Path) -> Path:
    """Write one parameter combination to the pybgs OpenCV FileStorage config."""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{algo_name}.xml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for key, value in params.items():
        fs.write(key, value)
    fs.release()
    return path


def build_combos(sweep: dict):
    """Expand the cartesian product of the 'params' section into a list of dicts.

    Each combo is the swept values plus the 'fixed' values (always applied).
    Returns (combos, swept_param_names).
    """
    swept_names = list(sweep["params"].keys())
    fixed = dict(sweep.get("fixed", {}))
    value_lists = [sweep["params"][name] for name in swept_names]
    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(swept_names, values))
        combo.update(fixed)
        combos.append(combo)
    return combos, swept_names


def print_sweep_table(rows: list, swept_names: list) -> None:
    """Compact table: video, the swept params, and the key outcome columns."""
    cols = (
        ["video"]
        + swept_names
        + ["trigger_rate_pct", "avg_frame_coverage_pct", "avg_bgs_latency_ms"]
    )
    if not rows:
        return
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.rjust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[c]).rjust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parameter sweep for pybgs background-subtraction algorithms."
    )
    ap.add_argument("--sweep", required=True, help="Path to the sweep definition JSON.")
    ap.add_argument("--config", default="config.json", help="Shared harness config.")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override max frames per video (small value = smoke test). "
        "Defaults to the config's max_frames (0 = full).",
    )
    ap.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional subset of video stems to run (e.g. motion_target wind_trees).",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/sweep",
        help="Where to write sweep_metrics.csv/json and annotated videos.",
    )
    ap.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep a per-combo annotated video. Default reuses one dir so only the "
        "last combo's videos remain (the CSV is the source of truth).",
    )
    args = ap.parse_args()

    with open(args.sweep, "r") as f:
        sweep = json.load(f)
    algo_name = sweep["algorithm"]

    if not hasattr(pybgs, algo_name):
        raise SystemExit(
            f"Algorithm '{algo_name}' is not a pybgs algorithm. Sweep files target "
            "pybgs algorithms (Frigate variants use config.json, not the XML sweep)."
        )

    config = load_config(args.config)
    if args.max_frames is not None:
        config["max_frames"] = args.max_frames

    input_dir = Path("processed_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path("config")

    video_files = sorted(set(input_dir.glob("*.mp4")) | set(input_dir.glob("*.mkv")))
    if not video_files:
        raise SystemExit(f"No videos in '{input_dir}'. Run 'make convert' first.")
    if args.videos:
        wanted = set(args.videos)
        video_files = [v for v in video_files if v.stem in wanted]
        if not video_files:
            raise SystemExit(
                f"None of --videos {sorted(wanted)} matched files in '{input_dir}'."
            )

    combos, swept_names = build_combos(sweep)
    fixed = dict(sweep.get("fixed", {}))
    param_columns = swept_names + list(fixed.keys())

    xml_path = config_dir / f"{algo_name}.xml"
    backup_path = None
    if xml_path.exists():
        backup_path = xml_path.with_suffix(".xml.sweep-bak")
        shutil.copyfile(xml_path, backup_path)

    print(f"Sweeping {algo_name}: {len(combos)} combo(s) x {len(video_files)} video(s).")
    print(f"  Swept params: {sweep['params']}")
    if fixed:
        print(f"  Fixed params: {fixed}")
    if args.max_frames is not None:
        print(f"  max_frames: {args.max_frames} (SMOKE TEST)")

    all_rows = []
    t0 = time.perf_counter()
    try:
        for i, combo in enumerate(combos, 1):
            write_algo_config(algo_name, combo, config_dir)
            video_dir = (
                output_dir / f"combo_{i:02d}" if args.keep_videos else output_dir
            )
            video_dir.mkdir(parents=True, exist_ok=True)
            swept_view = ", ".join(f"{k}={combo[k]}" for k in swept_names)
            print(f"\n[{i}/{len(combos)}] {algo_name}  {swept_view}")
            for video in video_files:
                row = process_video(video, algo_name, config, video_dir)
                if not row:
                    continue
                merged = {"video": row["video"], "algorithm": row["algorithm"]}
                for k in param_columns:
                    merged[k] = combo[k]
                for k, v in row.items():
                    if k not in merged:
                        merged[k] = v
                all_rows.append(merged)
    finally:
        if backup_path and backup_path.exists():
            shutil.copyfile(backup_path, xml_path)
            backup_path.unlink()
            print(f"\nRestored original {xml_path.name}.")

    if not all_rows:
        raise SystemExit("No metrics collected.")

    std_keys = [
        k
        for k in all_rows[0].keys()
        if k not in ("video", "algorithm") and k not in param_columns
    ]
    fieldnames = ["video", "algorithm"] + param_columns + std_keys

    csv_path = output_dir / "sweep_metrics.csv"
    json_path = output_dir / "sweep_metrics.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_file": args.sweep,
        "algorithm": algo_name,
        "swept_params": sweep["params"],
        "fixed_params": fixed,
        "max_frames": config.get("max_frames", 0),
        "combos": combos,
        "runs": all_rows,
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print_sweep_table(all_rows, swept_names)
    print(
        f"\n[DONE] {len(combos)} combos x {len(video_files)} videos in "
        f"{time.perf_counter() - t0:.0f}s."
    )
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
