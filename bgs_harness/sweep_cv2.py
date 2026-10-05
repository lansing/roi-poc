"""Parameter-sweep driver for the OpenCV MOG2 wrapper (cv2_bgss.OpenCVMOG2).

Unlike sweep.py (which targets pybgs algorithms via their XML config file), this
sweeps the constructor knobs of the cv2 MOG2 wrapper directly:
  - var_threshold : cv2 MOG2 variance threshold (default 16)
  - history       : background-model init/length in frames (default 500)
  - learning_rate : per-frame background adaptation rate (default 0.05)
  - shadow_mode   : fixed in the spec ("keep" or "background")

A FRESH OpenCVMOG2 is constructed for every (combo, video) pair and passed to
harness.process_video, so MOG2's internal background model never carries state
across videos or across combos. write_video=False skips annotated-video encoding
(the CSV/JSON metrics are the source of truth).

Usage:
  python sweep_cv2.py --sweep sweeps/OpenCVMOG2.json                # full sweep
  python sweep_cv2.py --sweep sweeps/OpenCVMOG2.json --max-frames 30 # smoke test
"""

import argparse
import csv
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cv2_bgss import CV2_ALGORITHMS, OpenCVMOG2
from harness import load_config, process_video

# Both registry names map to the same class; shadow_mode (in the spec's "fixed"
# section) is what distinguishes keep vs. no-shadow.
_CLASS_FOR = {
    "OpenCVMOG2": OpenCVMOG2,
    "OpenCVMOG2_noshadow": OpenCVMOG2,
}


def build_combos(sweep: dict):
    """Expand the cartesian product of 'params' + apply 'fixed' values.

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
        description="Parameter sweep for the OpenCV MOG2 (cv2) wrapper."
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
        default="outputs/sweep_cv2",
        help="Where to write sweep_metrics.csv/json.",
    )
    args = ap.parse_args()

    with open(args.sweep, "r") as f:
        sweep = json.load(f)
    algo_name = sweep["algorithm"]

    if algo_name not in CV2_ALGORITHMS:
        raise SystemExit(
            f"Algorithm '{algo_name}' is not an OpenCV cv2 algorithm. "
            f"Supported: {sorted(CV2_ALGORITHMS)}"
        )
    builder = _CLASS_FOR[algo_name]

    config = load_config(args.config)
    if args.max_frames is not None:
        config["max_frames"] = args.max_frames

    input_dir = Path("processed_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(
        f"Sweeping {algo_name}: {len(combos)} combo(s) x {len(video_files)} video(s)."
    )
    print(f"  Swept params: {sweep['params']}")
    if fixed:
        print(f"  Fixed params: {fixed}")
    if args.max_frames is not None:
        print(f"  max_frames: {args.max_frames} (SMOKE TEST)")

    all_rows = []
    t0 = time.perf_counter()
    for i, combo in enumerate(combos, 1):
        swept_view = ", ".join(f"{k}={combo[k]}" for k in swept_names)
        print(f"\n[{i}/{len(combos)}] {algo_name}  {swept_view}")
        for video in video_files:
            # Fresh MOG2 per (combo, video) so the background model never leaks
            # state across videos or across combos.
            bgs_algo = builder(**combo)
            row = process_video(
                video, algo_name, config, output_dir, bgs_algo=bgs_algo,
                write_video=False,
            )
            if not row:
                continue
            merged = {"video": row["video"], "algorithm": row["algorithm"]}
            for k in param_columns:
                merged[k] = combo[k]
            for k, v in row.items():
                if k not in merged:
                    merged[k] = v
            all_rows.append(merged)

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
