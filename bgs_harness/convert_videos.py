import json
from fractions import Fraction
from pathlib import Path

import av


def load_config(config_path: str = "config.json") -> dict:
    """Loads the stream conversion settings from the config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def convert_video(
    video_path: Path, stream_cfg: dict, output_dir: Path
) -> dict:
    """Re-encodes a single input video to the configured stream format.

    Resizes to the configured resolution, resamples to the configured frame
    rate by dropping/duplicating frames (nearest-timestamp match), and
    encodes with the configured codec/bitrate. Audio is stripped.
    """
    out_path = output_dir / f"{video_path.stem}.mp4"
    width = int(stream_cfg["width"])
    height = int(stream_cfg["height"])
    fps = float(stream_cfg["fps"])
    fps_fraction = Fraction(fps).limit_denominator(10000)
    bit_rate = int(stream_cfg["bitrate_kbps"] * 1000)
    codec = stream_cfg.get("codec", "h264")

    in_container = av.open(str(video_path))
    in_stream = in_container.streams.video[0]
    in_fps = float(in_stream.average_rate or in_stream.guessed_rate or 30.0)

    out_container = av.open(str(out_path), mode="w")
    out_stream = out_container.add_stream(codec, rate=fps_fraction)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.bit_rate = bit_rate

    frames_in = 0
    frames_out = 0
    last_index = -1

    for frame in in_container.decode(video=0):
        frames_in += 1
        t = frame.time
        if t is None:
            t = frames_in / in_fps
        target_index = int(round(t * fps))
        if target_index <= last_index:
            continue
        last_index = target_index

        frame = frame.reformat(width=width, height=height, format="yuv420p")
        frame.time_base = 1 / fps_fraction
        frame.pts = target_index
        for packet in out_stream.encode(frame):
            out_container.mux(packet)
        frames_out += 1

    for packet in out_stream.encode(None):
        out_container.mux(packet)

    in_container.close()
    out_container.close()

    return {
        "input": video_path.name,
        "output": out_path.name,
        "source_fps": round(in_fps, 3),
        "frames_in": frames_in,
        "frames_out": frames_out,
        "codec": codec,
        "bitrate_kbps": stream_cfg["bitrate_kbps"],
        "resolution": f"{width}x{height}",
        "fps": fps,
    }


def main():
    config = load_config("config.json")
    stream_cfg = config["stream"]
    input_dir = Path("input_videos")
    output_dir = Path("processed_videos")
    output_dir.mkdir(exist_ok=True)

    video_files = sorted(
        set(input_dir.glob("*.mp4")) | set(input_dir.glob("*.mkv"))
    )

    if not video_files:
        print(f"No video files found in '{input_dir}'.")
        return

    print(
        f"Converting {len(video_files)} video(s) to {stream_cfg['codec']} "
        f"{stream_cfg['width']}x{stream_cfg['height']} @ {stream_cfg['fps']}fps, "
        f"{stream_cfg['bitrate_kbps']} kbps -> {output_dir}/"
    )

    for video_file in video_files:
        print(f"  -> {video_file.name} ...")
        result = convert_video(video_file, stream_cfg, output_dir)
        print(
            f"     {result['frames_in']} frames @ {result['source_fps']}fps "
            f"-> {result['frames_out']} frames @ {result['fps']}fps: "
            f"{result['output']}"
        )

    print("[DONE] Conversion complete.")


if __name__ == "__main__":
    main()
