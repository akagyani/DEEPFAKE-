"""Prepare video and still-image records into one baseline-ready frame manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_stage(command: list[str], stage: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{stage} failed with exit code {exc.returncode}") from exc


def merge_manifests(output: Path) -> int:
    inputs = (("video_frames", output / "video_frames" / "frames.jsonl"),
              ("images", output / "images" / "frames.jsonl"))
    destination = output / "frames.jsonl"
    temporary: Path | None = None
    records = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".combined-frames-", suffix=".tmp",
            dir=output, delete=False,
        ) as stream_out:
            temporary = Path(stream_out.name)
            for prefix, manifest in inputs:
                if not manifest.is_file():
                    raise ValueError(f"Preprocessing stage did not create its manifest: {manifest}")
                with manifest.open(encoding="utf-8") as stream_in:
                    for line_number, line in enumerate(stream_in, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{manifest}, line {line_number}: invalid JSON") from exc
                        if not isinstance(row, dict) or not isinstance(row.get("frames"), list) or not row["frames"]:
                            raise ValueError(f"{manifest}, line {line_number}: expected a record with frame paths")
                        paths = []
                        for frame in row["frames"]:
                            if not isinstance(frame, str) or not frame.strip() or Path(frame).is_absolute():
                                raise ValueError(f"{manifest}, line {line_number}: invalid relative frame path")
                            paths.append((Path(prefix) / Path(frame)).as_posix())
                        row["frames"] = paths
                        stream_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        records += 1
        if records == 0:
            raise ValueError("The manifest contains no supported visual images or videos")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL produced by data_ingestion/ingest.py")
    parser.add_argument("--media-root", type=Path, required=True, help="Root folder containing original media")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for derived data")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling rate")
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum extracted frames per video")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 1:
        parser.error("--fps must be positive and --max-frames at least 1")

    try:
        manifest = args.manifest.expanduser().resolve()
        media_root = args.media_root.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        if not manifest.is_file():
            raise ValueError(f"Input manifest is not a file: {manifest}")
        if not media_root.is_dir():
            raise ValueError(f"Media root is not a directory: {media_root}")
        if output == media_root or output in media_root.parents or media_root in output.parents:
            raise ValueError("Output directory must be separate from the source media directory")
        if manifest == output / "frames.jsonl":
            raise ValueError("Input manifest cannot be the combined output manifest")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory must be empty to prevent overwriting existing data: {output}")
        output.mkdir(parents=True, exist_ok=True)

        run_stage([
            sys.executable, str(SCRIPT_DIR / "extract_frames.py"),
            "--manifest", str(manifest), "--media-root", str(media_root),
            "--output", str(output / "video_frames"),
            "--fps", str(args.fps), "--max-frames", str(args.max_frames),
        ], "Video frame extraction")
        run_stage([
            sys.executable, str(SCRIPT_DIR / "prepare_images.py"),
            "--manifest", str(manifest), "--media-root", str(media_root),
            "--output", str(output / "images"),
        ], "Still-image preparation")
        count = merge_manifests(output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote a combined frame manifest with {count} visual media records: {output / 'frames.jsonl'}")
    print(f"Use this directory as --frames-root for splitting and training: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
