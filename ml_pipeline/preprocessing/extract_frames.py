"""Extract a bounded number of local video frames using FFmpeg."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

VIDEO = {"mp4", "mov", "mkv", "avi", "webm"}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="JSONL from data_ingestion/ingest.py")
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Separate directory for derived frames")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=32)
    args = p.parse_args()
    if args.fps <= 0 or args.max_frames < 1:
        p.error("--fps must be positive and --max-frames at least 1")
    root, output = args.media_root.expanduser().resolve(), args.output.expanduser().resolve()
    if output == root or root in output.parents:
        p.error("Frame output must be outside the source media folder")
    if args.manifest.expanduser().resolve() == output / "frames.jsonl":
        p.error("Input manifest cannot be the frame index output path")
    output.mkdir(parents=True, exist_ok=True)
    out_manifest = output / "frames.jsonl"
    count = 0
    try:
        with args.manifest.open(encoding="utf-8") as src, out_manifest.open("w", encoding="utf-8") as dst:
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if str(item.get("media_type", "")).lower() not in VIDEO:
                    continue
                if item.get("label") not in {"real", "synthetic", "unknown"} or not str(item.get("group_id") or "").strip():
                    raise ValueError(f"Manifest line {line_no}: expected a supported label and group_id")
                digest = str(item.get("sha256") or "").lower().strip()
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(f"Manifest line {line_no}: expected a 64-character SHA-256 digest")
                rel = Path(item["relative_path"])
                media = (root / rel).resolve()
                if root not in media.parents or not media.is_file():
                    raise ValueError(f"Manifest line {line_no}: path is missing or outside media root")
                if sha256_file(media) != digest:
                    raise ValueError(f"Manifest line {line_no}: file content no longer matches its SHA-256")
                key = digest
                settings = hashlib.sha256(f"{args.fps:.8f}:{args.max_frames}".encode()).hexdigest()[:12]
                folder = output / f"{key}-{settings}"
                folder.mkdir(parents=True, exist_ok=True)
                pattern = folder / "frame-%04d.jpg"
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media),
                    "-vf", f"fps={args.fps}", "-frames:v", str(args.max_frames),
                    "-q:v", "3", str(pattern),
                ], check=True, capture_output=True, text=True)
                frames = sorted(folder.glob("frame-*.jpg"))
                if not frames:
                    print(f"No frames extracted: {rel}", file=sys.stderr)
                    continue
                dst.write(json.dumps({
                    "relative_path": rel.as_posix(), "sha256": item.get("sha256"),
                    "label": item.get("label"), "group_id": item.get("group_id"),
                    "frames": [f.relative_to(output).as_posix() for f in frames],
                }) + "\n")
                count += 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {count} video records and frames to {output}; index: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
