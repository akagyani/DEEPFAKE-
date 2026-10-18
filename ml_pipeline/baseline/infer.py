"""Run local image/video detection and emit versioned JSONL predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model

VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def input_files(source: Path):
    if source.is_file():
        yield source
        return
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO | IMAGE:
            yield path


def load_frames(path: Path, temporary: Path, fps: float, max_frames: int) -> list[Path]:
    if path.suffix.lower() in IMAGE:
        return [path]
    pattern = temporary / "frame-%04d.jpg"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-q:v", "3", str(pattern),
    ], check=True, capture_output=True, text=True)
    return sorted(temporary.glob("frame-*.jpg"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="One local image/video or a folder of media")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="JSONL output path")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = p.parse_args()
    if args.fps <= 0 or min(args.max_frames, args.batch_size) < 1:
        p.error("fps, max-frames, and batch-size must be positive")
    source = args.input.expanduser().resolve()
    if not source.exists() or (source.is_file() and source.suffix.lower() not in VIDEO | IMAGE):
        p.error("input must be an existing supported image, video, or directory")
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_size = int(checkpoint["image_size"])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(tuple(checkpoint["mean"]), tuple(checkpoint["std"])),
    ])
    threshold = float(checkpoint.get("threshold", 0.5))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            for media in input_files(source):
                if media.suffix.lower() not in VIDEO | IMAGE:
                    continue
                with tempfile.TemporaryDirectory(prefix="detector-frames-") as tmp:
                    paths = load_frames(media, Path(tmp), args.fps, args.max_frames)
                    if not paths:
                        raise ValueError(f"No frames could be read from {media.name}")
                    frame_scores = []
                    for start in range(0, len(paths), args.batch_size):
                        batch_paths = paths[start:start + args.batch_size]
                        tensors = []
                        for frame_path in batch_paths:
                            with Image.open(frame_path) as image:
                                tensors.append(transform(image.convert("RGB")))
                        with torch.inference_mode():
                            scores = torch.sigmoid(model(torch.stack(tensors).to(device)).flatten()).cpu().tolist()
                        for offset, score in enumerate(scores):
                            frame_index = start + offset
                            frame_scores.append({
                                "frame_index": frame_index,
                                "time_seconds": (frame_index / args.fps) if media.suffix.lower() in VIDEO else None,
                                "score_synthetic": float(score),
                            })
                    media_score = sum(item["score_synthetic"] for item in frame_scores) / len(frame_scores)
                    prediction = {
                        "schema_version": "1.0",
                        "media_sha256": sha256_file(media),
                        "model_id": checkpoint.get("model_id", "unknown"),
                        "prediction": "synthetic" if media_score >= threshold else "real",
                        "score_synthetic": media_score,
                        "threshold": threshold,
                        "frame_scores": frame_scores,
                        "aggregation": "mean_frame_score",
                        "limitations": [
                            "Experimental frame-classifier baseline; not proof of authenticity.",
                            "Score is not calibrated unless documented by independent validation.",
                        ],
                    }
                    output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    count += 1
        if count == 0:
            raise ValueError("No supported image or video files were found")
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {count} prediction records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
