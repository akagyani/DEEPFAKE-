"""Score one unlabeled video with a trained or explicitly untrained AV checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from preprocess import wav_to_logmel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audio-seconds", type=float)
    p.add_argument("--image-size", type=int)
    p.add_argument("--threshold-report", type=Path,
                   help="Optional validation-selected threshold JSON from a single-holdout run")
    p.add_argument("--threshold", type=float,
                   help="Explicit decision threshold; use only if it was selected on validation data")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args()
    video, checkpoint = args.input.expanduser().resolve(), args.checkpoint.expanduser().resolve()
    if not video.is_file() or not checkpoint.is_file():
        p.error("input video and checkpoint must exist")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        p.error("ffmpeg and ffprobe must be installed and available on PATH")
    if args.threshold_report and args.threshold is not None:
        p.error("provide threshold-report or threshold, not both")
    threshold = args.threshold
    threshold_source = "explicit_validation_threshold" if threshold is not None else None
    if args.threshold_report:
        threshold_path = args.threshold_report.expanduser().resolve()
        try:
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(threshold_data["selected_threshold"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            p.error(f"could not read selected_threshold from threshold report: {exc}")
        threshold_source = str(args.threshold_report.expanduser().resolve())
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        p.error("threshold must be finite and within [0,1]")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available")
    if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        p.error("MPS was requested but is not available")
    device = torch.device(device_name)
    try:
        data = torch.load(checkpoint, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        p.error(f"could not load a safe model checkpoint: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("state_dict"), dict):
        p.error("checkpoint must contain a state_dict")
    if data.get("classes", {"real": 0, "synthetic": 1}) != {"real": 0, "synthetic": 1}:
        p.error("checkpoint class mapping is incompatible; expected real=0, synthetic=1")
    audio_seconds = args.audio_seconds if args.audio_seconds is not None else float(data.get("audio_seconds", 8.0))
    image_size = args.image_size if args.image_size is not None else int(data.get("image_size", 224))
    if not math.isfinite(audio_seconds) or audio_seconds <= 0 or image_size < 1:
        p.error("audio-seconds must be finite and positive; image-size must be positive")
    model = build_model().to(device); model.load_state_dict(data["state_dict"]); model.eval()
    with tempfile.TemporaryDirectory(prefix="av-inference-") as temporary:
        temp = Path(temporary); frame_path = temp / "frame.jpg"; wav_path = temp / "audio.wav"
        try:
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)], check=True, capture_output=True, text=True)
            duration = float(probe.stdout.strip())
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("video duration is invalid")
            timestamp = duration / 2.0
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(frame_path)], check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            p.error(f"could not decode a midpoint frame: {exc}")
        audio = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-t", str(audio_seconds), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)], capture_output=True)
        has_audio = audio.returncode == 0 and wav_path.is_file() and wav_path.stat().st_size > 44
        feature = wav_to_logmel(wav_path, audio_seconds) if has_audio else np.zeros((64, 256), dtype=np.float32)
        with Image.open(frame_path) as frame:
            image = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])(frame.convert("RGB"))
        with torch.no_grad():
            result = model(image.unsqueeze(0).to(device), torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device),
                           torch.tensor([float(has_audio)], device=device))
    trained = bool(data.get("trained", False))
    visual_score = float(torch.sigmoid(result["visual_logits"]).item())
    audio_score = float(torch.sigmoid(result["audio_logits"]).item()) if has_audio else None
    fused_score = float(torch.sigmoid(result["fused_logits"]).item())
    checkpoint_hash = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_stream:
        for block in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    report = {"schema_version": "1.0", "input": str(video), "model_trained": trained,
              "model_identifier": checkpoint_hash.hexdigest(),
              "device": str(device), "video_duration_seconds": duration,
              "frame_timestamp_seconds": timestamp, "audio_present": has_audio,
              "visual_synthetic_score": visual_score, "audio_synthetic_score": audio_score,
              "fused_synthetic_score": fused_score,
              "decision_threshold": threshold, "threshold_source": threshold_source,
              "predicted_label": ("synthetic" if fused_score >= threshold else "real") if trained and threshold is not None else None,
              "interpretation": "untrained random weights; score and label are meaningless" if not trained else "model score is uncertain evidence, not proof of authenticity or manipulation",
              "limitations": ["This model uses a midpoint frame and a short audio window; it does not analyze full temporal dynamics or crop faces.",
                              "A missing audio track makes audio_synthetic_score unavailable; any decision reflects the model's missing-audio path.",
                              "Scores are not calibrated probabilities. Use a validation-selected threshold and evaluate on independent source groups."]}
    output = args.output.expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
