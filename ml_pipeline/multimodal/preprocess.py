"""Create one visual frame and a fixed-size log-mel audio view per video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int):
    import torch
    low = 0.0
    high = 2595.0 * math.log10(1.0 + (sample_rate / 2) / 700.0)
    points = torch.linspace(low, high, n_mels + 2)
    hz = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long().clamp(0, n_fft // 2)
    bank = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1].item(), bins[m].item(), bins[m + 1].item()
        if center > left:
            bank[m - 1, left:center] = torch.arange(left, center) / max(center - left, 1) - left / max(center - left, 1)
        if right > center:
            bank[m - 1, center:right] = (right - torch.arange(center, right)) / max(right - center, 1)
    return bank


def wav_to_logmel(wav_path: Path, seconds: float, n_mels: int = 64, target_frames: int = 256):
    import torch
    with wave.open(str(wav_path), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Expected 16-bit PCM audio from ffmpeg")
    signal = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    wanted = int(seconds * rate)
    signal = signal[:wanted]
    if signal.size < wanted:
        signal = np.pad(signal, (0, wanted - signal.size))
    tensor = torch.from_numpy(signal)
    n_fft, hop = 400, 160
    spec = torch.stft(tensor, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft), return_complex=True).abs().pow(2)
    mel = mel_filterbank(rate, n_fft, n_mels) @ spec
    logmel = torch.log(mel.clamp_min(1e-8))
    logmel = torch.nn.functional.interpolate(logmel[None, None], size=(n_mels, target_frames), mode="bilinear", align_corners=False)[0, 0]
    return logmel.numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Consent-aware ingestion JSONL")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Derived AV directory, outside source media")
    parser.add_argument("--audio-seconds", type=float, default=8.0)
    args = parser.parse_args()
    if not math.isfinite(args.audio_seconds) or args.audio_seconds <= 0:
        parser.error("--audio-seconds must be finite and positive")
    root, out = args.media_root.expanduser().resolve(), args.output.expanduser().resolve()
    if not root.is_dir() or out == root or root in out.parents:
        parser.error("media root must exist and output must be outside it")
    out.mkdir(parents=True, exist_ok=True)
    records = 0
    index = out / "av_manifest.jsonl"
    try:
        with args.manifest.open(encoding="utf-8") as src, index.open("w", encoding="utf-8", newline="\n") as dst:
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("media_type", "")).lower() not in {"mp4", "mov", "mkv", "avi", "webm"}:
                    continue
                label, group = row.get("label"), str(row.get("group_id") or "").strip()
                digest = str(row.get("sha256") or "").lower()
                if label not in {"real", "synthetic", "unknown"} or not group or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(f"Manifest line {line_no}: invalid label, group_id, or sha256")
                rel = Path(row["relative_path"])
                media = (root / rel).resolve()
                if root not in media.parents or not media.is_file() or sha256_file(media) != digest:
                    raise ValueError(f"Manifest line {line_no}: missing/outside media or SHA-256 mismatch")
                stem = digest
                frame = out / "frames" / f"{stem}.jpg"
                spec_path = out / "audio" / f"{stem}.npy"
                frame.parent.mkdir(parents=True, exist_ok=True)
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(media)], check=True, capture_output=True, text=True)
                duration = max(0.0, float(probe.stdout.strip()))
                at = max(0.0, duration / 2.0)
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.3f}", "-i", str(media), "-frames:v", "1", "-q:v", "3", str(frame)], check=True, capture_output=True)
                wav_path = out / "audio" / f"{stem}.wav"
                audio_proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media), "-t", str(args.audio_seconds), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)], capture_output=True)
                has_audio = audio_proc.returncode == 0 and wav_path.is_file() and wav_path.stat().st_size > 44
                if has_audio:
                    feature = wav_to_logmel(wav_path, args.audio_seconds)
                    wav_path.unlink(missing_ok=True)
                else:
                    import torch
                    feature = torch.zeros(64, 256, dtype=torch.float32).numpy()
                    wav_path.unlink(missing_ok=True)
                np.save(spec_path, feature)
                dst.write(json.dumps({"relative_path": rel.as_posix(), "sha256": digest, "label": label, "group_id": group,
                                      "frame": frame.relative_to(out).as_posix(), "log_mel": spec_path.relative_to(out).as_posix(),
                                      "audio_present": bool(has_audio), "audio_seconds": args.audio_seconds,
                                      "frame_timestamp_seconds": at}) + "\n")
                records += 1
                if records % 25 == 0:
                    print(f"Prepared {records} videos", flush=True)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Prepared {records} video records. AV manifest: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
