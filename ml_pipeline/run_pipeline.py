"""Run the local deepfake detection pipeline from media and metadata to a test report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n== {name} ==", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see the stage output above")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True, help="Folder containing local source media")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with per-file labels, permissions, and group IDs")
    parser.add_argument("--dataset-card", type=Path, required=True, help="Completed copy of DATASET_CARD_TEMPLATE.md")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for this complete run")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--split-seed", default="2026")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--inference-input", type=Path, help="Optional unseen image, video, or directory to score after training")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip decoding each prepared image during readiness checks")
    args = parser.parse_args()

    try:
        media_root = args.media_root.expanduser().resolve()
        metadata = args.metadata.expanduser().resolve()
        dataset_card = args.dataset_card.expanduser().resolve()
        inference_input = args.inference_input.expanduser().resolve() if args.inference_input else None
        output = args.output_dir.expanduser().resolve()
        if not media_root.is_dir():
            raise ValueError(f"Media root is not an existing directory: {media_root}")
        if not metadata.is_file():
            raise ValueError(f"Metadata CSV is not a file: {metadata}")
        if inference_input is not None and not (inference_input.is_file() or inference_input.is_dir()):
            raise ValueError(f"Inference input does not exist: {inference_input}")
        if inference_input is not None and (output in inference_input.parents or inference_input in output.parents or inference_input == output):
            raise ValueError("Output directory and inference input must be separate")
        if not dataset_card.is_file():
            raise ValueError(f"Dataset card is not a file: {dataset_card}")
        if output == media_root or output in media_root.parents or media_root in output.parents:
            raise ValueError("Output directory must be separate from the source media directory")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output directory must be new or empty: {output}")
        if not math.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 1:
            raise ValueError("fps must be finite and positive, and max-frames must be at least 1")
        if not math.isfinite(args.train_fraction) or not math.isfinite(args.validation_fraction):
            raise ValueError("split fractions must be finite")
        test_fraction = 1.0 - args.train_fraction - args.validation_fraction
        if min(args.train_fraction, args.validation_fraction, test_fraction) <= 0:
            raise ValueError("train and validation fractions must be positive and sum to less than 1")
        if min(args.epochs, args.batch_size, args.image_size) < 1 or args.workers < 0:
            raise ValueError("epochs, batch size, and image size must be positive; workers cannot be negative")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive")

        needs_ffmpeg = False
        with metadata.open(encoding="utf-8-sig", newline="") as metadata_stream:
            reader = csv.DictReader(metadata_stream)
            if not reader.fieldnames or "relative_path" not in reader.fieldnames:
                raise ValueError("Metadata CSV must include a relative_path column")
            needs_ffmpeg = any(Path(row.get("relative_path") or "").suffix.lower() in VIDEO_EXTENSIONS for row in reader)
        preflight = [sys.executable, str(ROOT / "baseline" / "check_environment.py")]
        preflight.extend(["--device", args.device])
        if needs_ffmpeg:
            preflight.append("--require-ffmpeg")
        run_stage("Runtime preflight", preflight)
        output.mkdir(parents=True, exist_ok=True)
        prepared = output / "derived_media"
        splits = output / "splits"
        baseline = output / "baseline"
        config = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {"media_root": str(media_root), "metadata_csv": str(metadata), "dataset_card": str(dataset_card), "inference_input": str(inference_input) if inference_input else None},
            "outputs": {"manifest": str(output / "manifest.jsonl"), "prepared_frames": str(prepared),
                        "splits": str(splits), "baseline": str(baseline)},
            "preprocessing": {"fps": args.fps, "max_frames": args.max_frames},
            "split": {"seed": args.split_seed, "train_fraction": args.train_fraction,
                      "validation_fraction": args.validation_fraction, "test_fraction": test_fraction},
            "training": {"epochs": args.epochs, "batch_size": args.batch_size, "image_size": args.image_size,
                         "learning_rate": args.learning_rate, "workers": args.workers, "device": args.device},
        }
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        manifest = output / "manifest.jsonl"
        run_stage("Ingestion", [sys.executable, str(ROOT / "data_ingestion" / "ingest.py"),
                                 "--input", str(media_root), "--metadata", str(metadata), "--output", str(manifest)])
        run_stage("Image and video preparation", [sys.executable, str(ROOT / "preprocessing" / "prepare_media.py"),
                                                   "--manifest", str(manifest), "--media-root", str(media_root),
                                                   "--output-dir", str(prepared), "--fps", str(args.fps),
                                                   "--max-frames", str(args.max_frames)])
        run_stage("Group-disjoint splitting", [sys.executable, str(ROOT / "splits" / "make_splits.py"),
                                                 "--manifest", str(prepared / "frames.jsonl"),
                                                 "--output-dir", str(splits), "--seed", str(args.split_seed),
                                                 "--train", str(args.train_fraction),
                                                 "--validation", str(args.validation_fraction)])
        readiness = [sys.executable, str(ROOT / "baseline" / "check_readiness.py"),
                     "--train-manifest", str(splits / "train.jsonl"),
                     "--validation-manifest", str(splits / "validation.jsonl"),
                     "--test-manifest", str(splits / "test.jsonl"), "--frames-root", str(prepared),
                     "--dataset-card", str(dataset_card), "--report", str(output / "readiness.json")]
        if args.skip_image_check:
            readiness.append("--skip-image-check")
        run_stage("Pre-training readiness", readiness)
        run_stage("Training", [sys.executable, str(ROOT / "baseline" / "train.py"),
                                "--train-manifest", str(splits / "train.jsonl"),
                                "--validation-manifest", str(splits / "validation.jsonl"),
                                "--test-manifest", str(splits / "test.jsonl"), "--frames-root", str(prepared),
                                "--output-dir", str(baseline), "--epochs", str(args.epochs),
                                "--batch-size", str(args.batch_size), "--image-size", str(args.image_size),
                                "--learning-rate", str(args.learning_rate), "--workers", str(args.workers),
                                "--device", args.device, "--defer-test-evaluation"])
        threshold_path = output / "threshold.json"
        run_stage("Validation threshold selection", [sys.executable, str(ROOT / "baseline" / "select_threshold.py"),
                                                       "--predictions", str(baseline / "validation_predictions.jsonl"),
                                                       "--report", str(threshold_path)])
        threshold_report = json.loads(threshold_path.read_text(encoding="utf-8"))
        selected_threshold = float(threshold_report["selected_threshold"])
        if not math.isfinite(selected_threshold) or not 0 <= selected_threshold <= 1:
            raise ValueError("Threshold selection produced a value outside [0,1]")
        run_stage("Held-out test evaluation", [sys.executable, str(ROOT / "baseline" / "evaluate.py"),
                                                 "--predictions", str(baseline / "test_predictions.jsonl"),
                                                 "--threshold", str(selected_threshold),
                                                 "--report", str(output / "evaluation.json")])
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if inference_input is not None:
        try:
            run_stage("Inference", [sys.executable, str(ROOT / "baseline" / "infer.py"),
                                    "--input", str(inference_input), "--checkpoint", str(baseline / "best_model.pt"),
                                    "--output", str(output / "inference_predictions.jsonl"),
                                    "--fps", str(args.fps), "--max-frames", str(args.max_frames),
                                    "--batch-size", str(args.batch_size), "--device", args.device])
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    print(f"\nPipeline completed. Run artifacts: {output}")
    print(f"Selected validation threshold: {selected_threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
