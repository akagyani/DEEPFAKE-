"""Run the local audio-visual pipeline from reviewed media metadata to held-out evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def run_stage(name: str, command: list[str]) -> None:
    print(f"\n== {name} ==", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed (exit {result.returncode}); inspect the stage output")


def manifest_signature(path: Path, *, ingested: bool, include_group: bool = True) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cached manifest JSON at {path}:{line_no}: {exc}") from exc
            if ingested and str(row.get("media_type", "")).lower() not in {"mp4", "mov", "mkv", "avi", "webm"}:
                continue
            fields = (row.get("relative_path"), str(row.get("sha256") or "").lower(), row.get("label"))
            if include_group:
                fields += (str(row.get("group_id") or ""),)
            digest.update(json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def rebase_cached_av_manifest(current_manifest: Path, cached_av_manifest: Path, output: Path) -> int:
    """Apply reviewed current group IDs to validated cached AV feature records."""
    current: dict[str, tuple[str, str, str]] = {}
    with current_manifest.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("media_type", "")).lower() not in {ext.lstrip(".") for ext in VIDEO_EXTENSIONS}:
                continue
            relative = str(row.get("relative_path") or "")
            if not relative or relative in current:
                raise ValueError(f"Current manifest has an empty or duplicate video path at line {line_no}")
            current[relative] = (str(row.get("sha256") or "").lower(), str(row.get("label") or ""),
                                 str(row.get("group_id") or ""))

    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with cached_av_manifest.open(encoding="utf-8") as source, output.open("w", encoding="utf-8", newline="\n") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            relative = str(row.get("relative_path") or "")
            expected = current.get(relative)
            if expected is None or relative in seen:
                raise ValueError(f"Cached AV manifest has an unexpected or duplicate video path at line {line_no}")
            actual = (str(row.get("sha256") or "").lower(), str(row.get("label") or ""))
            if actual != expected[:2] or not expected[2]:
                raise ValueError(f"Cached AV content/label does not match current media metadata for {relative}")
            row["group_id"] = expected[2]
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(relative)
    if seen != current.keys():
        missing = len(current.keys() - seen)
        raise ValueError(f"Cached AV manifest does not cover {missing} current video record(s)")
    return len(seen)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--dataset-card", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reuse-preprocessed-from", type=Path,
                   help="Reuse a prior run's validated manifest/features and rerun splitting, training, and evaluation")
    p.add_argument("--group-cv-folds", type=int, default=0,
                   help="Run source-group-stratified cross-validation (0 keeps the single holdout workflow; use at least 3 folds)")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--train-fraction", type=float, default=.70)
    p.add_argument("--validation-fraction", type=float, default=.15)
    p.add_argument("--audio-seconds", type=float, default=8.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--pretrained-visual", action="store_true", help="Initialize the visual ResNet18 from ImageNet weights")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--inference-input", type=Path, help="Optional single image-independent video to score after training")
    args = p.parse_args()
    try:
        media_root, metadata = args.media_root.expanduser().resolve(), args.metadata.expanduser().resolve()
        card = args.dataset_card.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        reuse_from = args.reuse_preprocessed_from.expanduser().resolve() if args.reuse_preprocessed_from else None
        inference = args.inference_input.expanduser().resolve() if args.inference_input else None
        if not media_root.is_dir() or not metadata.is_file() or not card.is_file():
            raise ValueError("media-root, metadata CSV, and dataset-card file must exist")
        if output == media_root or output in media_root.parents or media_root in output.parents:
            raise ValueError("output directory must be separate from source media")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be new or empty: {output}")
        if inference and (not inference.is_file() or inference.suffix.lower() not in VIDEO_EXTENSIONS):
            raise ValueError("inference-input must be an existing supported video file")
        if inference and (output == inference or output in inference.parents or inference in output.parents):
            raise ValueError("output directory and inference input must be separate")
        test_fraction = 1 - args.train_fraction - args.validation_fraction
        if not all(math.isfinite(x) for x in (test_fraction, args.train_fraction, args.validation_fraction)) or min(test_fraction, args.train_fraction, args.validation_fraction) <= 0:
            raise ValueError("train/validation fractions must be positive and sum to less than one")
        if args.audio_seconds <= 0 or min(args.epochs, args.batch_size, args.image_size) < 1 or args.workers < 0 or args.learning_rate <= 0:
            raise ValueError("audio duration, epochs, batch size, image size, and learning rate must be positive; workers cannot be negative")
        if args.group_cv_folds not in (0,) and args.group_cv_folds < 3:
            raise ValueError("group-cv-folds must be 0 or at least 3")
        if args.group_cv_folds and inference:
            raise ValueError("inference-input cannot be combined with group cross-validation; train a final inference checkpoint separately")
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "relative_path" not in reader.fieldnames:
                raise ValueError("metadata CSV must include relative_path")
            if not any(Path(row.get("relative_path") or "").suffix.lower() in VIDEO_EXTENSIONS for row in reader):
                raise ValueError("metadata CSV contains no supported video files; AV preprocessing ignores still images")
        output.mkdir(parents=True, exist_ok=True)
        manifest = (reuse_from / "manifest.jsonl") if reuse_from else (output / "manifest.jsonl")
        av_root = (reuse_from / "derived_av") if reuse_from else (output / "derived_av")
        split_manifest = av_root / "av_manifest.jsonl"
        if reuse_from:
            previous_config_path = reuse_from / "run_config.json"
            if not previous_config_path.is_file() or not manifest.is_file() or not (av_root / "av_manifest.jsonl").is_file():
                raise ValueError(f"Reusable preprocessing is incomplete: {reuse_from}")
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
            previous_seconds = previous_config.get("configuration", {}).get("audio_seconds")
            if previous_seconds != args.audio_seconds:
                raise ValueError("Cannot reuse preprocessing: audio-seconds differs from the cached feature settings")
            if manifest_signature(manifest, ingested=True) != manifest_signature(av_root / "av_manifest.jsonl", ingested=False):
                raise ValueError("Cannot reuse preprocessing: cached AV manifest does not match the ingestion manifest")
            # Colab often extracts the same Drive archive under a fresh, run-specific
            # /content path. Absolute paths therefore cannot establish input identity.
            # Re-ingest the current media + metadata (including current authorization
            # checks and SHA-256 hashes), then compare semantic manifest signatures to
            # the cached AV features before reusing them. This costs a file scan/hash,
            # but avoids expensive video/audio feature extraction and fails closed if
            # the files or labels changed. Group IDs may legitimately be corrected
            # independently of the media (for example, reviewed Vid-N source aliases).
            manifest = output / "manifest.jsonl"
            run_stage("Revalidate current media and metadata", [sys.executable,
                      str(ROOT / "data_ingestion" / "ingest.py"), "--input", str(media_root),
                      "--metadata", str(metadata), "--output", str(manifest)])
            if manifest_signature(manifest, ingested=True, include_group=False) != manifest_signature(av_root / "av_manifest.jsonl", ingested=False, include_group=False):
                raise ValueError("Cannot reuse preprocessing: current media/labels differ from the cached feature inventory")
            split_manifest = output / "av_manifest_rebased.jsonl"
            rebased_count = rebase_cached_av_manifest(manifest, av_root / "av_manifest.jsonl", split_manifest)
            print(f"Rebased current reviewed group IDs onto {rebased_count} validated cached AV records.", flush=True)
        else:
            manifest = output / "manifest.jsonl"
            av_root = output / "derived_av"
        splits = output / "splits"
        model_out = output / "multimodal"
        config = {"schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "inputs": {"media_root": str(media_root), "metadata_csv": str(metadata), "dataset_card": str(card),
                             "inference_input": str(inference) if inference else None},
                  "configuration": {"audio_seconds": args.audio_seconds, "split_seed": args.split_seed,
                                    "train_fraction": args.train_fraction, "validation_fraction": args.validation_fraction,
                                    "test_fraction": test_fraction, "epochs": args.epochs, "batch_size": args.batch_size,
                                    "image_size": args.image_size, "learning_rate": args.learning_rate,
                                    "pretrained_visual": args.pretrained_visual,
                                    "workers": args.workers, "device": args.device},
                  "artifacts": {"av_root": str(av_root), "splits": str(splits), "model": str(model_out)}}
        if reuse_from:
            config["preprocessing_reused_from"] = str(reuse_from)
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        python = sys.executable
        run_stage("Runtime preflight", [python, str(ROOT / "baseline" / "check_environment.py"), "--require-ffmpeg", "--device", args.device])
        if reuse_from:
            print(f"\n== Reusing validated preprocessing ==\nManifest: {manifest}\nAV features: {av_root}", flush=True)
        else:
            run_stage("Authorization-aware ingestion", [python, str(ROOT / "data_ingestion" / "ingest.py"), "--input", str(media_root),
                                                   "--metadata", str(metadata), "--output", str(manifest)])
            run_stage("Audio-visual feature preparation", [python, str(ROOT / "multimodal" / "preprocess.py"),
                                                            "--manifest", str(manifest), "--media-root", str(media_root),
                                                            "--output", str(av_root), "--audio-seconds", str(args.audio_seconds)])
        if not split_manifest.is_file() or split_manifest.stat().st_size == 0:
            raise ValueError("AV preprocessing produced no video records")
        if args.group_cv_folds:
            cv_output = output / "group_cv"
            cv_command = [python, str(ROOT / "multimodal" / "group_cross_validation.py"),
                          "--manifest", str(split_manifest), "--av-root", str(av_root),
                          "--dataset-card", str(card), "--output-dir", str(cv_output),
                          "--folds", str(args.group_cv_folds), "--seed", str(args.split_seed),
                          "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                          "--image-size", str(args.image_size), "--learning-rate", str(args.learning_rate),
                          "--validation-fraction", str(args.validation_fraction),
                          "--workers", str(args.workers), "--device", args.device,
                          "--audio-seconds", str(args.audio_seconds)]
            if args.pretrained_visual:
                cv_command.append("--pretrained-visual")
            run_stage("Paired-source-group cross-validation", cv_command)
            cv_summary = cv_output / "group_cv_summary.json"
            if not cv_summary.is_file():
                raise ValueError("Cross-validation finished without a summary report")
            config["configuration"]["evaluation_mode"] = "stratified_paired_source_group_k_fold"
            config["configuration"]["group_cv_folds"] = args.group_cv_folds
            config["configuration"]["group_cv_validation_fraction"] = args.validation_fraction
            config["configuration"]["single_holdout_fractions_ignored_for_group_cv"] = True
            config["artifacts"]["group_cv_summary"] = str(cv_summary)
            (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"\nPaired-source-group cross-validation completed. Artifacts: {cv_output}")
            return 0
        run_stage("Group-disjoint split assignment", [python, str(ROOT / "splits" / "make_splits.py"),
                                                        "--manifest", str(split_manifest), "--output-dir", str(splits),
                                                        "--seed", str(args.split_seed), "--train", str(args.train_fraction),
                                                        "--validation", str(args.validation_fraction)])
        readiness_path = output / "av_readiness.json"
        run_stage("AV data readiness", [python, str(ROOT / "multimodal" / "check_readiness.py"),
                                         "--train-manifest", str(splits / "train.jsonl"),
                                         "--validation-manifest", str(splits / "validation.jsonl"),
                                         "--test-manifest", str(splits / "test.jsonl"), "--av-root", str(av_root),
                                         "--dataset-card", str(card), "--report", str(readiness_path)])
        training_command = [python, str(ROOT / "multimodal" / "train.py"),
                                                   "--train-manifest", str(splits / "train.jsonl"),
                                                   "--validation-manifest", str(splits / "validation.jsonl"),
                                                   "--test-manifest", str(splits / "test.jsonl"), "--av-root", str(av_root),
                                                   "--output-dir", str(model_out), "--epochs", str(args.epochs),
                                                   "--batch-size", str(args.batch_size), "--image-size", str(args.image_size),
                                                   "--learning-rate", str(args.learning_rate), "--workers", str(args.workers),
                                                   "--device", args.device, "--seed", str(args.split_seed),
                                                   "--audio-seconds", str(args.audio_seconds)]
        if args.pretrained_visual:
            training_command.append("--pretrained-visual")
        run_stage("Audio-visual model training", training_command)
        ablations = {}
        for branch in ("fused", "visual", "audio"):
            validation_predictions = model_out / f"validation_{branch}_predictions.jsonl"
            test_predictions = model_out / f"test_{branch}_predictions.jsonl"
            if branch == "audio":
                if not validation_predictions.is_file() or not test_predictions.is_file() or not test_predictions.stat().st_size:
                    print("\n== Audio-only evaluation ==\nSkipped: no audio-bearing test videos.", flush=True)
                    continue
                if _prediction_labels(validation_predictions) != {"real", "synthetic"}:
                    print("\n== Audio-only evaluation ==\nSkipped: audio-bearing validation set lacks one class, so no valid threshold can be selected.", flush=True)
                    continue
            threshold_path = output / f"threshold_{branch}.json"
            run_stage(f"{branch.title()} validation threshold selection", [python, str(ROOT / "baseline" / "select_threshold.py"),
                         "--predictions", str(validation_predictions), "--report", str(threshold_path)])
            threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["selected_threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"{branch} validation threshold is outside [0,1]")
            report_path = output / f"evaluation_{branch}.json"
            run_stage(f"{branch.title()} held-out test evaluation", [python, str(ROOT / "baseline" / "evaluate.py"),
                         "--predictions", str(test_predictions),
                         "--threshold", str(threshold), "--report", str(report_path)])
            ablations[branch] = {"threshold": str(threshold_path), "evaluation": str(report_path)}
        config["artifacts"]["branch_ablation_reports"] = ablations
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if inference:
            run_stage("Unlabeled video inference", [python, str(ROOT / "multimodal" / "infer_media.py"),
                                                     "--input", str(inference), "--checkpoint", str(model_out / "best_av_model.pt"),
                                                     "--output", str(output / "inference.json"), "--audio-seconds", str(args.audio_seconds),
                                                     "--image-size", str(args.image_size),
                                                     "--threshold-report", str(output / "threshold_fused.json"),
                                                     "--device", args.device])
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"\nAV pipeline completed. Artifacts: {output}")
    return 0


def _prediction_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                labels.add(json.loads(line).get("label"))
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
