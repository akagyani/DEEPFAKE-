"""Frozen visual-only evaluation of SDFVD2 CV checkpoints on FF++ C23 frames."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += sum(label for _, label in ordered[i:j]) * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    predicted = [int(score >= threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predicted, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))
    pos, neg = sum(labels), len(labels) - sum(labels)
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    return {
        "video_count": len(labels),
        "class_counts": {"real": neg, "synthetic": pos},
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "roc_auc": auc(scores, labels),
        "threshold_from_sdfvd2_validation": threshold,
        "confusion_matrix": {"true_synthetic": tp, "false_synthetic": fp,
                             "true_real": tn, "missed_synthetic": fn},
    }


def locate_ffpp_root(root: Path) -> Path:
    if (root / "original_sequences").is_dir() and (root / "manipulated_sequences").is_dir():
        return root
    candidates = [p for p in root.rglob("original_sequences")
                  if p.is_dir() and (p.parent / "manipulated_sequences").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one FF++ root with original_sequences and manipulated_sequences; found {len(candidates)}")
    return candidates[0].parent


def sequence_frames(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTENSIONS),
                  key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.name.casefold()))


def sample_even(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    if count == 1:
        return [paths[(len(paths) - 1) // 2]]
    indices = sorted({round(i * (len(paths) - 1) / (count - 1)) for i in range(count)})
    return [paths[i] for i in indices]


def discover_sequences(root: Path, frames_per_video: int) -> tuple[Path, list[dict], dict]:
    ffpp = locate_ffpp_root(root.expanduser().resolve())
    real_frame_roots = list(ffpp.glob("original_sequences/**/c23/frames"))
    fake_frame_roots = list(ffpp.glob("manipulated_sequences/*/c23/frames"))
    if not real_frame_roots or not fake_frame_roots:
        raise ValueError("FF++ C23 frame folders missing; expected original_sequences/**/c23/frames and manipulated_sequences/*/c23/frames")
    sequences, skipped = [], []
    for frame_root, label, method in [(p, 0, "real") for p in real_frame_roots] + [
        (p, 1, p.parents[1].name) for p in fake_frame_roots
    ]:
        for folder in sorted(p for p in frame_root.iterdir() if p.is_dir()):
            frames = sequence_frames(folder)
            if not frames:
                skipped.append(folder.relative_to(ffpp).as_posix())
                continue
            sequences.append({"sequence": folder.relative_to(ffpp).as_posix(), "label": label,
                              "method": method, "frame_paths": sample_even(frames, frames_per_video),
                              "available_frame_count": len(frames)})
    seen = set()
    unique = []
    for row in sorted(sequences, key=lambda r: (r["label"], r["method"].casefold(), r["sequence"])):
        key = row["sequence"]
        if key not in seen:
            seen.add(key)
            unique.append(row)
    labels = [r["label"] for r in unique]
    if not unique or not any(labels) or all(labels):
        raise ValueError("FF++ inventory must include at least one real and one manipulated sequence")
    audit = {"real_sequences": sum(r["label"] == 0 for r in unique),
             "manipulated_sequences": sum(r["label"] == 1 for r in unique),
             "sequences_by_method": {}, "sequences_without_frames": len(skipped),
             "skipped_examples": skipped[:20], "sampled_frames_per_video_max": frames_per_video}
    for method in sorted({r["method"] for r in unique}, key=str.casefold):
        audit["sequences_by_method"][method] = sum(r["method"] == method for r in unique)
    return ffpp, unique, audit


class Frames(Dataset):
    def __init__(self, items: list[tuple[int, Path]], size: int):
        self.items = items
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sequence_index, path = self.items[i]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sequence_index


def folds(cv_root: Path) -> list[tuple[Path, Path, Path]]:
    result = []
    for fold_dir in sorted(cv_root.glob("fold_*")):
        checkpoint = fold_dir / "multimodal" / "best_av_model.pt"
        threshold = fold_dir / "threshold_visual.json"
        if checkpoint.is_file() and threshold.is_file():
            result.append((fold_dir, checkpoint, threshold))
    if len(result) != 5:
        raise ValueError(f"Expected exactly five complete SDFVD2 fold checkpoints under {cv_root}, found {len(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True, help="FF++ C23 root or extracted collection root")
    parser.add_argument("--cv-root", type=Path, required=True, help="SDFVD2 paired-source group_cv directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    try:
        cv_root = args.cv_root.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        if not cv_root.is_dir() or args.frames_per_video < 1 or args.batch_size < 1 or args.workers < 0:
            raise ValueError("Invalid checkpoint/data paths or nonpositive frame/batch settings")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Output directory must be new or empty: {output}")
        ffpp_root, sequences, audit = discover_sequences(args.frames_root, args.frames_per_video)
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but unavailable")
        device = torch.device(device_name)
        output.mkdir(parents=True, exist_ok=True)
        frame_items = [(si, frame) for si, seq in enumerate(sequences) for frame in seq["frame_paths"]]
        fold_reports, predictions = [], []
        for fold_dir, checkpoint_path, threshold_path in folds(cv_root):
            threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["selected_threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid visual threshold: {threshold_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(pretrained_visual=False).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            loader = DataLoader(Frames(frame_items, int(ckpt.get("image_size", 224))), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            per_sequence = [[] for _ in sequences]
            with torch.inference_mode():
                for images, seq_ids in loader:
                    logits = model.visual_head(model.visual(images.to(device))).squeeze(1)
                    scores = torch.sigmoid(logits).cpu().tolist()
                    for si, score in zip(seq_ids.tolist(), scores):
                        per_sequence[si].append(float(score))
            if any(not values for values in per_sequence):
                raise RuntimeError(f"Some videos received no scores in {fold_dir.name}")
            video_scores = [statistics.mean(values) for values in per_sequence]
            labels = [r["label"] for r in sequences]
            fold_metrics = {"fold": fold_dir.name, **metrics(video_scores, labels, threshold)}
            fold_reports.append(fold_metrics)
            for seq, score, frame_scores in zip(sequences, video_scores, per_sequence):
                predictions.append({"fold": fold_dir.name, "sequence": seq["sequence"], "method": seq["method"],
                                    "label": "synthetic" if seq["label"] else "real", "sampled_frame_count": len(frame_scores),
                                    "available_frame_count": seq["available_frame_count"], "score_synthetic": score,
                                    "threshold_from_sdfvd2_validation": threshold,
                                    "prediction": "synthetic" if score >= threshold else "real"})
            print(f"Scored {fold_dir.name}: {len(sequences)} videos", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_mean = {}
        for key in ("accuracy", "balanced_accuracy", "roc_auc"):
            vals = [r[key] for r in fold_reports if r[key] is not None]
            fold_mean[key] = {"mean": statistics.mean(vals), "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                              "fold_values": vals}
        report = {
            "schema_version": "1.0", "dataset": "Kaggle FF++ C23 (FaceForensics++ subtree only)",
            "evaluation": "frozen external visual evaluation; all inventoried FF++ C23 sequences; no training or threshold tuning",
            "training_performed": False, "score_unit": "video sequence; score is mean of deterministic evenly sampled frame scores",
            "ffpp_root": ffpp_root.name, "inventory": audit, "folds": fold_reports, "fold_mean": fold_mean,
            "limitations": [
                "Kaggle upload is a mixed collection; evaluation only scans the FaceForensics++ subtree and ignores other datasets.",
                "Kaggle labels and file provenance are not independently authenticated; compare inventory with official FF++ records before treating as benchmark-grade.",
                "Thresholds are each fold's SDFVD2 visual validation threshold; no threshold was fitted on FF++.",
                "Frame scores are averaged per video, but related originals and manipulations may share source identities; source independence is unverified.",
                "This is video-only visual evaluation and says nothing about the audio branch or fused multimodal accuracy.",
                "Use is subject to the original FaceForensics terms despite any Kaggle-level listing license."
            ]
        }
        (output / "external_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (output / "external_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(json.dumps({"inventory": audit, "fold_mean": fold_mean, "device": str(device)}, indent=2))
        print(f"Report: {output / 'external_evaluation.json'}")
        print(f"Predictions: {output / 'external_predictions.csv'}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
