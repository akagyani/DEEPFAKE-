"""Evaluate SDFVD2 visual CV checkpoints on Kaggle FF++/Celeb-DF test frames.

This is a test-only external evaluation. It never tunes on the Kaggle data;
each fold uses the threshold selected on that fold's SDFVD2 validation set.
The score unit is an image, not an independent source video.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_DIRS = {"real": 0, "fake": 1, "deepfake": 1, "synthetic": 1}


def auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += sum(label for _, label in ordered[i:j]) * average_rank
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def score_metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    predicted = [int(score >= threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predicted, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    tpr = tp / positives if positives else None
    tnr = tn / negatives if negatives else None
    return {
        "count": len(labels),
        "class_counts": {"real": negatives, "synthetic": positives},
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "roc_auc": auc(scores, labels),
        "threshold_from_sdfvd2_validation": threshold,
        "confusion_matrix": {"true_synthetic": tp, "false_synthetic": fp,
                             "true_real": tn, "missed_synthetic": fn},
    }


def find_class(path: Path, root: Path) -> int | None:
    for part in reversed(path.relative_to(root).parts[:-1]):
        label = LABEL_DIRS.get(part.casefold())
        if label is not None:
            return label
    return None


def inventory_and_duplicate_audit(dataset_root: Path) -> tuple[dict, list[dict]]:
    """Count train/val/test by class and find exact duplicate image bytes across splits."""
    counts: dict[str, Counter] = defaultdict(Counter)
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unclassified = 0
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root)
        parts = [p.casefold() for p in rel.parts]
        split = next((p for p in parts if p in {"train", "training", "val", "valid", "validation", "test", "testing"}), None)
        label = find_class(path, dataset_root)
        if split is None or label is None:
            unclassified += 1
            continue
        split = {"training": "train", "valid": "val", "validation": "val", "testing": "test"}.get(split, split)
        class_name = "synthetic" if label else "real"
        counts[split][class_name] += 1
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[digest.hexdigest()].append((split, rel.as_posix()))
    duplicate_groups = []
    for digest, rows in hashes.items():
        splits = sorted({split for split, _ in rows})
        if len(splits) > 1:
            duplicate_groups.append({"sha256": digest, "splits": splits,
                                     "files": [path for _, path in rows]})
    report = {
        "counts_by_split": {split: dict(sorted(counts[split].items())) for split in sorted(counts)},
        "unclassified_image_files": unclassified,
        "exact_byte_duplicate_groups_crossing_splits": len(duplicate_groups),
        "cross_split_duplicate_examples": duplicate_groups[:100],
        "duplicate_scan_scope": "exact SHA-256 byte matches only; does not detect re-encoded frames or shared source identities",
    }
    return report, duplicate_groups


def discover_test_records(test_root: Path) -> list[dict]:
    records = []
    for path in sorted(test_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        label = find_class(path, test_root)
        if label is None:
            continue
        records.append({"path": path, "label": label, "relative_path": path.relative_to(test_root).as_posix()})
    return records


class ImageDataset(Dataset):
    def __init__(self, records: list[dict], size: int):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        with Image.open(self.records[idx]["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, idx


def discover_folds(cv_root: Path) -> list[tuple[Path, Path, Path]]:
    folds = []
    for fold_dir in sorted(cv_root.glob("fold_*")):
        checkpoint = fold_dir / "multimodal" / "best_av_model.pt"
        threshold_path = fold_dir / "threshold_visual.json"
        if checkpoint.is_file() and threshold_path.is_file():
            folds.append((fold_dir, checkpoint, threshold_path))
    if len(folds) < 3:
        raise ValueError(f"Expected at least three fold checkpoints and visual thresholds under {cv_root}")
    return folds


def predict(model, loader, device: torch.device, count: int) -> list[float]:
    result = [math.nan] * count
    with torch.inference_mode():
        for images, indices in loader:
            logits = model.visual_head(model.visual(images.to(device))).squeeze(1)
            for idx, value in zip(indices.tolist(), torch.sigmoid(logits).cpu().tolist()):
                result[idx] = float(value)
    if any(not math.isfinite(value) for value in result):
        raise RuntimeError("Not all test images received a finite score")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted Kaggle dataset root containing train/val/test")
    parser.add_argument("--test-root", type=Path, required=True, help="The test split directory only; e.g. dataset/test")
    parser.add_argument("--cv-root", type=Path, required=True, help="SDFVD2 group_cv directory with five fold checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    try:
        dataset_root, test_root, cv_root, output = (p.expanduser().resolve() for p in
            (args.dataset_root, args.test_root, args.cv_root, args.output_dir))
        if not dataset_root.is_dir() or not test_root.is_dir() or not cv_root.is_dir():
            raise ValueError("dataset-root, test-root, and cv-root must be existing directories")
        if args.batch_size < 1 or args.workers < 0:
            raise ValueError("batch-size must be positive and workers nonnegative")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be new or empty: {output}")
        try:
            test_root.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError("test-root must be inside dataset-root") from exc
        if not any(part.casefold() in {"test", "testing"} for part in test_root.parts):
            raise ValueError("Refusing evaluation: test-root path must explicitly identify the test split")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but unavailable")
        device = torch.device(device_name)

        inventory, duplicate_groups = inventory_and_duplicate_audit(dataset_root)
        records = discover_test_records(test_root)
        labels = [row["label"] for row in records]
        if not records or not any(labels) or all(labels):
            raise ValueError("Test split needs at least one image in both real and fake class folders")
        folds = discover_folds(cv_root)
        output.mkdir(parents=True, exist_ok=True)
        fold_reports, predictions = [], []
        for fold_dir, checkpoint_path, threshold_path in folds:
            threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(threshold_data["selected_threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid threshold in {threshold_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            image_size = int(checkpoint.get("image_size", 224))
            model = build_model(pretrained_visual=False).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            loader = DataLoader(ImageDataset(records, image_size), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            scores = predict(model, loader, device, len(records))
            report = {"fold": fold_dir.name, **score_metrics(scores, labels, threshold)}
            fold_reports.append(report)
            for row, score in zip(records, scores):
                predictions.append({"fold": fold_dir.name, "test_image": row["relative_path"],
                    "label": "synthetic" if row["label"] else "real", "score_synthetic": score,
                    "threshold_from_sdfvd2_validation": threshold,
                    "prediction": "synthetic" if score >= threshold else "real"})
            print(f"Scored {fold_dir.name}: {len(records)} test images; ROC AUC={report['roc_auc']}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary = {}
        for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
            values = [row[metric] for row in fold_reports if row[metric] is not None]
            summary[metric] = {"mean": statistics.mean(values) if values else None,
                               "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                               "fold_values": values}
        report = {
            "schema_version": "1.0",
            "dataset": "Kaggle deepfake-dataset (FaceForensics++ C23 and Celeb-DF v2 per publisher description)",
            "evaluation_split": "test only",
            "training_performed": False,
            "external_evaluation_status": "exploratory_test_split_with_unverified_upstream_lineage",
            "run_device": str(device),
            "score_unit": "individual face-frame image; frames may be correlated and are not independent videos",
            "inventory_and_exact_duplicate_audit": inventory,
            "folds": fold_reports,
            "fold_mean": summary,
            "limitations": [
                "The dataset publisher's split and source-video disjointness claims were not independently verified.",
                "The exact duplicate scan detects identical file bytes across splits, not re-encoded copies, near-duplicates, shared identities, or related frames.",
                "The Kaggle archive contains face-frame images; this visual-only evaluation does not assess audio or temporal video evidence.",
                "Metrics are image-weighted; multiple correlated frames may come from the same source video.",
                "No model was trained or calibrated on this dataset. Fold-specific thresholds come only from SDFVD2 validation data.",
            ],
        }
        (output / "external_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (output / "external_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(json.dumps({"status": report["external_evaluation_status"], "inventory": inventory,
                          "fold_mean": summary, "test_images": len(records)}, indent=2))
        print(f"External evaluation artifacts: {output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
