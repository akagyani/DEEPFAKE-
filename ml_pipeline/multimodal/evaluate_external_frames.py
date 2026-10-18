"""Evaluate trained visual CV checkpoints on a labeled frame archive.

The evaluator treats each filename-derived video stem as one candidate unit and
uses one deterministic median-frame face crop per unit. This is provisional
when the archive's filename-to-source-video mapping has not been verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model

FRAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<frame>\d+)_face_(?P<face>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
LABEL_DIRS = {"real": 0, "deepfake": 1, "synthetic": 1, "fake": 1}
PUBLISHED_COUNTS = {"videos": 480, "real_videos": 30, "synthetic_videos": 450,
                    "real_frames": 3746, "synthetic_frames": 106948}


def auc(scores: list[float], labels: list[int]) -> float | None:
    positive, negative = sum(labels), len(labels) - sum(labels)
    if not positive or not negative:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank_sum += sum(label for _, label in ordered[index:end]) * ((index + 1 + end) / 2)
        index = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    pred = [int(score >= threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(pred, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(pred, labels))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    tpr = tp / positives if positives else None
    tnr = tn / negatives if negatives else None
    return {"count": len(labels), "class_counts": {"real": negatives, "synthetic": positives},
            "accuracy": (tp + tn) / len(labels) if labels else None,
            "balanced_accuracy": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
            "roc_auc": auc(scores, labels), "threshold_from_validation": threshold,
            "confusion_matrix": {"true_synthetic": tp, "false_synthetic": fp,
                                 "true_real": tn, "missed_synthetic": fn}}


def discover_records(root: Path) -> tuple[list[dict], dict]:
    """Find images beneath real/fake class folders and collapse by filename stem."""
    counts = {"real_frames": 0, "synthetic_frames": 0, "ignored_files": 0,
              "unparsed_images": 0, "candidate_video_units": 0}
    all_candidates: dict[tuple[int, str], list[tuple[int, int, Path]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        rel_parts = [part.lower() for part in path.relative_to(root).parts]
        label_name = next((part for part in rel_parts if part in LABEL_DIRS), None)
        if label_name is None:
            counts["ignored_files"] += 1
            continue
        label = LABEL_DIRS[label_name]
        counts["synthetic_frames" if label else "real_frames"] += 1
        match = FRAME_RE.match(path.name)
        if not match:
            counts["unparsed_images"] += 1
            continue
        video = match.group("video")
        frame, face = int(match.group("frame")), int(match.group("face"))
        key = (label, video)
        all_candidates.setdefault(key, []).append((frame, face, path))
    records = []
    for (label, video), candidates in sorted(all_candidates.items(), key=lambda item: (item[0][0], item[0][1])):
        candidates.sort(key=lambda row: (row[0], row[1], row[2].as_posix()))
        frame, face, path = candidates[(len(candidates) - 1) // 2]
        records.append({"candidate_video": video, "label": label, "frame_index": frame,
                        "face_index": face, "path": path})
    counts["candidate_video_units"] = len(records)
    return records, counts


class FrameDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def discover_folds(cv_root: Path) -> list[tuple[Path, Path, Path]]:
    folds = []
    for fold_dir in sorted(cv_root.glob("fold_*")):
        checkpoint = fold_dir / "multimodal" / "best_av_model.pt"
        threshold_path = fold_dir / "threshold_visual.json"
        if checkpoint.is_file() and threshold_path.is_file():
            folds.append((fold_dir, checkpoint, threshold_path))
    if len(folds) < 3:
        raise ValueError(f"Expected at least three fold checkpoints with validation thresholds under {cv_root}")
    return folds


def predict_fold(records: list[dict], loader: DataLoader, checkpoint_path: Path, device: torch.device) -> tuple[list[float], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model = build_model(pretrained_visual=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scores = [math.nan] * len(records)
    with torch.inference_mode():
        for images, indices in loader:
            logits = model.visual_head(model.visual(images.to(device))).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().tolist()
            for index, score in zip(indices.tolist(), probabilities):
                scores[index] = float(score)
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("Some external records were not scored")
    return scores, int(checkpoint.get("image_size", 224))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True,
                        help="Extracted archive directory containing real/ and deepfake/ image folders")
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="Prior five-fold CV folder containing fold_XX model checkpoints and validation thresholds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--mapping-reviewed", action="store_true",
                        help="Only set after independently verifying filename-to-source-video mapping and counts")
    args = parser.parse_args()
    try:
        root = args.frames_root.expanduser().resolve()
        cv_root = args.cv_root.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        if not root.is_dir() or not cv_root.is_dir():
            raise ValueError("frames-root and cv-root must be existing directories")
        if args.batch_size < 1 or args.workers < 0:
            raise ValueError("batch-size must be positive and workers nonnegative")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be new or empty: {output}")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but unavailable")
        device = torch.device(device_name)
        records, inventory = discover_records(root)
        if not records or not any(row["label"] == 0 for row in records) or not any(row["label"] == 1 for row in records):
            raise ValueError("Need at least one parsed candidate video in each class folder")
        folds = discover_folds(cv_root)
        output.mkdir(parents=True, exist_ok=True)
        fold_reports = []
        all_rows = []
        for fold_dir, checkpoint_path, threshold_path in folds:
            threshold_info = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold = float(threshold_info["selected_threshold"])
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid validation threshold in {threshold_path}")
            checkpoint_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            image_size = int(checkpoint_meta.get("image_size", 224))
            loader = DataLoader(FrameDataset(records, image_size), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            scores, _ = predict_fold(records, loader, checkpoint_path, device)
            labels = [row["label"] for row in records]
            report = {"fold": fold_dir.name, **metrics(scores, labels, threshold)}
            fold_reports.append(report)
            for row, score in zip(records, scores):
                all_rows.append({"fold": fold_dir.name, "candidate_video": row["candidate_video"],
                                 "label": "synthetic" if row["label"] else "real",
                                 "selected_frame": row["path"].relative_to(root).as_posix(),
                                 "frame_index": row["frame_index"], "face_index": row["face_index"],
                                 "score_synthetic": score, "threshold_from_validation": threshold,
                                 "prediction": "synthetic" if score >= threshold else "real"})
            print(f"Scored {fold_dir.name}: {report['count']} candidate videos; ROC AUC={report['roc_auc']}", flush=True)
        summary = {}
        for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
            values = [row[metric] for row in fold_reports if row[metric] is not None]
            summary[metric] = {"mean": statistics.mean(values) if values else None,
                               "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                               "fold_values": values}
        frame_counts_match = inventory["real_frames"] == PUBLISHED_COUNTS["real_frames"] and inventory["synthetic_frames"] == PUBLISHED_COUNTS["synthetic_frames"]
        videos_match = inventory["candidate_video_units"] == PUBLISHED_COUNTS["videos"]
        status = "reviewed_mapping" if args.mapping_reviewed and frame_counts_match and videos_match else "provisional_unreconciled_inventory_or_mapping"
        result = {"schema_version": "1.0", "dataset": "Comprehensive Deepfake Detection Dataset v3",
                  "external_evaluation_status": status, "run_device": str(device),
                  "aggregation_unit": "one deterministic median-index face crop per filename-derived candidate video stem",
                  "inventory_observed": inventory, "publisher_reported_counts": PUBLISHED_COUNTS,
                  "publisher_frame_counts_match": frame_counts_match, "publisher_video_count_match": videos_match,
                  "source_video_mapping_reviewed_by_operator": bool(args.mapping_reviewed),
                  "folds": fold_reports, "fold_mean": summary,
                  "limitations": [
                      "A filename-derived stem is only a candidate video unit; source-video lineage and subject overlap remain unverified.",
                      "Observed archive counts differ from the publisher-reported counts; this run evaluates only parsed files present in the supplied directory.",
                      "The external images are face crops and differ from the video-frame inputs used for SDFVD2 training.",
                      "Each fold checkpoint is reported separately; scores from independently trained folds are not pooled.",
                      "Thresholds are taken from each fold's SDFVD2 validation data and are not tuned on the external dataset.",
                  ]}
        (output / "external_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (output / "external_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        print(json.dumps({"status": status, "inventory": inventory, "fold_mean": summary}, indent=2))
        print(f"External evaluation artifacts: {output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
