"""Evaluate labeled JSONL predictions with bounded-memory binary metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

BATCH_SIZE = 5000


def auc_from_disk(db: sqlite3.Connection, positives: int, negatives: int) -> float | None:
    """Compute exact ROC-AUC by rank while keeping only one tied-score group in RAM."""
    if not positives or not negatives:
        return None
    positive_rank_sum = 0.0
    seen = 0
    pending_score: float | None = None
    tied_positive = tied_count = 0
    for label, score in db.execute("SELECT label, score FROM predictions ORDER BY score"):
        score = float(score)
        if pending_score is not None and score != pending_score:
            average_rank = seen + (tied_count + 1) / 2
            positive_rank_sum += tied_positive * average_rank
            seen += tied_count
            tied_positive = tied_count = 0
        pending_score = score
        tied_count += 1
        tied_positive += int(label == 1)
    if pending_score is not None:
        average_rank = seen + (tied_count + 1) / 2
        positive_rank_sum += tied_positive * average_rank
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate(predictions: Path, threshold: float) -> dict:
    tp = fp = tn = fn = positives = negatives = count = 0
    with tempfile.TemporaryDirectory(prefix="detector-evaluate-", dir=predictions.parent) as temporary:
        db = sqlite3.connect(Path(temporary) / "scores.sqlite")
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE predictions(label INTEGER NOT NULL, score REAL NOT NULL)")
            pending: list[tuple[int, float]] = []
            with predictions.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Record {line_number} is invalid JSON: {exc.msg}") from exc
                    if not isinstance(row, dict) or row.get("label") not in {"real", "synthetic"}:
                        raise ValueError(f"Record {line_number} must have label real or synthetic")
                    raw_score = row.get("score_synthetic")
                    if isinstance(raw_score, bool):
                        raise ValueError(f"Record {line_number} score_synthetic must be a number")
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Record {line_number} score_synthetic must be a number") from exc
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Record {line_number} score_synthetic must be finite and within [0,1]")
                    label = int(row["label"] == "synthetic")
                    count += 1
                    positives += label
                    negatives += 1 - label
                    if score >= threshold:
                        if label:
                            tp += 1
                        else:
                            fp += 1
                    elif label:
                        fn += 1
                    else:
                        tn += 1
                    pending.append((label, score))
                    if len(pending) >= BATCH_SIZE:
                        db.executemany("INSERT INTO predictions VALUES (?, ?)", pending)
                        db.commit()
                        pending.clear()
            if pending:
                db.executemany("INSERT INTO predictions VALUES (?, ?)", pending)
            db.commit()
            if count == 0:
                raise ValueError("Prediction file contains no records")
            div = lambda numerator, denominator: numerator / denominator if denominator else None
            recall_pos, recall_neg = div(tp, tp + fn), div(tn, tn + fp)
            return {
                "count": count,
                "class_counts": {"real": negatives, "synthetic": positives},
                "threshold": threshold,
                "confusion_matrix": {
                    "true_synthetic": tp,
                    "false_synthetic": fp,
                    "true_real": tn,
                    "missed_synthetic": fn,
                },
                "accuracy": (tp + tn) / count,
                "precision_synthetic": div(tp, tp + fp),
                "recall_synthetic": recall_pos,
                "recall_real": recall_neg,
                "balanced_accuracy": (recall_pos + recall_neg) / 2 if recall_pos is not None and recall_neg is not None else None,
                "roc_auc": auc_from_disk(db, positives, negatives),
                "note": "Metrics describe this labeled sample only; check group separation, label quality, and calibration.",
            }
        finally:
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL with label and score_synthetic")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--report", type=Path, help="Optional path for the JSON evaluation report")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite and between 0 and 1")
    predictions = args.predictions.expanduser().resolve()
    report_path = args.report.expanduser().resolve() if args.report else None
    if report_path and report_path == predictions:
        parser.error("--report cannot overwrite --predictions")
    try:
        report = evaluate(predictions, args.threshold)
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_report = report_path.with_name(report_path.name + ".tmp")
            temporary_report.write_text(rendered + "\n", encoding="utf-8")
            os.replace(temporary_report, report_path)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
