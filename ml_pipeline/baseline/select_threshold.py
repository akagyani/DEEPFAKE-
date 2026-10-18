"""Select a classification threshold using labeled validation predictions only."""
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


def evaluate_counts(tp: int, fp: int, tn: int, fn: int, threshold: float, positives: int, negatives: int) -> dict:
    recall_synthetic = tp / positives if positives else None
    recall_real = tn / negatives if negatives else None
    balanced = (recall_synthetic + recall_real) / 2 if recall_synthetic is not None and recall_real is not None else None
    return {
        "threshold": threshold,
        "balanced_accuracy": balanced,
        "true_synthetic": tp,
        "false_synthetic": fp,
        "true_real": tn,
        "missed_synthetic": fn,
        "precision_synthetic": tp / (tp + fp) if tp + fp else None,
        "recall_synthetic": recall_synthetic,
        "recall_real": recall_real,
        "false_positive_rate": fp / negatives if negatives else None,
    }


def select_threshold(predictions: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="detector-threshold-", dir=predictions.parent) as temporary:
        db = sqlite3.connect(Path(temporary) / "validation-scores.sqlite")
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE predictions(label INTEGER NOT NULL, score REAL NOT NULL)")
            pending: list[tuple[int, float]] = []
            positives = negatives = count = 0
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
                    positives += label
                    negatives += 1 - label
                    count += 1
                    pending.append((label, score))
                    if len(pending) >= BATCH_SIZE:
                        db.executemany("INSERT INTO predictions VALUES (?, ?)", pending)
                        db.commit()
                        pending.clear()
            if pending:
                db.executemany("INSERT INTO predictions VALUES (?, ?)", pending)
            db.commit()
            if count == 0:
                raise ValueError("Validation prediction file contains no records")
            if not positives or not negatives:
                raise ValueError("Threshold selection requires both real and synthetic validation examples")

            best: dict | None = None
            best_key: tuple[float, float, float] | None = None

            def consider(threshold: float, tp: int, fp: int, tn: int, fn: int) -> None:
                nonlocal best, best_key
                metrics = evaluate_counts(tp, fp, tn, fn, threshold, positives, negatives)
                # Maximize balanced accuracy, then minimize false positives, then prefer the higher threshold.
                key = (metrics["balanced_accuracy"], -metrics["false_positive_rate"], threshold)
                if best_key is None or key > best_key:
                    best, best_key = metrics, key

            tp, fp, tn, fn = positives, negatives, 0, 0  # threshold 0 predicts every record as synthetic
            consider(0.0, tp, fp, tn, fn)
            current_score: float | None = None
            tied_positive = tied_negative = 0
            for label, score in db.execute("SELECT label, score FROM predictions ORDER BY score"):
                score = float(score)
                if current_score is not None and score != current_score:
                    consider(current_score, tp, fp, tn, fn)
                    tp -= tied_positive
                    fn += tied_positive
                    fp -= tied_negative
                    tn += tied_negative
                    tied_positive = tied_negative = 0
                current_score = score
                if label:
                    tied_positive += 1
                else:
                    tied_negative += 1
            if current_score is not None:
                consider(current_score, tp, fp, tn, fn)
                tp -= tied_positive
                fn += tied_positive
                fp -= tied_negative
                tn += tied_negative
                if current_score < 1.0:
                    consider(1.0, tp, fp, tn, fn)

            assert best is not None
            return {
                "schema_version": "1.0",
                "selection_metric": "balanced_accuracy",
                "tie_break": "minimize_false_positive_rate_then_choose_higher_threshold",
                "validation_count": count,
                "validation_class_counts": {"real": negatives, "synthetic": positives},
                "selected_threshold": best["threshold"],
                "validation_metrics_at_selected_threshold": best,
                "note": "Choose this threshold using validation predictions only; estimate final performance once on a separate held-out test set.",
            }
        finally:
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Labeled validation JSONL (label and score_synthetic)")
    parser.add_argument("--report", type=Path, help="Optional path for the JSON threshold report")
    args = parser.parse_args()
    predictions = args.predictions.expanduser().resolve()
    report_path = args.report.expanduser().resolve() if args.report else None
    if report_path and report_path == predictions:
        parser.error("--report cannot overwrite --predictions")
    try:
        report = select_threshold(predictions)
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
