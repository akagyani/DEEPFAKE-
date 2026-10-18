"""Run paired-source-group-stratified cross-validation on a prepared AV manifest.

Each outer fold holds out complete source groups. A validation subset is chosen
from the remaining groups for checkpoint selection and threshold calibration.
Fold scores are summarized separately; scores from different fold models are
not pooled because their calibration scales may differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"real", "synthetic"}
HASH_RE = re.compile(r"[0-9a-f]{64}")
AUGMENTED_RE = re.compile(r"_aug_\d+(?:[-_.\s]|$)", re.IGNORECASE)


def _read_manifest(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    seen_hashes: dict[str, tuple[str, str]] = {}
    duplicates = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid manifest JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_no} must be an object")
            label = row.get("label")
            group = str(row.get("group_id") or "").strip()
            digest = str(row.get("sha256") or "").lower()
            if label not in LABELS or not group or not HASH_RE.fullmatch(digest):
                raise ValueError(f"Manifest row {line_no} needs a binary label, group_id, and SHA-256")
            if not row.get("relative_path"):
                raise ValueError(f"Manifest row {line_no} has no relative_path")
            previous = seen_hashes.get(digest)
            if previous:
                duplicates += 1
                if previous[1] != label:
                    raise ValueError(f"Exact duplicate SHA-256 has conflicting labels: {digest}")
                # Ignore later copies. The grouping step remains conservative:
                # merging exact duplicates across groups is handled by rejecting
                # such ambiguity rather than allowing a duplicate to cross folds.
                if previous[0] != group:
                    raise ValueError(f"Exact duplicate content crosses source groups: {digest}")
                continue
            seen_hashes[digest] = (group, label)
            rows.append(row)
    if not rows:
        raise ValueError("AV manifest has no labeled video records")
    return rows, duplicates


def _rank(seed: int, purpose: str, fold: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{fold}:{group}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_fold_manifests(rows: list[dict], output: Path, *, folds: int, seed: int,
                         validation_fraction: float, duplicates_removed: int) -> dict:
    # A source group may contain both classes when it represents a real video
    # and its manipulated counterpart. Balance folds by record counts per class,
    # while assigning each complete source group to exactly one outer fold.
    group_class_counts: dict[str, Counter] = defaultdict(Counter)
    group_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group = str(row["group_id"])
        group_class_counts[group][row["label"]] += 1
        group_labels[group].add(row["label"])
    total_class_counts = Counter(row["label"] for row in rows)
    class_group_counts = {label: sum(label in labels for labels in group_labels.values()) for label in LABELS}
    if min(class_group_counts.values()) < folds:
        raise ValueError(f"Each label must occur in at least {folds} source groups for {folds}-fold evaluation")

    targets = {label: total_class_counts[label] / folds for label in LABELS}
    fold_class_counts = [Counter() for _ in range(folds)]
    fold_group_counts = [0] * folds
    ordered_groups = sorted(
        group_labels,
        key=lambda group: (-sum(group_class_counts[group].values()), _rank(seed, "outer", 0, group)),
    )
    test_fold: dict[str, int] = {}
    for group in ordered_groups:
        candidates = []
        for fold in range(folds):
            projected = fold_class_counts[fold].copy()
            projected.update(group_class_counts[group])
            total_error = 0.0
            for index, counts in enumerate(fold_class_counts):
                candidate_counts = projected if index == fold else counts
                total_error += sum(((candidate_counts[label] - targets[label]) / max(targets[label], 1)) ** 2
                                   for label in LABELS)
            candidates.append((total_error, fold_group_counts[fold], _rank(seed, f"fold-tie-{fold}", 0, group), fold))
        chosen = min(candidates)[-1]
        test_fold[group] = chosen
        fold_class_counts[chosen].update(group_class_counts[group])
        fold_group_counts[chosen] += 1

    fold_records = []
    fold_summaries = []
    all_groups = set(group_labels)
    target_validation = {label: total_class_counts[label] * validation_fraction for label in LABELS}
    for fold in range(folds):
        test_groups = {group for group, assigned in test_fold.items() if assigned == fold}
        remaining = all_groups - test_groups
        val_groups: set[str] = set()
        val_counts = Counter()
        candidate_groups = sorted(remaining, key=lambda group: _rank(seed, "validation", fold, group))
        while candidate_groups:
            current_error = sum(((val_counts[label] - target_validation[label]) / max(target_validation[label], 1)) ** 2 for label in LABELS)
            options = []
            for group in candidate_groups:
                projected = val_counts + group_class_counts[group]
                error = sum(((projected[label] - target_validation[label]) / max(target_validation[label], 1)) ** 2 for label in LABELS)
                options.append((error, _rank(seed, f"validation-choice-{fold}", 0, group), group))
            error, _, chosen = min(options)
            if val_groups and error >= current_error:
                break
            val_groups.add(chosen)
            val_counts.update(group_class_counts[chosen])
            candidate_groups.remove(chosen)
            if all(val_counts[label] > 0 for label in LABELS) and all(val_counts[label] >= target_validation[label] for label in LABELS):
                break
        train_groups = remaining - val_groups
        split_groups = {"train": train_groups, "validation": val_groups, "test": test_groups}
        for split, group_set in split_groups.items():
            found = {label for group in group_set for label in group_labels[group]}
            if found != LABELS:
                raise ValueError(f"Fold {fold + 1} {split} split does not contain both labels; inspect source grouping")

        fold_dir = output / f"fold_{fold + 1:02d}"
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise ValueError(f"Fold output must be new or empty: {fold_dir}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        by_split: dict[str, list[dict]] = {}
        for name, group_set in split_groups.items():
            selected = [row for row in rows if str(row["group_id"]) in group_set]
            _write_jsonl(fold_dir / f"{name}.jsonl", selected)
            by_split[name] = selected
            fold_records.append({"fold": fold + 1, "split": name,
                                 "groups": sorted(group_set), "records": len(selected)})
        counts = {split: {label: sum(row["label"] == label for row in subset)
                          for label in sorted(LABELS)} for split, subset in by_split.items()}
        fold_summaries.append({"fold": fold + 1,
                               "group_counts": {split: len(groups) for split, groups in split_groups.items()},
                               "record_counts": {split: len(subset) for split, subset in by_split.items()},
                               "class_counts": counts})

    all_tested = [group for group, fold in test_fold.items()]
    if len(all_tested) != len(set(all_tested)) or set(all_tested) != set(group_labels):
        raise AssertionError("Outer folds did not assign each source group exactly once")
    _write_jsonl(output / "fold_assignments.jsonl",
                 [{"group_id": group, "test_fold": fold + 1,
                   "labels": sorted(group_labels[group]),
                   "class_counts": dict(group_class_counts[group])}
                  for group, fold in sorted(test_fold.items())])
    return {"schema_version": "1.0", "method": "stratified_paired_source_group_k_fold",
            "folds": folds, "seed": seed, "validation_fraction_of_full_dataset": validation_fraction,
            "outer_test_fraction_of_full_dataset": 1 / folds,
            "approximate_train_fraction_of_full_dataset": 1 - validation_fraction - 1 / folds,
            "unique_record_count": len(rows), "source_group_count": len(group_labels),
            "source_group_counts_by_label": class_group_counts,
            "source_groups_with_multiple_labels": sum(len(labels) > 1 for labels in group_labels.values()),
            "duplicate_records_removed": duplicates_removed, "fold_summaries": fold_summaries,
            "fold_manifest_records": fold_records}


def _run(stage: str, command: list[str]) -> None:
    print(f"\n== {stage} ==", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _original_only(predictions: Path, test_manifest: Path, output: Path) -> int:
    test_by_hash: dict[str, dict] = {}
    with test_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                test_by_hash[row["sha256"]] = row
    kept = 0
    with predictions.open(encoding="utf-8") as source, output.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_row = test_by_hash.get(row.get("sha256"))
            if manifest_row is None:
                raise ValueError(f"Prediction SHA is absent from test manifest: {row.get('sha256')}")
            if not AUGMENTED_RE.search(str(manifest_row.get("relative_path") or "")):
                target.write(json.dumps(row) + "\n")
                kept += 1
    return kept


def _group_auc(groups: list[dict]) -> float | None:
    values = sorted((float(row["mean_score_synthetic"]), int(row["label"] == "synthetic")) for row in groups)
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        rank_sum += sum(label for _, label in values[index:end]) * ((index + 1 + end) / 2)
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _group_rows(predictions: Path, test_manifest: Path, fold: int) -> list[dict]:
    manifest = {}
    with test_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                manifest[row["sha256"]] = row
    grouped: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            pred = json.loads(line)
            row = manifest.get(pred.get("sha256"))
            if row is None:
                raise ValueError("Prediction is missing its test-manifest row")
            is_aug = bool(AUGMENTED_RE.search(str(row.get("relative_path") or "")))
            grouped[str(row["group_id"])].append((str(row["label"]), float(pred["score_synthetic"]), not is_aug))
    result = []
    for group_id, items in sorted(grouped.items()):
        for label in sorted({item[0] for item in items}):
            class_items = [item for item in items if item[0] == label]
            scores = [score for _, score, _ in class_items]
            result.append({"fold": fold, "group_id": group_id, "label": label,
                           "record_count": len(class_items),
                           "original_count": sum(is_original for _, _, is_original in class_items),
                           "mean_score_synthetic": statistics.mean(scores),
                           "median_score_synthetic": statistics.median(scores)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--av-root", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--audio-seconds", type=float, default=8.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--pretrained-visual", action="store_true")
    args = parser.parse_args()
    try:
        if args.folds < 3 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
            raise ValueError("folds must be at least 3; epochs/batch size positive; workers nonnegative")
        if not 0 < args.learning_rate or not math.isfinite(args.learning_rate):
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1 - 1 / args.folds:
            raise ValueError("validation fraction must be positive and leave groups for training after the outer test fold")
        manifest = args.manifest.expanduser().resolve()
        av_root = args.av_root.expanduser().resolve()
        card = args.dataset_card.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        if not manifest.is_file() or not av_root.is_dir() or not card.is_file():
            raise ValueError("manifest, AV feature root, and dataset card must exist")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output directory must be new or empty: {output}")
        rows, duplicates_removed = _read_manifest(manifest)
        output.mkdir(parents=True, exist_ok=True)
        plan = _make_fold_manifests(rows, output, folds=args.folds, seed=args.seed,
                                    validation_fraction=args.validation_fraction,
                                    duplicates_removed=duplicates_removed)
        (output / "group_cv_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        fold_reports: dict[str, list[dict]] = {"fused": [], "visual": []}
        all_group_rows: dict[str, list[dict]] = {"fused": [], "visual": []}
        for fold_info in plan["fold_summaries"]:
            fold = fold_info["fold"]
            fold_dir = output / f"fold_{fold:02d}"
            model_dir = fold_dir / "multimodal"
            readiness = fold_dir / "av_readiness.json"
            _run(f"Fold {fold}/{args.folds}: data readiness", [sys.executable, str(ROOT / "multimodal" / "check_readiness.py"),
                 "--train-manifest", str(fold_dir / "train.jsonl"), "--validation-manifest", str(fold_dir / "validation.jsonl"),
                 "--test-manifest", str(fold_dir / "test.jsonl"), "--av-root", str(av_root),
                 "--dataset-card", str(card), "--report", str(readiness)])
            ready = _load_json(readiness)
            if not ready.get("ready"):
                raise ValueError(f"Fold {fold} failed readiness: {ready.get('errors')}")
            train_cmd = [sys.executable, "-u", str(ROOT / "multimodal" / "train.py"),
                         "--train-manifest", str(fold_dir / "train.jsonl"),
                         "--validation-manifest", str(fold_dir / "validation.jsonl"),
                         "--test-manifest", str(fold_dir / "test.jsonl"), "--av-root", str(av_root),
                         "--output-dir", str(model_dir), "--epochs", str(args.epochs),
                         "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                         "--image-size", str(args.image_size), "--learning-rate", str(args.learning_rate),
                         "--audio-seconds", str(args.audio_seconds), "--device", args.device,
                         "--seed", str(args.seed + fold)]
            if args.pretrained_visual:
                train_cmd.append("--pretrained-visual")
            _run(f"Fold {fold}/{args.folds}: training", train_cmd)
            for branch in ("fused", "visual"):
                val_pred = model_dir / f"validation_{branch}_predictions.jsonl"
                test_pred = model_dir / f"test_{branch}_predictions.jsonl"
                threshold_file = fold_dir / f"threshold_{branch}.json"
                test_report = fold_dir / f"evaluation_{branch}.json"
                original_pred = fold_dir / f"test_{branch}_original_predictions.jsonl"
                original_report = fold_dir / f"evaluation_{branch}_originals.json"
                _run(f"Fold {fold}/{args.folds}: {branch} validation calibration", [sys.executable,
                     str(ROOT / "baseline" / "select_threshold.py"), "--predictions", str(val_pred),
                     "--report", str(threshold_file)])
                threshold = float(_load_json(threshold_file)["selected_threshold"])
                _run(f"Fold {fold}/{args.folds}: {branch} held-out evaluation", [sys.executable,
                     str(ROOT / "baseline" / "evaluate.py"), "--predictions", str(test_pred),
                     "--threshold", str(threshold), "--report", str(test_report)])
                original_count = _original_only(test_pred, fold_dir / "test.jsonl", original_pred)
                if original_count:
                    _run(f"Fold {fold}/{args.folds}: {branch} original-video diagnostic", [sys.executable,
                         str(ROOT / "baseline" / "evaluate.py"), "--predictions", str(original_pred),
                         "--threshold", str(threshold), "--report", str(original_report)])
                metrics = _load_json(test_report)
                metrics.update({"fold": fold, "source_group_count": fold_info["group_counts"]["test"],
                                "original_video_count": original_count,
                                "threshold": threshold,
                                "originals_report": str(original_report) if original_count else None,
                                "originals_metrics": _load_json(original_report) if original_count else None})
                fold_reports[branch].append(metrics)
                all_group_rows[branch].extend(_group_rows(test_pred, fold_dir / "test.jsonl", fold))

        result = {"schema_version": "1.0", "method": "stratified_paired_source_group_k_fold",
                  "folds": args.folds, "seed": args.seed,
                  "interpretation": "Report fold means and standard deviations. Do not pool scores from different fold models.",
                  "branches": {}}
        for branch, reports in fold_reports.items():
            summary = {}
            for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
                values = [float(report[metric]) for report in reports if report.get(metric) is not None]
                summary[metric] = {"mean": statistics.mean(values) if values else None,
                                   "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                                   "fold_values": values}
            group_auc_by_fold = []
            for fold in range(1, args.folds + 1):
                group_rows = [row for row in all_group_rows[branch] if row["fold"] == fold]
                group_auc_by_fold.append(_group_auc(group_rows))
            originals = [report["originals_metrics"] for report in reports if report.get("originals_metrics")]
            original_summary = {}
            for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
                values = [float(report[metric]) for report in originals if report.get(metric) is not None]
                original_summary[metric] = {"mean": statistics.mean(values) if values else None,
                                            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                                            "fold_values": values}
            result["branches"][branch] = {"folds": reports, "summary": summary,
                                          "original_only_summary": original_summary,
                                          "paired_source_group_auc_by_fold": group_auc_by_fold}
            _write_jsonl(output / f"per_group_{branch}.jsonl", all_group_rows[branch])
        (output / "group_cv_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Paired-source-group cross-validation completed:")
        print(json.dumps({branch: data["summary"] for branch, data in result["branches"].items()}, indent=2))
        print("CV artifacts:", output)
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
