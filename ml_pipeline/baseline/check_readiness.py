"""Check dataset-card completeness and split manifests before baseline training."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "validation", "test")
LABELS = ("real", "synthetic")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
MAX_ISSUE_EXAMPLES = 20


class ReadinessCheck:
    def __init__(self) -> None:
        self.counts = Counter()
        self.examples: dict[str, list[str]] = {"errors": [], "warnings": []}

    def issue(self, severity: str, message: str) -> None:
        self.counts[severity] += 1
        if len(self.examples[severity]) < MAX_ISSUE_EXAMPLES:
            self.examples[severity].append(message)


def check_dataset_card(path: Path, check: ReadinessCheck) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        check.issue("errors", f"Cannot read dataset card {path}: {exc}")
        return
    if not text.strip():
        check.issue("errors", f"Dataset card is empty: {path}")
        return
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"\s*-\s+([^:]+):\s*(.*?)\s*$", line)
        if not match:
            continue
        field, value = match.groups()
        normalized = value.strip().lower()
        if not normalized or normalized in {"todo", "tbd", "fill in"} or normalized.startswith("<"):
            check.issue("errors", f"Dataset card field is incomplete at line {line_number}: {field.strip()}")


def check_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return str(exc)
    return None


def inspect_split(
    split: str,
    manifest: Path,
    frames_root: Path,
    db: sqlite3.Connection,
    check: ReadinessCheck,
    decode_images: bool,
    summaries: dict[str, dict],
) -> None:
    summary = {
        "records": 0,
        "frames": 0,
        "groups": 0,
        "records_by_label": {label: 0 for label in LABELS},
        "frames_by_label": {label: 0 for label in LABELS},
    }
    labels = set()
    if not manifest.is_file():
        check.issue("errors", f"Missing {split} manifest: {manifest}")
        summaries[split] = summary
        return

    try:
        stream = manifest.open(encoding="utf-8")
    except OSError as exc:
        check.issue("errors", f"Cannot read {split} manifest {manifest}: {exc}")
        summaries[split] = summary
        return

    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            context = f"{manifest}, line {line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                check.issue("errors", f"{context}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(row, dict):
                check.issue("errors", f"{context}: record must be a JSON object")
                continue

            label = row.get("label")
            group = str(row.get("group_id") or "").strip()
            digest = str(row.get("sha256") or "").lower().strip()
            frames = row.get("frames")
            valid_record = True
            if label not in LABELS:
                check.issue("errors", f"{context}: label must be real or synthetic")
                valid_record = False
            if not group:
                check.issue("errors", f"{context}: missing group_id")
                valid_record = False
            if not DIGEST_RE.fullmatch(digest):
                check.issue("errors", f"{context}: sha256 must contain 64 hexadecimal characters")
                valid_record = False
            if not isinstance(frames, list) or not frames:
                check.issue("errors", f"{context}: frames must be a non-empty list")
                continue
            if valid_record:
                summary["records"] += 1
                summary["records_by_label"][label] += 1
                labels.add(label)
                previous = db.execute("SELECT split FROM groups_seen WHERE group_id=?", (group,)).fetchone()
                if previous and previous[0] != split:
                    check.issue("errors", f"group_id {group!r} occurs in {previous[0]} and {split}")
                else:
                    db.execute("INSERT OR IGNORE INTO groups_seen VALUES (?, ?)", (group, split))
                if DIGEST_RE.fullmatch(digest):
                    previous = db.execute("SELECT group_id, split, label FROM hashes_seen WHERE sha256=?", (digest,)).fetchone()
                    if previous and previous != (group, split, label):
                        check.issue("errors", f"content hash {digest} occurs in conflicting group, split, or label")
                    elif previous:
                        check.issue("warnings", f"duplicate record content appears in {split} for group {group!r}")
                    else:
                        db.execute("INSERT INTO hashes_seen VALUES (?, ?, ?, ?)", (digest, group, split, label))
                if summary["records"] % 5000 == 0:
                    db.commit()

            for frame in frames:
                if not isinstance(frame, str) or not frame.strip():
                    check.issue("errors", f"{context}: frame paths must be non-empty strings")
                    continue
                relative = Path(frame)
                if relative.is_absolute():
                    check.issue("errors", f"{context}: frame path must be relative: {frame}")
                    continue
                resolved = (frames_root / relative).resolve()
                if resolved == frames_root or frames_root not in resolved.parents:
                    check.issue("errors", f"{context}: frame path escapes --frames-root: {frame}")
                    continue
                if not resolved.is_file():
                    check.issue("errors", f"{context}: missing frame file: {frame}")
                    continue
                summary["frames"] += 1
                if label in LABELS:
                    summary["frames_by_label"][label] += 1
                previous = db.execute("SELECT split FROM frame_paths WHERE path=?", (relative.as_posix(),)).fetchone()
                if previous and previous[0] != split:
                    check.issue("errors", f"frame path occurs in both {previous[0]} and {split}: {frame}")
                else:
                    db.execute("INSERT OR IGNORE INTO frame_paths VALUES (?, ?)", (relative.as_posix(), split))
                if decode_images:
                    image_error = check_image(resolved)
                    if image_error:
                        check.issue("errors", f"{context}: unreadable image {frame} ({image_error})")

    summary["groups"] = db.execute("SELECT COUNT(*) FROM groups_seen WHERE split=?", (split,)).fetchone()[0]
    summaries[split] = summary
    if summary["records"] == 0:
        check.issue("errors", f"{split} manifest has no valid labeled records")
    if len(labels) < 2:
        severity = "errors" if split == "validation" else "warnings"
        check.issue(severity, f"{split} contains only one class; some evaluation metrics will be unavailable")
    if summary["groups"] < 1:
        check.issue("errors", f"{split} contains no groups")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path, required=True, help="Completed copy of DATASET_CARD_TEMPLATE.md")
    parser.add_argument("--report", type=Path, help="Optional path for a JSON readiness report")
    parser.add_argument("--skip-image-check", action="store_true", help="Check paths but skip decoding each image")
    args = parser.parse_args()

    check = ReadinessCheck()
    frames_root = args.frames_root.expanduser().resolve()
    if not frames_root.is_dir():
        check.issue("errors", f"Frames root is not an existing directory: {frames_root}")
    card = args.dataset_card.expanduser().resolve()
    check_dataset_card(card, check)

    summaries: dict[str, dict] = {}
    temporary_parent = frames_root if frames_root.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="deepfake-readiness-", dir=temporary_parent) as temporary:
        db = sqlite3.connect(Path(temporary) / "readiness.sqlite")
        try:
            db.execute("CREATE TABLE groups_seen(group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
            db.execute("CREATE TABLE hashes_seen(sha256 TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL, label TEXT NOT NULL)")
            db.execute("CREATE TABLE frame_paths(path TEXT PRIMARY KEY, split TEXT NOT NULL)")
            for split, manifest in zip(SPLITS, (args.train_manifest, args.validation_manifest, args.test_manifest)):
                inspect_split(split, manifest.expanduser().resolve(), frames_root, db, check, not args.skip_image_check, summaries)
            db.commit()
        finally:
            db.close()

    train_classes = {label for label, count in summaries.get("train", {}).get("records_by_label", {}).items() if count}
    if train_classes != set(LABELS):
        check.issue("errors", "training split must include at least one record from both real and synthetic classes")

    report = {
        "schema_version": "1.0",
        "ready": check.counts["errors"] == 0,
        "dataset_card": str(card),
        "frames_root": str(frames_root),
        "image_decode_check": not args.skip_image_check,
        "splits": summaries,
        "error_count": check.counts["errors"],
        "warning_count": check.counts["warnings"],
        "errors": check.examples["errors"],
        "warnings": check.examples["warnings"],
        "note": "A readiness pass checks inputs and split integrity; it does not establish model accuracy or data-label correctness.",
    }
    print(json.dumps(report, indent=2))
    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)

