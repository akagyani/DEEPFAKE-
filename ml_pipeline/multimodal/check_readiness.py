"""Validate an audio-visual dataset card, feature manifests, files, and split isolation."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "validation", "test")
LABELS = {"real", "synthetic"}
HASH = re.compile(r"[0-9a-f]{64}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--av-root", type=Path, required=True)
    p.add_argument("--dataset-card", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.av_root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    summaries: dict[str, dict] = {}
    try:
        card = args.dataset_card.read_text(encoding="utf-8")
        if not card.strip():
            errors.append("Dataset card is empty")
        for line_number, line in enumerate(card.splitlines(), 1):
            match = re.match(r"\s*-\s+([^:]+):\s*(.*?)\s*$", line)
            if match and (not match.group(2) or match.group(2).lower() in {"todo", "tbd", "fill in"} or match.group(2).startswith("<")):
                errors.append(f"Dataset card line {line_number} is incomplete: {match.group(1).strip()}")
    except OSError as exc:
        errors.append(f"Cannot read dataset card: {exc}")
    if not root.is_dir():
        errors.append(f"AV feature root does not exist: {root}")
    with tempfile.TemporaryDirectory(prefix="av-readiness-") as temp:
        db = sqlite3.connect(Path(temp) / "checks.sqlite")
        db.execute("CREATE TABLE groups(group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        db.execute("CREATE TABLE hashes(sha256 TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL, label TEXT NOT NULL)")
        for split, manifest in zip(SPLITS, (args.train_manifest, args.validation_manifest, args.test_manifest)):
            counts = Counter()
            group_ids: set[str] = set()
            path = manifest.expanduser().resolve()
            if not path.is_file():
                errors.append(f"Missing {split} manifest: {path}")
                summaries[split] = dict(records=0, groups=0, classes={})
                continue
            try:
                with path.open(encoding="utf-8") as stream:
                    for line_no, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        ctx = f"{path}:{line_no}"
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            errors.append(f"{ctx}: invalid JSON ({exc.msg})"); continue
                        if not isinstance(row, dict):
                            errors.append(f"{ctx}: row must be an object"); continue
                        label, group = row.get("label"), str(row.get("group_id") or "").strip()
                        digest = str(row.get("sha256") or "").lower()
                        if label not in LABELS:
                            errors.append(f"{ctx}: expected real/synthetic label"); continue
                        if not group:
                            errors.append(f"{ctx}: missing group_id"); continue
                        if not HASH.fullmatch(digest):
                            errors.append(f"{ctx}: invalid SHA-256"); continue
                        previous = db.execute("SELECT split FROM groups WHERE group_id=?", (group,)).fetchone()
                        if previous and previous[0] != split:
                            errors.append(f"group_id {group!r} occurs in both {previous[0]} and {split}")
                        else:
                            db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?)", (group, split))
                        prior_hash = db.execute("SELECT group_id, split, label FROM hashes WHERE sha256=?", (digest,)).fetchone()
                        if prior_hash and prior_hash != (group, split, label):
                            errors.append(f"duplicate media hash occurs in conflicting group/split/label: {digest}")
                        elif prior_hash:
                            counts["exact_duplicate_records"] += 1
                        else:
                            db.execute("INSERT INTO hashes VALUES (?, ?, ?, ?)", (digest, group, split, label))
                        group_ids.add(group); counts[label] += 1
                        audio_present = row.get("audio_present")
                        if not isinstance(audio_present, bool):
                            errors.append(f"{ctx}: audio_present must be boolean")
                        elif not audio_present:
                            counts["audio_missing"] += 1
                        for key in ("frame", "log_mel"):
                            relative = Path(str(row.get(key) or ""))
                            resolved = (root / relative).resolve()
                            if relative.is_absolute() or root not in resolved.parents or not resolved.is_file():
                                errors.append(f"{ctx}: missing or unsafe {key} path {relative}")
                                continue
                            if key == "frame":
                                try:
                                    with Image.open(resolved) as image:
                                        image.verify()
                                except (OSError, ValueError, UnidentifiedImageError) as exc:
                                    errors.append(f"{ctx}: invalid frame image ({exc})")
                            else:
                                try:
                                    feature = np.load(resolved, mmap_mode="r", allow_pickle=False)
                                    if feature.shape != (64, 256) or not np.isfinite(feature).all():
                                        errors.append(f"{ctx}: log-mel must have finite shape (64, 256)")
                                except (OSError, ValueError) as exc:
                                    errors.append(f"{ctx}: unreadable log-mel ({exc})")
            except OSError as exc:
                errors.append(f"Cannot read {split} manifest: {exc}")
            summaries[split] = {"records": sum(counts[k] for k in LABELS), "groups": len(group_ids),
                                "classes": {label: counts[label] for label in sorted(LABELS)},
                                "audio_missing": counts["audio_missing"],
                                "exact_duplicate_records_after_first_occurrence": counts["exact_duplicate_records"]}
            if summaries[split]["records"] == 0:
                errors.append(f"{split} has no labeled video rows")
            if split in {"train", "validation"} and any(counts[label] == 0 for label in LABELS):
                errors.append(f"{split} must include both real and synthetic rows")
            if split == "test" and any(counts[label] == 0 for label in LABELS):
                warnings.append("test contains only one class; ROC-AUC/balanced results may be incomplete")
            if counts["audio_missing"]:
                warnings.append(f"{split}: {counts['audio_missing']} video(s) have no decodable audio")
        db.commit(); db.close()
    report = {"schema_version": "1.0", "ready": not errors, "errors": errors, "warnings": warnings,
              "av_root": str(root), "splits": summaries,
              "limits": ["Structural checks cannot verify that labels, permissions, source mappings, or group IDs are factually correct.",
                         "A complete dataset card is documentation, not legal clearance."]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
