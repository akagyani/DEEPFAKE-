"""Build an authorization-aware JSONL manifest with bounded memory use."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm",
    ".jpg", ".jpeg", ".png", ".webp",
    ".wav", ".mp3", ".flac", ".m4a",
}
REQUIRED_METADATA = {"relative_path", "label", "consent_status", "usage_allowed", "source", "group_id"}
ALLOWED_LABELS = {"real", "synthetic", "unknown"}


@contextmanager
def open_database(path: Path):
    db = sqlite3.connect(path)
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata_to_sqlite(path: Path, db: sqlite3.Connection) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_METADATA - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {', '.join(sorted(missing))}")
        count = 0
        for line_number, row in enumerate(reader, 2):
            key = (row.get("relative_path") or "").strip().replace("\\", "/")
            group = (row.get("group_id") or "").strip()
            if not key:
                raise ValueError(f"Metadata line {line_number} needs relative_path")
            if not group:
                raise ValueError(f"Metadata line {line_number} needs a non-empty group_id")
            values = tuple((row.get(name) or "").strip() for name in
                           ("label", "consent_status", "usage_allowed", "source", "group_id", "usage_basis"))
            try:
                db.execute("INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)", (key, *values))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate metadata path: {key}") from exc
            count += 1
            if count % 5000 == 0:
                db.commit()
    db.commit()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Root folder containing local media")
    parser.add_argument("--metadata", required=True, type=Path, help="CSV with label, consent, source, and group ID")
    parser.add_argument("--output", required=True, type=Path, help="JSONL manifest written outside the media folder")
    args = parser.parse_args()
    root = args.input.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Input folder does not exist or is not a folder: {root}")
    if output == metadata_path or output == root or root in output.parents:
        parser.error("Output manifest must be outside the input media folder and metadata CSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".ingest-index-", dir=output.parent) as temp:
            db_path = Path(temp) / "metadata.sqlite"
            temp_output = Path(temp) / "manifest.jsonl"
            with open_database(db_path) as db:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("CREATE TABLE metadata(relative_path TEXT PRIMARY KEY, label TEXT, consent_status TEXT, usage_allowed TEXT, source TEXT, group_id TEXT, usage_basis TEXT)")
                metadata_count = load_metadata_to_sqlite(metadata_path, db)
                timestamp = datetime.now(timezone.utc).isoformat()
                included = missing = rejected = 0
                examples: dict[str, list[str]] = {"missing metadata": [], "not authorized": []}
                with temp_output.open("w", encoding="utf-8", newline="\n") as out:
                    for path in root.rglob("*"):
                        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            continue
                        relative = path.relative_to(root).as_posix()
                        row = db.execute(
                            "SELECT label, consent_status, usage_allowed, source, group_id, usage_basis FROM metadata WHERE relative_path=?",
                            (relative,),
                        ).fetchone()
                        if row is None:
                            missing += 1
                            if len(examples["missing metadata"]) < 20: examples["missing metadata"].append(relative)
                            continue
                        label, consent, usage, source, group, usage_basis = row
                        consent_value = (consent or "").strip().lower()
                        basis = (usage_basis or "").strip()
                        consent_authorized = consent_value == "granted"
                        license_authorized = (consent_value == "not_applicable_dataset_license"
                                              and "cc by 4.0" in basis.lower()
                                              and bool((source or "").strip()))
                        if (not (consent_authorized or license_authorized)
                                or (usage or "").lower() not in {"yes", "true", "1"}):
                            rejected += 1
                            if len(examples["not authorized"]) < 20: examples["not authorized"].append(relative)
                            continue
                        label = (label or "").lower()
                        if label not in ALLOWED_LABELS:
                            raise ValueError(f"Invalid label for {relative}: use real, synthetic, or unknown")
                        record = {
                            "relative_path": relative, "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size, "media_type": path.suffix.lower().lstrip("."),
                            "label": label, "source": source, "group_id": group,
                            "consent_status": consent_value, "usage_allowed": True,
                            "usage_basis": basis or "documented individual consent", "ingested_at": timestamp,
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        included += 1
                os.replace(temp_output, output)
        print(f"Manifest written: {output} ({included} records; {metadata_count} metadata rows)")
        for category, total in (("missing metadata", missing), ("not authorized", rejected)):
            if total:
                print(f"Skipped {total} file(s) ({category}); showing up to 20:", file=sys.stderr)
                for example in examples[category]:
                    print(f"  {example}", file=sys.stderr)
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
