"""Create reproducible group-disjoint train/validation/test JSONL manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path


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


def allocate_group_counts(total: int, fractions: list[float]) -> list[int]:
    """Allocate whole groups by largest remainder, keeping every split nonempty when possible."""
    raw = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    minimum = 1 if total >= len(fractions) else 0
    counts = [max(minimum, count) for count in counts]
    while sum(counts) < total:
        index = max(range(len(counts)), key=lambda i: (raw[i] - counts[i], -i))
        counts[index] += 1
    while sum(counts) > total:
        eligible = [i for i, count in enumerate(counts) if count > minimum]
        if not eligible:
            break
        index = max(eligible, key=lambda i: (counts[i] - raw[i], -i))
        counts[index] -= 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Frame JSONL from preprocessing/extract_frames.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=str, default="2026")
    p.add_argument("--train", type=float, default=.70, help="Approximate fraction of groups assigned to train")
    p.add_argument("--validation", type=float, default=.15, help="Approximate fraction of groups assigned to validation")
    args = p.parse_args()
    test = 1 - args.train - args.validation
    if not all(math.isfinite(value) for value in (args.train, args.validation, test)) or min(args.train, args.validation, test) <= 0:
        p.error("train and validation fractions must be positive and sum to less than 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = 0
    unknown_records = 0
    group_count = 0
    fractions = [args.train, args.validation, test]
    counts: list[int] = []
    names = ["train", "validation", "test"]
    with tempfile.TemporaryDirectory(prefix=".split-index-", dir=args.output_dir) as temp:
        with open_database(Path(temp) / "index.sqlite") as db:
            db.execute("CREATE TABLE records(row_id INTEGER PRIMARY KEY, group_id TEXT, source_group_id TEXT, sha256 TEXT, label TEXT NOT NULL, payload TEXT)")
            db.execute("CREATE TABLE duplicate_records(row_id INTEGER PRIMARY KEY, group_id TEXT, source_group_id TEXT, sha256 TEXT, payload TEXT)")
            db.execute("CREATE TABLE groups(group_id TEXT PRIMARY KEY, sort_key TEXT NOT NULL, split TEXT)")
            db.execute("CREATE TABLE hashes(sha256 TEXT PRIMARY KEY, group_id TEXT NOT NULL, label TEXT NOT NULL)")
            db.execute("CREATE TABLE group_aliases(source_group_id TEXT PRIMARY KEY, group_id TEXT NOT NULL)")
            db.execute("CREATE TABLE assignments(group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
            exact_duplicate_records = 0
            exact_duplicate_group_merges = 0
            input_records = 0
            with args.manifest.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        p.error(f"Record {line_number} is invalid JSON: {exc}")
                    source_group = str(row.get("group_id") or "").strip()
                    if not source_group:
                        p.error(f"Record {line_number} has no group_id")
                    alias = db.execute("SELECT group_id FROM group_aliases WHERE source_group_id=?", (source_group,)).fetchone()
                    group = alias[0] if alias else source_group
                    if row.get("label") == "unknown":
                        unknown_records += 1
                        continue
                    if row.get("label") not in {"real", "synthetic"}:
                        p.error(f"Record {line_number} has an unsupported label")
                    input_records += 1
                    digest = str(row.get("sha256") or "").lower().strip()
                    if not re.fullmatch(r"[0-9a-f]{64}", digest):
                        p.error(f"Record {line_number} must contain a 64-character SHA-256 digest")
                    db.execute("INSERT OR IGNORE INTO groups(group_id, sort_key) VALUES (?, ?)", (
                        group, hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest(),
                    ))
                    db.execute("INSERT OR IGNORE INTO group_aliases VALUES (?, ?)", (source_group, group))
                    previous = db.execute("SELECT group_id, label FROM hashes WHERE sha256=?", (digest,)).fetchone()
                    if previous:
                        exact_duplicate_records += 1
                        if previous[1] != row["label"]:
                            p.error(f"Exact duplicate content has conflicting labels on record {line_number}")
                        if previous[0] != group:
                            # Exact byte duplicates must be assigned together. Merge their inferred
                            # source groups (and any groups already linked through other duplicates).
                            canonical, merged = sorted((previous[0], group))
                            db.execute("UPDATE records SET group_id=? WHERE group_id=?", (canonical, merged))
                            db.execute("UPDATE duplicate_records SET group_id=? WHERE group_id=?", (canonical, merged))
                            db.execute("UPDATE hashes SET group_id=? WHERE group_id=?", (canonical, merged))
                            db.execute("UPDATE group_aliases SET group_id=? WHERE group_id=?", (canonical, merged))
                            db.execute("DELETE FROM groups WHERE group_id=?", (merged,))
                            exact_duplicate_group_merges += 1
                            group = canonical
                        db.execute("INSERT INTO duplicate_records(group_id, source_group_id, sha256, payload) VALUES (?, ?, ?, ?)", (
                            group, source_group, digest, json.dumps(row, ensure_ascii=False),
                        ))
                        if input_records % 5000 == 0:
                            db.commit()
                        continue
                    else:
                        db.execute("INSERT INTO hashes VALUES (?, ?, ?)", (digest, group, row["label"]))
                    db.execute("INSERT OR IGNORE INTO groups(group_id, sort_key) VALUES (?, ?)", (
                        group, hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest(),
                    ))
                    db.execute("INSERT INTO records(group_id, source_group_id, sha256, label, payload) VALUES (?, ?, ?, ?, ?)", (
                        group, source_group, digest, row["label"], json.dumps(row, ensure_ascii=False),
                    ))
                    records += 1
                    if records % 5000 == 0:
                        db.commit()
            db.commit()
            group_count = db.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
            if not records:
                p.error("Manifest contains no records")
            if group_count < 3:
                p.error("At least 3 distinct groups are required for non-empty train, validation, and test splits")
            group_labels = db.execute(
                "SELECT group_id, MIN(label), COUNT(DISTINCT label) FROM records GROUP BY group_id ORDER BY group_id"
            ).fetchall()
            mixed_label_groups = sum(label_count > 1 for _, _, label_count in group_labels)
            if mixed_label_groups == 0:
                # SDFVD2 and other single-label source-family datasets need group-level
                # stratification; hash-ordering all groups together can yield a severely
                # skewed validation set even when the source classes are balanced.
                by_label: dict[str, list[str]] = {}
                for group, label, _ in group_labels:
                    by_label.setdefault(label, []).append(group)
                for label, label_groups in sorted(by_label.items()):
                    label_groups.sort(key=lambda group: db.execute(
                        "SELECT sort_key FROM groups WHERE group_id=?", (group,)
                    ).fetchone()[0])
                    label_counts = allocate_group_counts(len(label_groups), fractions)
                    position = 0
                    for split_name, split_count in zip(names, label_counts):
                        for group in label_groups[position:position + split_count]:
                            db.execute("INSERT INTO assignments VALUES (?, ?)", (group, split_name))
                        position += split_count
                stratification_method = "per_label_group_stratified"
            else:
                raw = [group_count * fraction for fraction in fractions]
                counts = allocate_group_counts(group_count, fractions)
                cursor = db.execute("SELECT group_id FROM groups ORDER BY sort_key, group_id")
                position = 0
                boundary = counts[0]
                split_index = 0
                for (group,) in cursor:
                    while split_index < 2 and position >= boundary:
                        split_index += 1
                        boundary += counts[split_index]
                    db.execute("INSERT INTO assignments VALUES (?, ?)", (group, names[split_index]))
                    position += 1
                stratification_method = "deterministic_group_hash_mixed_label_fallback"
            db.commit()
            counts = [db.execute("SELECT COUNT(*) FROM assignments WHERE split=?", (name,)).fetchone()[0]
                      for name in names]
            for name in names:
                with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                    query = db.execute("SELECT records.group_id, records.source_group_id, records.payload FROM records JOIN assignments USING(group_id) WHERE assignments.split=? ORDER BY records.row_id", (name,))
                    for effective_group, source_group, payload in query:
                        row = json.loads(payload)
                        row["group_id"] = effective_group
                        if source_group != effective_group:
                            row["source_group_id"] = source_group
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (args.output_dir / "group_assignments.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                for group, split in db.execute("SELECT group_id, split FROM assignments ORDER BY group_id"):
                    out.write(json.dumps({"group_id": group, "split": split}, ensure_ascii=False) + "\n")
            with (args.output_dir / "group_aliases.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                for source_group, effective_group in db.execute("SELECT source_group_id, group_id FROM group_aliases ORDER BY source_group_id"):
                    out.write(json.dumps({"source_group_id": source_group, "group_id": effective_group}, ensure_ascii=False) + "\n")
            with (args.output_dir / "duplicate_records.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                for effective_group, source_group, digest, payload in db.execute(
                        "SELECT group_id, source_group_id, sha256, payload FROM duplicate_records ORDER BY row_id"):
                    row = json.loads(payload)
                    out.write(json.dumps({"relative_path": row.get("relative_path"), "sha256": digest,
                                          "label": row.get("label"), "source_group_id": source_group,
                                          "group_id": effective_group,
                                          "excluded_reason": "exact_duplicate_content"}, ensure_ascii=False) + "\n")
            original_group_count = db.execute("SELECT COUNT(*) FROM group_aliases").fetchone()[0]
    with (args.output_dir / "split_config.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "fractions": {"train": args.train, "validation": args.validation, "test": test},
                   "fraction_basis": "group_count", "group_counts": dict(zip(names, counts)),
                   "stratification_method": stratification_method,
                   "input_labeled_record_count": input_records, "record_count": records,
                   "exact_duplicate_records_excluded": exact_duplicate_records,
                   "duplicate_records_file": "duplicate_records.jsonl",
                   "unknown_records_excluded": unknown_records, "group_count": group_count,
                   "original_group_count": original_group_count,
                   "exact_duplicate_records_after_first_occurrence": exact_duplicate_records,
                   "exact_duplicate_group_merges": exact_duplicate_group_merges,
                   "group_assignments_file": "group_assignments.jsonl",
                   "group_aliases_file": "group_aliases.jsonl"}, f, indent=2)
    print(f"Wrote {records} unique-content records across {group_count} effective groups to {args.output_dir}; "
          f"excluded {exact_duplicate_records} exact duplicate record(s) to duplicate_records.jsonl; "
          f"merged {exact_duplicate_group_merges} source group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
