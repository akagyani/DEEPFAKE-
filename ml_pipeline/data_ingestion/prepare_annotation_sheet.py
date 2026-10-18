"""Turn an intake inventory into a blank, review-required authorization sheet."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath


FIELDS = (
    "dataset", "relative_path", "label", "consent_status", "usage_allowed",
    "source", "group_id", "review_status", "label_candidate_reference",
    "label_basis_reference", "group_candidate_reference", "group_basis_reference",
    "group_requires_manual_review", "auxiliary_entry", "reviewer_notes",
)


def safe_relative_path(value: str, line_number: int) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized or normalized in {".", ".."} or path.is_absolute()
            or PureWindowsPath(normalized).is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and ":" in path.parts[0])):
        raise ValueError(f"Inventory line {line_number}: unsafe or empty relative_path")
    return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path, help="CSV produced by the cloud intake notebook")
    parser.add_argument("--output", required=True, type=Path, help="New CSV for human review")
    args = parser.parse_args()
    inventory = args.inventory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not inventory.is_file():
        parser.error(f"Inventory CSV does not exist: {inventory}")
    if output == inventory:
        parser.error("Output must not overwrite the source inventory")

    count = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with inventory.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "relative_path" not in reader.fieldnames:
                raise ValueError("Inventory must have a relative_path column")
            with output.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for line_number, row in enumerate(reader, 2):
                    relative_path = safe_relative_path((row.get("relative_path") or "").strip(), line_number)
                    writer.writerow({
                        "dataset": (row.get("dataset") or inventory.stem).strip(),
                        "relative_path": relative_path,
                        # These are intentionally blank. Intake guesses are reference-only.
                        "label": "",
                        "consent_status": "",
                        "usage_allowed": "",
                        "source": "",
                        "group_id": "",
                        "review_status": "unreviewed",
                        "label_candidate_reference": row.get("label_candidate", ""),
                        "label_basis_reference": row.get("label_basis", ""),
                        "group_candidate_reference": row.get("group_candidate", ""),
                        "group_basis_reference": row.get("group_basis", ""),
                        "group_requires_manual_review": row.get("group_requires_manual_review", "true"),
                        "auxiliary_entry": row.get("auxiliary_entry", ""),
                        "reviewer_notes": "",
                    })
                    count += 1
        print(f"Wrote {count} unreviewed metadata rows to {output}")
        print("Labels, permissions, sources, and group IDs remain blank for human review.")
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
