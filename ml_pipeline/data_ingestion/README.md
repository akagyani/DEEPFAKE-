# Data Ingestion

This utility builds a JSON Lines manifest for a local media dataset. It records each file's relative path, SHA-256 digest, size, media type, label, source, group ID, consent status, and ingestion time. It streams the media tree and uses a temporary SQLite index for metadata, rather than keeping the full dataset listing in RAM. It does not upload, alter, or copy media.

## Requirements

- Python 3.10 or newer
- A local folder of media files
- A metadata CSV with an explicit authorization row for each file

Supported extensions: MP4, MOV, MKV, AVI, WEBM, JPG, JPEG, PNG, WEBP, WAV, MP3, FLAC, and M4A.

## Metadata CSV

After a cloud intake run, create a human-review sheet from an inventory CSV:

```powershell
python data_ingestion/prepare_annotation_sheet.py --inventory "C:\data\intake_inventory.csv" --output "C:\data\metadata_review.csv"
```

The tool copies paths and keeps intake label/group hints in reference-only columns. The fields consumed by ingestion (`label`, `consent_status`, `usage_allowed`, `source`, and `group_id`) are deliberately blank. Review and fill them from documented sources and authorization records; do not copy guesses into these fields. Auxiliary entries are flagged for exclusion or review.

Required column names:

```csv
relative_path,label,consent_status,usage_allowed,source,group_id
subject_a/clip01.mp4,real,granted,yes,consented_collection,subject_a
sample02.mp4,synthetic,granted,yes,generated_with_permission,synthetic_source_b
```

`relative_path` must be relative to the input folder and use `/` separators. Labels are `real`, `synthetic`, or `unknown`. `group_id` is a stable pseudonymous value for the highest relevant leakage group (for example, one subject, recording session, or source collection); never put a person's name or other direct identity information here. Every row must have one. The split tool keeps one group wholly within one split. Only rows with `consent_status=granted` and `usage_allowed=yes` (also accepts `true` or `1`) are included. Keep the CSV and media local and only include material you have the right to use.

## Run

```powershell
python ingest.py --input "C:\path\to\media" --metadata "C:\path\to\metadata.csv" --output "manifest.jsonl"
```

Files without a metadata row or without explicit permission are skipped and reported (up to 20 example paths per category). The manifest is written as UTF-8 JSONL to a temporary file and moved into place when ingestion completes. No external packages are required; temporary disk space is needed for the metadata index.

## Before using on a full dataset

Try a tiny, permissioned sample first and inspect the manifest. The current version checks file presence, extension, file size, hash, required metadata fields, group ID presence, and authorization declarations; it does not inspect media contents or validate label, group, consent, or usage declarations. During splitting, exact duplicate content merges its source groups so it cannot cross splits, and duplicate copies are excluded from train/evaluation manifests. Every excluded copy is listed in `duplicate_records.jsonl`; label conflicts for byte-identical media fail closed.
