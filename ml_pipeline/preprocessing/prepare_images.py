"""Prepare consent-cleared still images as a frame manifest for the baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = {"real", "synthetic", "unknown"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest from data_ingestion/ingest.py")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Separate directory for derived image copies")
    args = parser.parse_args()

    root = args.media_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Media root does not exist or is not a directory: {root}")
    if output == root or root in output.parents:
        parser.error("Image output must be outside the source media folder")
    if manifest == output / "frames.jsonl":
        parser.error("Input manifest cannot be the frame index output path")

    temp_manifest: Path | None = None
    count = skipped = 0
    try:
        output.mkdir(parents=True, exist_ok=True)
        with manifest.open(encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", prefix=".frames-", suffix=".tmp",
                dir=output, delete=False,
            ) as destination:
                temp_manifest = Path(destination.name)
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Manifest line {line_number}: invalid JSON: {exc}") from exc
                    relative_value = str(row.get("relative_path") or "")
                    relative = Path(relative_value)
                    suffix = relative.suffix.lower()
                    if suffix not in IMAGE:
                        skipped += 1
                        continue
                    label = row.get("label")
                    group_id = str(row.get("group_id") or "").strip()
                    digest = str(row.get("sha256") or "").lower().strip()
                    if label not in LABELS or not group_id:
                        raise ValueError(f"Manifest line {line_number}: expected a supported label and non-empty group_id")
                    if not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError(f"Manifest line {line_number}: expected a 64-character SHA-256 digest")
                    media_type = str(row.get("media_type") or "").lower()
                    if media_type and media_type != suffix[1:]:
                        raise ValueError(f"Manifest line {line_number}: media_type does not match the file extension")
                    media = (root / relative).resolve()
                    if root not in media.parents or not media.is_file():
                        raise ValueError(f"Manifest line {line_number}: path is missing or outside media root")
                    if sha256_file(media) != digest:
                        raise ValueError(f"Manifest line {line_number}: file content no longer matches its SHA-256")
                    relative_key = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
                    folder = output / f"{digest}-{relative_key}"
                    folder.mkdir(parents=True, exist_ok=True)
                    image_path = folder / f"image{suffix}"
                    shutil.copy2(media, image_path)
                    destination.write(json.dumps({
                        "relative_path": relative.as_posix(),
                        "sha256": digest,
                        "label": label,
                        "group_id": group_id,
                        "frames": [image_path.relative_to(output).as_posix()],
                    }, ensure_ascii=False) + "\n")
                    count += 1
        os.replace(temp_manifest, output / "frames.jsonl")
        temp_manifest = None
    except (OSError, ValueError) as exc:
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {count} image records to {output}; skipped {skipped} non-image records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())