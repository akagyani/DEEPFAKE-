"""Build a provenance-rich metadata CSV from the documented SDFVD 2.0 layout.

Labels are derived from agreeing folder/filename evidence where available. Augmentation
suffixes are removed to keep each source family in one split. The report exposes every
ambiguous or non-standard item; nothing is silently discarded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4"}
PREFIX = re.compile(r"^(real|fake|forged|synthetic)[_-]+", re.IGNORECASE)
AUGMENTATION = re.compile(r"_aug_(\d+)(?:$|[.\s_-].*)$", re.IGNORECASE)
VID_NUMBER_ALIAS = re.compile(r"^vid[-_ ]?(\d+)$", re.IGNORECASE)
# The original SDFVD record describes fake clips made by face-swapping original
# source videos; the indexed vN/vsN archive naming is treated as paired lineage.
# Keep each inferred pair and its augmentations together to prevent leakage.
SDFVD_PAIR_ALIAS = re.compile(r"^v(s)?(\d+)$", re.IGNORECASE)
FOLDER_CLASS_SUFFIX = re.compile(r"(?:^|[\s_.-])(real|fake|forged|synthetic)$", re.IGNORECASE)
LABEL_MAP = {"real": "real", "fake": "synthetic", "forged": "synthetic", "synthetic": "synthetic"}
SOURCE = "Kaman & Makandar (2025), SDFVD2.0 V1, DOI:10.17632/zzb7jyy8w8.1; CC BY 4.0 dataset record"


def labels_from_folder(relative: Path) -> set[str]:
    candidates: set[str] = set()
    for component in relative.parts[:-1]:
        match = FOLDER_CLASS_SUFFIX.search(component)
        if match:
            candidates.add(LABEL_MAP[match.group(1).lower()])
    return candidates


def label_from_filename(filename: str) -> str | None:
    match = PREFIX.match(filename)
    return LABEL_MAP[match.group(1).lower()] if match else None


def derive_group(filename: str) -> tuple[str, bool, str]:
    stem = Path(filename).stem
    prefixed = PREFIX.match(stem)
    if prefixed:
        stem = stem[prefixed.end():]
    # Drive-generated disambiguation suffixes (and " - Copy" suffixes) can
    # follow the documented augmentation index, e.g. fake_vs10_aug_0-AbC123.mp4.
    # The source family ends immediately before `_aug_N`; everything after it
    # is an augmentation/download suffix, not a distinct source identity.
    augmented = AUGMENTATION.search(stem)
    has_aug = augmented is not None
    nonstandard = bool(augmented and int(augmented.group(1)) not in range(8))
    if augmented:
        stem = stem[:augmented.start()]
    # The published pattern embeds the original filename before `_aug_N`.
    # For an original named `11.mp4`, an augmented file may therefore be
    # `real_11.mp4_aug_0.mp4`; Path.stem removes only the final extension and
    # leaves this inner `.mp4` after the augmentation suffix is stripped.
    # Remove that embedded extension so the source and its derivatives share
    # the same group, and so `Vid-11.mp4` can match `11.mp4` below.
    stem = re.sub(r"\.mp4$", "", stem, flags=re.IGNORECASE)
    normalized = unicodedata.normalize("NFC", stem).casefold().strip()
    if not normalized:
        normalized = Path(filename).stem.casefold()
    # In SDFVD2, the documented Vid-N naming is the fake counterpart of N.mp4.
    # Co-locate both with all their augmentation descendants to prevent identity
    # leakage between train/validation/test. This alias is intentionally narrow.
    alias = VID_NUMBER_ALIAS.fullmatch(normalized)
    pair_alias = SDFVD_PAIR_ALIAS.fullmatch(normalized)
    if pair_alias:
        normalized = f"sdfvd-pair-{pair_alias.group(2)}"
        group_rule = "paired_v_vs_numeric_family"
    elif alias:
        normalized = alias.group(1)
        group_rule = "vid_number_alias_to_numeric_stem"
    else:
        group_rule = "augmentation_suffix" if has_aug else "filename_stem_fallback"
    group = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    if nonstandard:
        group_rule = "nonstandard_augmentation_index"
    return f"sdfvd2-source-{group}", has_aug, group_rule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-card", type=Path)
    args = parser.parse_args()
    root = args.media_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"SDFVD 2.0 extracted media folder not found: {root}")
    if output == root or root in output.parents or report_path == root or root in report_path.parents:
        parser.error("Metadata and report outputs must be outside the source media folder")

    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        parser.error(f"No MP4 files found under {root}")
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    conflicts: list[dict[str, str]] = []
    for path in videos:
        relative = path.relative_to(root)
        folder_labels = labels_from_folder(relative)
        filename_label = label_from_filename(path.name)
        folder_label = next(iter(folder_labels)) if len(folder_labels) == 1 else None
        reasons: list[str] = []
        if len(folder_labels) > 1:
            label = "unknown"
            reasons.append("conflicting_class_tokens_in_path")
        elif folder_label and filename_label and folder_label != filename_label:
            label = "unknown"
            reasons.append("folder_filename_label_conflict")
        else:
            label = folder_label or filename_label or "unknown"
            if not folder_label:
                reasons.append("label_from_filename_only" if filename_label else "no_label_evidence")
            if not filename_label:
                reasons.append("label_from_folder_only" if folder_label else "no_label_evidence")
        group_id, has_aug, group_rule = derive_group(path.name)
        if not has_aug:
            reasons.append("no_documented_aug_suffix_grouped_by_stem")
        if group_rule == "nonstandard_augmentation_index":
            reasons.append(group_rule)
        if group_rule == "vid_number_alias_to_numeric_stem":
            reasons.append(group_rule)
        if group_rule == "paired_v_vs_numeric_family":
            reasons.append("paired_real_fake_source_alias")
        if label == "unknown":
            conflicts.append({"relative_path": relative.as_posix(), "reason": ";".join(sorted(set(reasons)))} )
        counters[label] += 1
        rows.append({
            "relative_path": relative.as_posix(), "label": label,
            "consent_status": "not_applicable_dataset_license", "usage_allowed": "yes",
            "usage_basis": "Mendeley dataset record declares CC BY 4.0; retain attribution; underlying source-media rights not independently verified",
            "source": SOURCE, "group_id": group_id,
            "label_evidence": f"folder={folder_label or 'none'};filename_prefix={filename_label or 'none'}",
            "group_rule": group_rule, "augmented_filename": str(has_aug).lower(),
            "review_flags": ";".join(sorted(set(reasons))),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["group_id"] for row in rows)
    source_group_counts_by_label = {
        label: len({row["group_id"] for row in rows if row["label"] == label})
        for label in sorted({row["label"] for row in rows})
    }
    paired_source_groups = {
        row["group_id"] for row in rows if row["group_rule"] == "paired_v_vs_numeric_family"
    }
    report = {
        "schema_version": "1.0", "dataset": "SDFVD2.0 V1", "source": SOURCE,
        "media_root": str(root), "metadata_csv": str(output), "videos_found": len(videos),
        "label_counts": dict(sorted(counters.items())), "unique_candidate_source_groups": len(groups),
        "source_group_counts_by_label": source_group_counts_by_label,
        "paired_source_group_count": len(paired_source_groups),
        "group_rule_counts": dict(sorted(Counter(row["group_rule"] for row in rows).items())),
        "vid_number_alias_count": sum(row["group_rule"] == "vid_number_alias_to_numeric_stem" for row in rows),
        "augmented_filename_count": sum(row["augmented_filename"] == "true" for row in rows),
        "nonstandard_augmentation_index_count": sum("nonstandard_augmentation_index" in row["review_flags"] for row in rows),
        "grouping_rule": "strip optional class prefix and the _aug_N marker plus any following downloader/copy suffix; remove embedded .mp4; group identical case-folded NFC source stems across labels; pair documented SDFVD vN real and vsN synthetic source families by N",
        "license": "CC BY 4.0 per Mendeley dataset record; source-media rights not independently verified",
        "ambiguous_label_count": len(conflicts), "ambiguous_label_examples": conflicts[:100],
        "published_video_count": 922, "published_per_class_count": 461,
        "count_difference_from_published": len(videos) - 922,
        "training_clearance": "dataset license recorded; source-media rights and external redistribution remain unverified",
        "manual_review_required": bool(conflicts or len(videos) != 922 or any(row["review_flags"] for row in rows)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dataset_card:
        card_path = args.dataset_card.expanduser().resolve()
        if card_path == root or root in card_path.parents:
            parser.error("Dataset card must be written outside the source media folder")
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(
            "# SDFVD2.0 V1 training dataset card\n\n"
            "## Provenance and license\n\n"
            "- Record: Kaman & Makandar (2025), SDFVD2.0 V1, DOI 10.17632/zzb7jyy8w8.1.\n"
            "- Source: https://data.mendeley.com/datasets/zzb7jyy8w8/1\n"
            "- Published dataset license: CC BY 4.0; retain attribution and license notice.\n"
            "- Underlying source-media rights have not been independently verified; this run is research evaluation and does not authorize redistribution.\n\n"
            "## Observed archive\n\n"
            f"- Extracted MP4 count: {len(videos)}.\n"
            f"- Derived label counts: real={counters['real']}, synthetic={counters['synthetic']}, unknown={counters['unknown']}.\n"
            f"- Published count: 922 (461 real, 461 forged); observed difference: {len(videos) - 922}.\n"
            f"- Candidate paired source groups: {len(groups)} (index-based vN/vsN pairing).\n"
            f"- Candidate groups represented by class: {json.dumps(source_group_counts_by_label, sort_keys=True)}.\n"
            f"- Videos with documented _aug_N filename suffix: {sum(row['augmented_filename'] == 'true' for row in rows)}.\n"
            "- Labels use matching terminal class-folder and optional filename-prefix evidence; conflicts remain unknown and are excluded from supervised splits.\n"
            "- Group candidates strip an optional class prefix and the `_aug_N` marker plus any following downloader/copy suffix. Indexed SDFVD families `vN` and `vsN` are paired by N based on the archive naming convention; their originals and augmentation descendants share one group. Confirm this inferred index mapping against publisher documentation if available.\n"
            "- See the machine-readable metadata report for every anomaly and label evidence.\n\n"
            "## Processing and evaluation\n\n"
            "- One midpoint video frame and up to eight seconds of mono audio are converted to a 224x224 visual input and 64x256 log-mel feature.\n"
            "- Groups are assigned 70/15/15 to train/validation/test with seed 2026. The test split is held out during model selection and threshold selection.\n"
            "- Report fused, visual-only, and audio-only test metrics. Results reflect this small augmented dataset and are not a claim of broad real-world generalization.\n",
            encoding="utf-8",
        )
    summary = {k: report[k] for k in ("videos_found", "label_counts", "unique_candidate_source_groups", "vid_number_alias_count", "augmented_filename_count", "ambiguous_label_count", "count_difference_from_published", "metadata_csv")}
    summary["report_path"] = str(report_path)
    if args.dataset_card:
        summary["dataset_card"] = str(args.dataset_card.expanduser().resolve())
    print(json.dumps(summary, indent=2))
    return 0 if counters["real"] and counters["synthetic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
