# Detector pipeline: next steps

The existing `data_ingestion/ingest.py` creates a local JSONL inventory after explicit consent and usage permission checks. It now requires a pseudonymous `group_id` in each metadata row and carries that field into the manifest.

## Project layout

```text
data_ingestion/       existing manifest builder and template
preprocessing/        video extraction, still-image preparation, and combined manifest
splits/               reproducible group-aware train/validation/test assignment
baseline/             readiness check, train/evaluate, and inference commands
multimodal/           AV preprocessing, readiness, late-fusion training, and inference
outputs/              generated manifests, frames, split files, and model artifacts (local/ignored)
```

## Dataset-independent completion

- Added a runtime preflight before ingestion. The complete runner checks the supported Python version and required ML/image packages and requires FFmpeg when video paths appear in the metadata CSV. This catches common setup failures before preprocessing or training starts; it does not estimate dataset storage/compute needs or establish data rights, labels, or model quality.

## Dataset acquisition and intake status

- Both Mendeley ZIP archives are present in the user-designated Google Drive `Datasets` folder. The Drive listing confirmed `SDFVD2.0_v1.zip` (891,163,921 bytes) and `Comprehensive_Deepfake_Dataset_v3.zip` (4,103,008,442 bytes), each as a direct child of that folder.
- Colab extraction completed for SDFVD2: 927 MP4s (471 fake, 456 real), each passing ZIP CRC validation. Its archive SHA-256 is recorded in the dataset card. Counts differ from the published 461/461. The Comprehensive nested ZIP central directory lists 104,201 files under its deepfake folder and 3,745 under real, plus 2,325 Mac metadata entries and other noise; its class-folder counts differ from the published 106,948/3,746. Those Comprehensive counts are preliminary directory listings only; its media bytes/CRC are not yet verified.
- A Colab attempt to materialize both archives took long enough that it was stopped; full extraction is unnecessary for a directory-only inventory. Do not rerun `cloud_dataset_intake.ipynb` for counts. It is retained for a future full media-processing run, when CRC-checked extraction is actually needed.
- Dataset-specific draft cards are in `data_ingestion/dataset_cards/`. They record the count discrepancies and leave consent and underlying source-media rights unresolved.
- A streaming annotation-sheet generator is now available to convert a produced inventory CSV into a blank review CSV. It preserves heuristic labels/groups only in clearly marked reference columns; required training labels, permissions, provenance, and group IDs remain blank. Do not generate splits or train from filename guesses.
- Range-only inventory CSVs and blank per-file review sheets were generated in a cloud worker and uploaded to `Datasets/intake_reports`; the archives' media payloads were not read for this inventory. The compressed files are `deepfake_metadata_inventory.zip` and `deepfake_metadata_review_sheets.zip`.
- The latest SDFVD2 pilot completed on a Tesla T4 using a Drive byte-range reader; its report and checkpoint remain in private storage. It used 48 inferred groups per class, 8 frames per video, and early-stopped at epoch 7 (best validation epoch 4). Group-level test accuracy was 0.467 and ROC-AUC 0.446 over 15 test groups. Synthetic recall was 0.125. The model overfit and these exploratory metrics do not establish a usable detector. The previous run metrics are historical.
- An earlier AV run remains in private storage. Its nominal held-out score (56.1% accuracy / 0.590 ROC-AUC) is invalid for source-generalization because augmentation siblings crossed splits; a later audit found 14 synthetic filename families spanning splits. The run had no decodable audio, so fused and visual predictions were effectively identical.
- The observed archive names 53 real families `v1`–`v53` and 53 synthetic families `vs1`–`vs53`. The original SDFVD record says the fake videos were created by face-swapping original Pexels videos; pairing the indexed `vN`/`vsN` families is a conservative lineage inference from the naming and generation description. The earlier runs treated them as 106 separate groups, so those scores are invalid for paired-source-independent evaluation. The revised metadata and CV code pair them and retain the mapping as a review flag.
- A metadata-only audit of the Comprehensive inventory found 107,944 images (104,200 candidate synthetic, 3,744 candidate real), 30 real filename stems, and 870 synthetic stems. Parsing yields 866 candidate source/method keys with four filename collisions, 47 normalized names, 15 video indices, and two techniques. The user-reported Vid-N to N.mp4 convention gives 564 explicit-marker candidates; another 302 Tech-1 candidates omit Vid and remain an inference. Fifteen male_male_N real stems remain unmapped. Counts still do not reconcile with the publisher’s 450 synthetic videos or 30 people. Review sheets remain in private storage; these are filename-derived candidates, not verified source groups.

## Implementation choices

- `run_pipeline.py` ties the stages together for a complete local run, with optional post-training inference.

- Keep source media local; extraction writes derived frames to a separate output directory and never edits source files.
- Use `preprocessing/prepare_media.py` to route mixed manifests through the existing still-image and video processors and write one combined frame manifest. Video extraction verifies the source SHA-256; do not extract audio or identify/crop faces in this first iteration.
- Require an explicit `group_id` for each media item before making dataset splits. All items sharing a group stay in one split, preventing subject/source leakage. Exact duplicate hashes with different group IDs are rejected by the splitter.
- Record dataset origin, license, allowed uses, label meaning, and citation in `data_ingestion/DATASET_CARD_TEMPLATE.md` before training.
- Make split seed, ratios, frame rate, frame cap, and image size explicit and save them with generated artifacts.
- Train the provided ResNet-18 frame-classification baseline from scratch, or use the Colab pilot's pretrained GPU baseline for a quick experiment. Report per-class precision/recall, confusion matrix, ROC-AUC when defined, and split metadata. Aggregate correlated frame predictions to the source group for headline metrics. Never present a score as proof that media is authentic.
- Training loss is media-balanced and class-balanced: each unique media item has equal total weight within its class, then each class has equal total weight. The training report records per-class unique-media counts and the weighting rule.
- Evaluate labeled predictions incrementally with `baseline/evaluate.py`; exact ROC-AUC ranking uses temporary disk space so RAM use stays bounded. Choose any operating threshold on validation data before reporting test performance.
- Save predictions from the selected checkpoint on validation data, choose the operating threshold using validation predictions only, then evaluate once on the held-out test predictions.
- Inference emits versioned JSON with per-frame scores and an aggregated media score, model identifier, and caveat that scores are not calibrated without validation.

## Audio-visual implementation status

- The separate `multimodal/` path now prepares a midpoint frame and fixed-duration log-mel audio representation, builds an image/audio late-fusion model, validates split artifacts and dataset-card completeness, trains on group-disjoint splits, selects the checkpoint on fused validation data, selects thresholds on validation data, and evaluates held-out fused, visual-only, and audio-only predictions through the shared evaluator. Audio-only evaluation is restricted to videos with audio and can be skipped if the validation subset cannot support threshold selection.
- `multimodal/run_pipeline.py` orchestrates the AV path and `multimodal/infer_media.py` accepts one unlabeled video for post-training inference. The existing `run_pipeline.py` remains the visual-only orchestration path.
- The AV design uses one frame and the first eight seconds of audio; no face crop, long-range temporal model, speech transcription, or audio-video synchronization feature is implemented. A separate visual-only external evaluator has now been run on Comprehensive v3; see the external evaluation section below. The historical 2026-10-16 run found no decodable audio, so that experiment evaluated video signals only. In inference reports the audio score is explicitly null when audio cannot be decoded.
- Structural checks cannot establish whether source labels, Vid-N/source mappings, subject group IDs, consent, licenses, and underlying source-video permissions are correct. Resolve them from authoritative dataset documentation and review before using the AV runner.

## Before training
After reviewing the inventory sheets, complete per-file labels, permission status, source provenance, and leakage-resistant group IDs from authoritative documentation/consent records. Complete the dataset card too. Then run `baseline/check_readiness.py` after creating the split manifests. It checks required card fields, manifest structure, frame availability and readability, and cross-split group/hash/path separation. A one-class validation split stops the run because the operating threshold must be chosen from both validation classes. This structural check cannot confirm label truth, legal permission, or model quality. Model training is not yet valid because these authoritative inputs are still blank/unresolved.


Record dataset origin, license, allowed uses, label meaning, and citation in `data_ingestion/DATASET_CARD_TEMPLATE.md`. Confirm labels and consent independently. Exact duplicate hashes are checked during splitting; near duplicates and related clips still need to share a group ID. Do not use competition data outside its terms. Training streams image batches and uses a disk-backed frame index; storage capacity, bandwidth, compute, and time remain finite constraints.

## Corrected SDFVD 2.0 evaluation (2026-10-16)

- The run artifacts are retained in private storage.
- The historical metadata audit found 927 videos (456 real, 471 synthetic), 106 filename-family groups, and 17 exact duplicate records excluded from splitting. Pairing `vN` with `vsN` reduces the candidate lineage groups to 53, subject to report confirmation. The publisher reports 922 videos (461/class), leaving a five-file discrepancy for manual review.
- Historical held-out metrics were 35.25% accuracy, 35.29% balanced accuracy, and 0.2752 ROC-AUC; the five-fold result averaged 0.299 ROC-AUC. These runs were split by augmentation family but not by inferred real/fake source pair, so they are not valid estimates of paired-source-independent performance. All retained videos lacked decodable audio.
- Interpretation: the earlier runs' source-group isolation did not catch cross-label paired-source leakage. The revised pair-aware CV must be rerun before drawing a source-independent conclusion. Threshold tuning cannot repair an invalid source split.

## Pipeline implementation status (2026-10-16)

The remaining reusable pipeline and Colab handoff are implemented in the local project. No new model training was launched in this work, per the user's instruction.

- SDFVD2 metadata now groups indexed real `vN` and manipulated `vsN` families under one inferred paired-source ID, including augmentation descendants and downloader suffixes. The audit records this as an inference and reports the pair count and class coverage.
- Five-fold CV now assigns whole mixed-label source pairs to folds, creates group-disjoint validation subsets within each fold, asserts both classes in each split, and reports fold-wise metrics plus paired-source-group AUC without pooling model scores across folds.
- Colab launcher setup embeds the current source bundle and refreshes its Drive copy, avoiding stale/mismatched bundle errors. The generated source ZIP excludes the launcher notebook to avoid a recursive archive dependency.
- Single-video inference accepts a validation-selected threshold, records the checkpoint SHA-256 and decision context, validates the checkpoint safely, and leaves the audio score null if no audio is available. Batch inference follows the same null-audio convention.
- The README and dataset notes explain the evaluation limits and how to launch the path. The old scores remain historical and invalid for paired-source independence.

The next *dataset/model* work is intentionally deferred: the user will run training later. Before treating any future metrics as reliable, inspect the new metadata audit, confirm the inferred `vN`/`vsN` mapping with authoritative source documentation where possible, review the five-file count discrepancy, and evaluate on an independent dataset with verified labels/source identities. The Comprehensive dataset remains a separate future workstream until its candidate lineage groups, media bytes, counts, and permissions are reconciled. No current score establishes external generalization or a usable detector.

## Comprehensive v3 external evaluation handoff (2026-10-17)

- Added `multimodal/evaluate_external_frames.py` to score each saved SDFVD2 CV fold's visual branch on one deterministic median-index face crop per filename-derived candidate video. It uses each fold's validation-selected visual threshold, reports folds separately, and does not train or calibrate on Comprehensive v3.
- Added `comprehensive_external_evaluation.ipynb` for Colab. It reads the Comprehensive archive and completed SDFVD2 CV artifacts from Drive, extracts data to temporary Colab storage, and saves an external report plus per-candidate predictions to Drive. The refreshed source bundle is included in the Drive handoff.
- The code labels results provisional unless the publisher-reported and observed frame/video counts match and the operator explicitly marks the source mapping reviewed. Current inventory counts do not match the publisher and candidate filename stems do not reconcile to the reported 450 synthetic videos. The full mapping has not been reviewed, so any score is exploratory and its candidate video units may be correlated.
- The related conference paper says filenames encode frame index, generation tool, and source-video ID, and that a metadata CSV maps image, subject, label, source video, and tool. The supplied v3 archive inventory and designated Drive folder do not contain that publisher CSV; the Drive CSVs found are project-generated audit/review sheets. The paper is internally inconsistent about whether there are 450 or 900 fake videos, while the extracted inventory has 870 synthetic filename stems. See the Comprehensive v3 dataset card for the exact claims and the DOI link. Do not promote filename heuristics to verified lineage; the publisher mapping CSV or an author-confirmed mapping is required.
- Colab evaluation completed on 2026-10-16; its report remains in private storage. Five-fold mean ROC-AUC was 0.5248 (SD 0.0420), balanced accuracy 0.5116 (SD 0.0072), and accuracy 0.6967 (SD 0.2322). The candidate set is highly imbalanced (870 synthetic vs 30 real), so accuracy is misleading. This provisional result is near chance and does not establish generalization: observed frame/video counts differ from publisher claims, candidate units do not reconcile to the reported 480 videos, and source mappings remain unreviewed.
- Colab outputs remain in private storage; this repository copy does not include the dataset, raw predictions, or private storage identifiers.
- Personalized Colab notebooks and generated bundles are excluded. No project code license has been chosen; choose one before presenting a public repository as open source. Source media, model outputs, and checkpoints remain excluded by ignore rules.

## Still-image datasets

preprocessing/        video extraction, still-image preparation, and combined manifest
