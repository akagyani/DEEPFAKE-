# Baseline training and inference

This pipeline now includes a runnable **ResNet-18 frame-classification baseline**. It starts from random weights and does not download a pretrained model. It supports the binary labels `real` and `synthetic`; `unknown` records are excluded by the splitter. Model quality is not known until it is trained and independently evaluated on the supplied dataset.

## Install

Use an isolated virtual environment so unrelated system Python packages cannot conflict with PyTorch. In PowerShell, from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r baseline\requirements.txt
```

Also ensure FFmpeg is available on PATH. PyTorch selects CUDA, Apple MPS, or CPU automatically. Use a small batch size on limited accelerator memory. Run the training and inference commands below with `.\.venv\Scripts\python.exe`.

Before a long run, check the Python runtime and installed packages with:

```powershell
.\.venv\Scripts\python.exe baseline\check_environment.py
```

The complete runner performs this check before ingestion and requires FFmpeg when the metadata contains video paths. The report describes available runtime components only; it does not validate dataset permissions, labels, storage capacity, or model quality.

## Mixed image and video preparation

Use this command when the manifest contains both still images and videos. It runs the existing processors and writes one combined `frames.jsonl` for splitting. The output directory must be new or empty.

```powershell
python preprocessing/prepare_media.py --manifest "C:\data\manifest.jsonl" --media-root "C:\data\media" --output-dir "C:\data\derived_media" --fps 1 --max-frames 32
python splits/make_splits.py --manifest "C:\data\derived_media\frames.jsonl" --output-dir "C:\data\splits" --seed 2026
```

Use `C:\data\derived_media` as `--frames-root` for readiness checks and training.

## Pre-training readiness check

Run this after creating the three split manifests and completing a copy of the dataset card. It checks card fields, manifest structure, labels, image paths, image readability, and group/hash/path separation. The report is a data-integrity check; it cannot confirm labels, rights, or model accuracy.

```powershell
.\.venv\Scripts\python.exe baseline/check_readiness.py --train-manifest "C:\data\splits\train.jsonl" --validation-manifest "C:\data\splits\validation.jsonl" --test-manifest "C:\data\splits\test.jsonl" --frames-root "C:\data\derived_media" --dataset-card "C:\data\dataset-card.md" --report "C:\data\outputs\readiness.json"
```

By default it opens and verifies every frame image, which takes time proportional to dataset size. Use `--skip-image-check` to check manifest paths and split integrity without decoding image contents. A one-class validation split is an error because validation-based threshold selection needs both labels; one-class test data is reported as a warning because ROC-AUC will be unavailable.

## Train

From the project root:

```powershell
.\.venv\Scripts\python.exe baseline/train.py --train-manifest "C:\data\splits\train.jsonl" --validation-manifest "C:\data\splits\validation.jsonl" --test-manifest "C:\data\splits\test.jsonl" --frames-root "C:\data\derived_media" --output-dir "C:\data\outputs\baseline" --epochs 10 --batch-size 32 --workers 0 --defer-test-evaluation
```

Training consumes image-frame paths from the split manifests. Video frames come from `preprocessing/extract_frames.py`; still images can be adapted with `preprocessing/prepare_images.py`. Training uses a SQLite disk index and loads images in batches, so it does not load the full image dataset into RAM. Each unique source media item gets equal total loss weight within its class, then real and synthetic classes get equal total weight. This helps keep the larger Comprehensive synthetic class from dominating training. Temporary index space is needed under the output directory. `--batch-size`, `--image-size`, `--workers`, `--device`, and `--epochs` control the main resource tradeoffs. A practical dataset still needs enough disk capacity, read bandwidth, and training time; no pipeline can promise unlimited scale.

The program checks group and content-hash separation again before training. It selects a checkpoint with validation metrics, saves that checkpoint's `validation_predictions.jsonl` for threshold selection, and can defer test metrics with `--defer-test-evaluation` until the selected threshold is applied. Outputs include `best_model.pt`, `training_report.json`, `validation_predictions.jsonl`, and `test_predictions.jsonl`.

## Inference

```powershell
.\.venv\Scripts\python.exe baseline/infer.py --input "C:\data\unseen_video.mp4" --checkpoint "C:\data\outputs\baseline\best_model.pt" --output "C:\data\outputs\predictions.jsonl"
```

`--input` can be a supported local video/image or a directory. Videos are sampled with FFmpeg using `--fps` and `--max-frames`; inference batches frames using `--batch-size`. The output is JSONL with model ID, media SHA-256, per-frame scores, a mean-frame media score, and limitations. Scores are not calibrated unless independently calibrated and must not be treated as proof of authenticity.

## Validation threshold and standalone evaluation
Choose a threshold from the selected model validation predictions first:

```powershell
.\.venv\Scripts\python.exe baseline/select_threshold.py --predictions "C:\data\outputs\baseline\validation_predictions.jsonl" --report "C:\data\outputs\baseline\threshold.json"
```

Use the `selected_threshold` value from that report for the separate test predictions. In the example below, replace `0.5` with that value. Do not choose a threshold from the test predictions.


The trainer writes labeled test predictions to `test_predictions.jsonl`. This standalone evaluator can report the same metrics from that file or another labeled prediction JSONL. It streams records and uses temporary disk space for exact ROC-AUC ranking; memory use does not grow with the number of prediction rows. Set the threshold using validation data before reporting held-out test results.

```powershell
.\.venv\Scripts\python.exe baseline/evaluate.py --predictions "C:\data\outputs\baseline\test_predictions.jsonl" --threshold 0.5 --report "C:\data\outputs\baseline\evaluation.json"
```


## Metrics and limitations

The evaluation report includes confusion counts, accuracy, synthetic precision/recall, real recall, balanced accuracy, and ROC-AUC when both classes are present. Results depend on label quality, group definitions, source/license terms, and whether the test set reflects the intended use. The starter baseline detects frame-level visual patterns; temporal analysis and broader domain validation are future improvements, not claims of this version.

## Still-image training path

The frame baseline trains on video frames or still images. Use the same consent, usage, label, source, and group_id metadata with ingestion, then create a derived image-frame manifest.

    python preprocessing/prepare_images.py --manifest "C:\data\manifest.jsonl" --media-root "C:\data\media" --output "C:\data\derived_images"
    python splits/make_splits.py --manifest "C:\data\derived_images\frames.jsonl" --output-dir "C:\data\image_splits" --seed 2026
    python baseline/train.py --train-manifest "C:\data\image_splits\train.jsonl" --validation-manifest "C:\data\image_splits\validation.jsonl" --test-manifest "C:\data\image_splits\test.jsonl" --frames-root "C:\data\derived_images" --output-dir "C:\data\outputs\image_baseline"

The image preparer copies source images without editing them, checks their SHA-256 against the ingestion manifest, and writes one frame per image into a separate derived folder. It does not detect or crop faces. Unknown labels remain in the frame manifest and are excluded by the split generator.
