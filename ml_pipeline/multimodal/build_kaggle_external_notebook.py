"""Build a self-contained Colab notebook for test-only Kaggle evaluation."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "multimodal" / "evaluate_kaggle_external.py"
OUTPUT = ROOT / "kaggle_external_validation.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


evaluator_b64 = base64.b64encode(EVALUATOR.read_bytes()).decode("ascii")
cells = [
    md("""# Kaggle FF++ / Celeb-DF external visual validation\n\n"
       "This evaluates existing SDFVD 2.0 visual checkpoints on the Kaggle dataset's **test split only**. It does not train or tune on Kaggle. The Kaggle archive stays in Drive; extracted files and temporary source files stay in Colab runtime storage. Results are image-level because source-video mapping is not established.\n\n"
       "Before running: set `BASE` and `CV_RUN_FOLDER` to your Drive locations, then add Kaggle API credentials as Colab Secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY`. These are needed to download the dataset; never paste credentials into notebook cells or save them in Drive.\n"""),
    code("""from google.colab import drive, userdata
from pathlib import Path
import json, os, shutil, subprocess, sys, time, zipfile

drive.mount('/content/drive')
# Change this one path in Colab to the Drive folder containing the dataset artifacts.
BASE = Path('/content/drive/MyDrive/Datasets')
DEST = BASE / 'kaggle_mahsan8381_external_validation'
DEST.mkdir(parents=True, exist_ok=True)
CV_RUN_FOLDER = 'REPLACE_WITH_YOUR_SDFVD2_CV_RUN_FOLDER'
CV_ROOT = BASE / CV_RUN_FOLDER / 'group_cv'
SOURCE_B64 = BASE / 'poor_guys_deepfake_colab_source.b64'
assert CV_ROOT.is_dir(), f'Missing saved SDFVD2 fold checkpoints: {CV_ROOT}'
assert SOURCE_B64.is_file(), f'Missing project source bundle: {SOURCE_B64}'

username = userdata.get('KAGGLE_USERNAME')
api_key = userdata.get('KAGGLE_KEY')
if not username or not api_key:
    raise RuntimeError('Add Kaggle API credentials in Colab Secrets as KAGGLE_USERNAME and KAGGLE_KEY, then rerun this cell.')
kaggle_cfg = Path.home() / '.kaggle'
kaggle_cfg.mkdir(parents=True, exist_ok=True)
(kaggle_cfg / 'kaggle.json').write_text(json.dumps({'username': username, 'key': api_key}), encoding='utf-8')
os.chmod(kaggle_cfg / 'kaggle.json', 0o600)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)

archives = list(DEST.glob('deepfake-dataset*.zip'))
if not archives:
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'mahsan8381/deepfake-dataset', '-p', str(DEST)], check=True)
    archives = list(DEST.glob('deepfake-dataset*.zip'))
if len(archives) != 1:
    raise RuntimeError(f'Expected one Kaggle archive in {DEST}, found: {[p.name for p in archives]}')
ARCHIVE = archives[0]
print('Kaggle archive stored in Drive:', ARCHIVE, '| bytes:', ARCHIVE.stat().st_size)
print('Existing checkpoints:', CV_ROOT)
"""),
    code("""# Extract safely to ephemeral Colab storage. The ZIP remains in Drive for reproducibility.
WORK = Path('/content/kaggle_mahsan8381_external_validation')
EXTRACT_ROOT = WORK / 'extracted'
DATASET_ROOT = WORK / 'dataset'
if not (WORK / 'extraction_complete').is_file():
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        base = EXTRACT_ROOT.resolve()
        required = sum(info.file_size for info in zf.infolist())
        free = shutil.disk_usage('/content').free
        if free < required + 1024**3:
            raise RuntimeError(f'Not enough Colab temporary disk to extract safely: need about {required + 1024**3} bytes, have {free}')
        for item in zf.infolist():
            target = (EXTRACT_ROOT / item.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f'Unsafe path in Kaggle archive: {item.filename}')
        zf.extractall(EXTRACT_ROOT)
    tests = [p for p in EXTRACT_ROOT.rglob('test') if p.is_dir() and any((p / label).is_dir() for label in ('real', 'fake'))]
    if len(tests) != 1:
        raise RuntimeError(f'Could not identify a unique dataset/test directory with real/fake folders: {tests[:10]}')
    TEST_ROOT = tests[0]
    DATASET_ROOT = TEST_ROOT.parent
    (WORK / 'dataset_root.txt').write_text(str(DATASET_ROOT), encoding='utf-8')
    (WORK / 'test_root.txt').write_text(str(TEST_ROOT), encoding='utf-8')
    (WORK / 'extraction_complete').write_text('ok', encoding='utf-8')
else:
    DATASET_ROOT = Path((WORK / 'dataset_root.txt').read_text(encoding='utf-8'))
    TEST_ROOT = Path((WORK / 'test_root.txt').read_text(encoding='utf-8'))
print('Dataset root:', DATASET_ROOT)
print('Test-only evaluation root:', TEST_ROOT)
"""),
    code(f"""# Recover the project's model implementation from the Drive source bundle.
import base64
EVALUATOR_B64 = {evaluator_b64!r}
SOURCE_ZIP = WORK / 'poor_guys_deepfake_colab_source.zip'
SOURCE_ZIP.write_bytes(base64.b64decode(SOURCE_B64.read_bytes().strip()))
PROJECT_ROOT = WORK / 'project_source'
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(SOURCE_ZIP) as zf:
    zf.extractall(PROJECT_ROOT)
PROJECT = PROJECT_ROOT / 'poor_guys_deepfake_project'
sys.path.insert(0, str(PROJECT / 'multimodal'))
EVALUATOR_PATH = PROJECT / 'multimodal' / 'evaluate_kaggle_external.py'
EVALUATOR_PATH.write_bytes(base64.b64decode(EVALUATOR_B64))
print('Evaluator and model source are ready:', EVALUATOR_PATH)
"""),
    code("""# Evaluate each existing fold checkpoint once on the untouched test split.
OUTPUT = DEST / f'external_eval_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
cmd = [sys.executable, str(EVALUATOR_PATH),
       '--dataset-root', str(DATASET_ROOT), '--test-root', str(TEST_ROOT),
       '--cv-root', str(CV_ROOT), '--output-dir', str(OUTPUT), '--device', 'auto']
print('Starting test-only external evaluation...')
subprocess.run(cmd, cwd=PROJECT, check=True)
print('Report:', OUTPUT / 'external_evaluation.json')
print('Predictions:', OUTPUT / 'external_predictions.csv')
"""),
    code("""# Recalculate metrics after counting byte-identical test images once.
# This reuses saved predictions, without running inference again.
import csv, hashlib, json, statistics
from collections import defaultdict
from evaluate_kaggle_external import score_metrics

prediction_path = OUTPUT / 'external_predictions.csv'
image_hashes = {}
for image_path in sorted(TEST_ROOT.rglob('*')):
    if image_path.is_file() and image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
        h = hashlib.sha256()
        with image_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        image_hashes[image_path.relative_to(TEST_ROOT).as_posix()] = h.hexdigest()

unique = defaultdict(dict)
original_count = defaultdict(int)
with prediction_path.open(newline='', encoding='utf-8') as stream:
    for row in csv.DictReader(stream):
        digest = image_hashes.get(row['test_image'])
        if digest is None:
            raise RuntimeError(f"Prediction has no test image for hashing: {row['test_image']}")
        entry = {'label': 1 if row['label'] == 'synthetic' else 0,
                 'score': float(row['score_synthetic']),
                 'threshold': float(row['threshold_from_sdfvd2_validation']),
                 'path': row['test_image']}
        prior = unique[row['fold']].get(digest)
        if prior and prior['label'] != entry['label']:
            raise RuntimeError(f"Exact-byte duplicate has conflicting labels: {prior['path']} and {entry['path']}")
        unique[row['fold']].setdefault(digest, entry)
        original_count[row['fold']] += 1

deduplicated_folds = []
for fold, rows_by_hash in sorted(unique.items()):
    rows = list(rows_by_hash.values())
    labels = [row['label'] for row in rows]
    scores = [row['score'] for row in rows]
    thresholds = {row['threshold'] for row in rows}
    if len(thresholds) != 1:
        raise RuntimeError(f"Fold {fold} contains multiple thresholds")
    deduplicated_folds.append({'fold': fold, 'duplicate_test_rows_removed': original_count[fold] - len(rows),
                               **score_metrics(scores, labels, thresholds.pop())})

fold_mean = {}
for metric in ('accuracy', 'balanced_accuracy', 'roc_auc'):
    values = [row[metric] for row in deduplicated_folds if row[metric] is not None]
    fold_mean[metric] = {'mean': statistics.mean(values) if values else None,
                         'stdev': statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                         'fold_values': values}
deduplicated_report = {
    'evaluation': 'test split; one image per exact SHA-256 content per fold',
    'prediction_rows_before_deduplication': sum(original_count.values()),
    'prediction_rows_after_deduplication': sum(len(rows) for rows in unique.values()),
    'folds': deduplicated_folds,
    'fold_mean': fold_mean,
    'interpretation': 'Exact-byte deduplication removes repeated image weighting only; source-video and identity independence remain unverified.',
}
dedup_path = OUTPUT / 'exact_duplicate_deduplicated_evaluation.json'
dedup_path.write_text(json.dumps(deduplicated_report, indent=2) + '\\n', encoding='utf-8')
print(json.dumps({'deduplicated_test_counts': [row['count'] for row in deduplicated_folds], 'fold_mean': fold_mean}, indent=2))
print('Deduplicated report:', dedup_path)
"""),
    md("""## How to read the report\n\nThe fold mean and spread summarize five separately trained SDFVD2 checkpoints evaluated with their own SDFVD2 validation thresholds. The Kaggle test split was not used for training or threshold selection. The report includes exact-file duplicate checks across train/validation/test; this does not prove source-video or identity separation. Metrics are frame-image weighted, not video weighted, and should be described as exploratory cross-dataset visual results until source lineage and upstream licenses are independently confirmed. The final code cell recalculates metrics after removing repeated byte-identical test images, using the saved scores and no new inference.\n"""),
]
notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created {OUTPUT}")
