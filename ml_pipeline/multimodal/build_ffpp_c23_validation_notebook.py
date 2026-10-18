"""Build a Colab handoff for frozen SDFVD2 visual evaluation on Kaggle FF++ C23."""
from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "multimodal" / "evaluate_ffpp_c23.py"
OUTPUT = ROOT / "kaggle_ffpp_c23_external_validation.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


evaluator_b64 = base64.b64encode(EVALUATOR.read_bytes()).decode("ascii")
cells = [
    markdown("""# FF++ C23 external visual evaluation

This notebook evaluates the five frozen SDFVD 2.0 paired-source CV checkpoints on the FaceForensics++ C23 subtree from the approved Kaggle collection. It does **not** train or tune on FF++. The evaluation uses the SDFVD2 validation-selected visual threshold for each fold and averages five evenly spaced frame scores into one score per video sequence.

The Kaggle listing is a mixed collection. The download stays in Colab temporary storage; only evaluation reports are written to Drive. The notebook extracts only the FF++ subtree. Before running, set `BASE` to the Drive folder containing your checkpoints and source bundle, and set `CV_RUN_FOLDER` to your paired-source CV run folder. Add Kaggle credentials to Colab Secrets as `KAGGLE_USERNAME` and `KAGGLE_KEY`; never paste credentials into a cell or save them to Drive.
"""),
    code("""from google.colab import drive, userdata
from pathlib import Path
import base64, json, os, shutil, subprocess, sys, time, zipfile

drive.mount('/content/drive')
# Change this one path in Colab to the Drive folder containing the dataset artifacts.
BASE = Path('/content/drive/MyDrive/Datasets')
CV_RUN_FOLDER = 'REPLACE_WITH_YOUR_SDFVD2_CV_RUN_FOLDER'
CV_ROOT = BASE / CV_RUN_FOLDER / 'group_cv'
SOURCE_B64 = BASE / 'poor_guys_deepfake_colab_source.b64'
DEST_ROOT = BASE / 'kaggle_ffpp_c23_external_validation'
DEST_ROOT.mkdir(parents=True, exist_ok=True)
assert CV_ROOT.is_dir(), f'Missing paired-source checkpoints: {CV_ROOT}'
assert SOURCE_B64.is_file(), f'Missing project source bundle: {SOURCE_B64}'

username = userdata.get('KAGGLE_USERNAME')
api_key = userdata.get('KAGGLE_KEY')
if not username or not api_key:
    raise RuntimeError('Add KAGGLE_USERNAME and KAGGLE_KEY in Colab Secrets, enable notebook access, then rerun this cell.')
kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(parents=True, exist_ok=True)
(kaggle_dir / 'kaggle.json').write_text(json.dumps({'username': username, 'key': api_key}), encoding='utf-8')
os.chmod(kaggle_dir / 'kaggle.json', 0o600)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'], check=True)

DOWNLOAD = Path('/content/kaggle_ffpp_download')
DOWNLOAD.mkdir(parents=True, exist_ok=True)
archives = list(DOWNLOAD.glob('*.zip'))
if not archives:
    subprocess.run(['kaggle', 'datasets', 'download', '-d',
                    'ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure',
                    '-p', str(DOWNLOAD)], check=True)
    archives = list(DOWNLOAD.glob('*.zip'))
if len(archives) != 1:
    raise RuntimeError(f'Expected one Kaggle ZIP, found {[p.name for p in archives]}')
ARCHIVE = archives[0]
print('Downloaded archive to temporary Colab storage:', ARCHIVE.name, f'{ARCHIVE.stat().st_size:,} bytes')
print('Frozen checkpoint root:', CV_ROOT)
"""),
    code("""# Extract only the FaceForensics++ subtree; skip unrelated datasets in the mixed collection.
WORK = Path('/content/kaggle_ffpp_c23_work')
EXTRACT_ROOT = WORK / 'extracted'
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as zf:
    members = [i for i in zf.infolist() if '/FaceForensics++/' in ('/' + i.filename.replace(chr(92), '/'))]
    if not members:
        raise RuntimeError('No FaceForensics++ subtree found in the Kaggle ZIP; stop and inspect the archive listing before evaluation.')
    needed = sum(i.file_size for i in members)
    free = shutil.disk_usage('/content').free
    if free < needed + 2 * 1024**3:
        raise RuntimeError(f'Insufficient temporary disk for FF++ extraction: need about {needed + 2 * 1024**3:,} bytes, have {free:,}')
    base = EXTRACT_ROOT.resolve()
    for item in members:
        target = (EXTRACT_ROOT / item.filename).resolve()
        if target != base and base not in target.parents:
            raise RuntimeError(f'Unsafe archive path: {item.filename}')
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)

roots = [p.parent for p in EXTRACT_ROOT.rglob('original_sequences')
         if p.is_dir() and (p.parent / 'manipulated_sequences').is_dir()]
if len(roots) != 1:
    raise RuntimeError(f'Expected one FF++ root after extraction, found {roots[:10]}')
FFPP_ROOT = roots[0]
print('FF++ subtree extracted:', FFPP_ROOT)
print('Files extracted:', len(members), '| extracted bytes:', needed)
"""),
    code(f"""# Reuse the existing project model definition and install the FF++ video-level evaluator.
SOURCE_ZIP = WORK / 'poor_guys_deepfake_colab_source.zip'
SOURCE_ZIP.write_bytes(base64.b64decode(SOURCE_B64.read_bytes().strip()))
PROJECT_ROOT = WORK / 'project_source'
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(SOURCE_ZIP) as zf:
    zf.extractall(PROJECT_ROOT)
PROJECT = PROJECT_ROOT / 'poor_guys_deepfake_project'
sys.path.insert(0, str(PROJECT / 'multimodal'))
EVALUATOR = PROJECT / 'multimodal' / 'evaluate_ffpp_c23.py'
EVALUATOR.write_bytes(base64.b64decode({evaluator_b64!r}))
print('Evaluator ready:', EVALUATOR)
"""),
    code("""# Run the five fold checkpoints; no fitting or threshold selection occurs on FF++.
OUTPUT = DEST_ROOT / f'external_eval_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
cmd = [sys.executable, str(EVALUATOR), '--frames-root', str(FFPP_ROOT),
       '--cv-root', str(CV_ROOT), '--output-dir', str(OUTPUT),
       '--frames-per-video', '5', '--batch-size', '64', '--workers', '2', '--device', 'auto']
print('Starting frozen visual-only external evaluation...')
subprocess.run(cmd, cwd=PROJECT, check=True)
print('Drive report:', OUTPUT / 'external_evaluation.json')
print('Drive predictions:', OUTPUT / 'external_predictions.csv')
"""),
    markdown("""## Interpreting the result

This is an external visual-only result on the FF++ subtree. It does not validate audio or multimodal fusion. The package also contains other datasets, which this notebook deliberately ignores. Its Kaggle description and CC0 label do not replace the original FaceForensics terms; retain the original non-commercial research/education conditions. Treat the result as exploratory until the extracted inventory and method counts are compared with your approved FF++ release and its source-video lineage is confirmed.
"""),
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
