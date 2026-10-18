"""Train and evaluate a disk-backed, group-disjoint frame classifier baseline."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from model import build_model

CLASSES = {"real": 0, "synthetic": 1}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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


class FrameDataset(Dataset):
    """Images remain on disk; SQLite stores only paths and labels."""

    def __init__(self, db_path: Path, split: str, frames_root: Path, image_size: int):
        self.db_path, self.split, self.frames_root = str(db_path), split, frames_root
        with open_database(Path(self.db_path)) as db:
            self.length = db.execute("SELECT COUNT(*) FROM frames WHERE split=?", (split,)).fetchone()[0]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self._db = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_db"] = None
        return state

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __getitem__(self, index):
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
        row = self._db.execute(
            "SELECT image_path, label, weight, media_key FROM frames WHERE split=? AND ordinal=?",
            (self.split, index),
        ).fetchone()
        image_path, label, weight, media_key = row
        with Image.open(self.frames_root / image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, weight, media_key


class BlockShuffleSampler(Sampler[int]):
    """Shuffle bounded blocks so epoch ordering does not require randperm(N) RAM."""

    def __init__(self, length: int, seed: int, block_size: int = 8192):
        self.length, self.seed, self.block_size, self.epoch = length, seed, block_size, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks = list(range((self.length + self.block_size - 1) // self.block_size))
        rng.shuffle(blocks)
        for block in blocks:
            start = block * self.block_size
            indices = list(range(start, min(start + self.block_size, self.length)))
            rng.shuffle(indices)
            yield from indices


def add_manifest(db: sqlite3.Connection, path: Path, split: str, frames_root: Path) -> int:
    rows_added = 0
    ordinal = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, line {line_number}: invalid JSON: {exc}") from exc
            label = row.get("label")
            group = str(row.get("group_id") or "").strip()
            frames = row.get("frames")
            if label not in CLASSES or not group or not isinstance(frames, list) or not frames:
                raise ValueError(f"{path}, line {line_number}: expected binary label, group_id, and non-empty frames")
            db.execute("INSERT OR IGNORE INTO groups_seen VALUES (?, ?)", (group, split))
            existing = db.execute("SELECT split FROM groups_seen WHERE group_id=?", (group,)).fetchone()[0]
            if existing != split:
                raise ValueError(f"group_id {group!r} occurs in both {existing} and {split} manifests")
            digest = str(row.get("sha256") or "").lower().strip()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{path}, line {line_number}: expected a 64-character SHA-256 digest")
            prior = db.execute("SELECT group_id, split, label FROM hashes_seen WHERE sha256=?", (digest,)).fetchone()
            if prior and (prior[0] != group or prior[1] != split or prior[2] != CLASSES[label]):
                raise ValueError(f"duplicate content hash occurs in conflicting groups, splits, or labels: {digest}")
            db.execute("INSERT OR IGNORE INTO hashes_seen VALUES (?, ?, ?, ?)", (digest, group, split, CLASSES[label]))
            media_key = digest
            for frame_path in frames:
                relative = Path(str(frame_path))
                resolved = (frames_root / relative).resolve()
                if frames_root not in resolved.parents or not resolved.is_file():
                    raise ValueError(f"{path}, line {line_number}: frame is missing or outside --frames-root: {frame_path}")
                db.execute(
                    "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)",
                    (split, ordinal, relative.as_posix(), CLASSES[label], 1.0 / len(frames), media_key),
                )
                ordinal += 1
                rows_added += 1
            if rows_added % 5000 == 0:
                db.commit()
    db.commit()
    return rows_added


def apply_class_balanced_media_weights(db: sqlite3.Connection) -> dict[str, int]:
    """Give each class equal total loss weight and each media item equal weight within its class."""
    counts = {
        int(label): int(media_count)
        for label, media_count in db.execute(
            "SELECT label, COUNT(DISTINCT media_key) FROM frames WHERE split='train' GROUP BY label"
        )
    }
    if set(counts) != {0, 1} or min(counts.values()) < 1:
        raise ValueError("Class-balanced training requires at least one real and one synthetic media item")
    media_rows = db.execute(
        "SELECT media_key, label, COUNT(*) FROM frames WHERE split='train' GROUP BY media_key, label"
    )
    for media_key, label, frame_count in media_rows:
        weight = 1.0 / (counts[int(label)] * int(frame_count))
        db.execute("UPDATE frames SET weight=? WHERE split='train' AND media_key=?", (weight, media_key))
    db.commit()
    return {"real": counts[0], "synthetic": counts[1]}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_and_store(model, loader, device, db, split: str) -> None:
    db.execute("DELETE FROM predictions WHERE split=?", (split,))
    model.eval()
    with torch.inference_mode():
        for images, labels, _weights, media_keys in loader:
            scores = torch.sigmoid(model(images.to(device)).flatten()).cpu().tolist()
            entries = [(split, key, int(label), float(score)) for key, label, score in zip(media_keys, labels.tolist(), scores)]
            db.executemany(
                "INSERT INTO predictions(split, media_key, label, score_sum, frame_count) VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(split, media_key) DO UPDATE SET score_sum=score_sum+excluded.score_sum, frame_count=frame_count+1",
                entries,
            )
            db.commit()


def write_stored_predictions(db, split: str, prediction_path: Path) -> None:
    """Write labeled media-level predictions without calculating test metrics."""
    with prediction_path.open("w", encoding="utf-8", newline="\n") as out:
        cursor = db.execute(
            "SELECT label, score_sum / frame_count AS score, media_key FROM predictions WHERE split=? ORDER BY media_key",
            (split,),
        )
        for label, score, media_key in cursor:
            out.write(json.dumps({"media_sha256": media_key, "label": "synthetic" if label else "real", "score_synthetic": float(score)}) + "\n")


def report_metrics(db, split: str, threshold: float, prediction_path: Path | None = None) -> dict:
    tp = fp = tn = fn = 0
    rows = 0
    positive_rank_sum = positives = negatives = 0.0
    seen = 0
    pending_score = None
    tied_pos = tied_neg = 0
    cursor = db.execute(
            "SELECT label, score_sum / frame_count AS score, media_key FROM predictions WHERE split=? ORDER BY score_sum / frame_count, media_key",
        (split,),
    )
    out = prediction_path.open("w", encoding="utf-8", newline="\n") if prediction_path else None
    try:
        for label, score, media_key in cursor:
            label, score = int(label), float(score)
            rows += 1
            if score >= threshold:
                if label: tp += 1
                else: fp += 1
            else:
                if label: fn += 1
                else: tn += 1
            if out:
                out.write(json.dumps({"media_sha256": media_key, "label": "synthetic" if label else "real", "score_synthetic": score}) + "\n")
            if pending_score is not None and score != pending_score:
                group_count = tied_pos + tied_neg
                average_rank = seen + (group_count + 1) / 2
                positive_rank_sum += tied_pos * average_rank
                seen += group_count
                tied_pos = tied_neg = 0
            pending_score = score
            if label: tied_pos += 1; positives += 1
            else: tied_neg += 1; negatives += 1
        if pending_score is not None:
            group_count = tied_pos + tied_neg
            average_rank = seen + (group_count + 1) / 2
            positive_rank_sum += tied_pos * average_rank
            seen += group_count
    finally:
        if out:
            out.close()
    precision = tp / (tp + fp) if tp + fp else None
    recall_syn = tp / (tp + fn) if tp + fn else None
    recall_real = tn / (tn + fp) if tn + fp else None
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives) if positives and negatives else None
    return {
        "count": rows, "threshold": threshold,
        "confusion_matrix": {"true_synthetic": tp, "false_synthetic": fp, "true_real": tn, "missed_synthetic": fn},
        "accuracy": (tp + tn) / rows if rows else None,
        "precision_synthetic": precision, "recall_synthetic": recall_syn, "recall_real": recall_real,
        "balanced_accuracy": (recall_syn + recall_real) / 2 if recall_syn is not None and recall_real is not None else None,
        "roc_auc": auc,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--frames-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--defer-test-evaluation", action="store_true", help="Write test predictions but leave test metrics to the later evaluator")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1 or args.workers < 0 or args.learning_rate <= 0:
        p.error("epochs, batch size, image size, and learning rate must be positive; workers cannot be negative")
    if not 0 <= args.threshold <= 1:
        p.error("threshold must be between 0 and 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    frames_root = args.frames_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    with ExitStack() as stack:
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix=".frame-index-", dir=args.output_dir))
        db_path = Path(tmp) / "index.sqlite"
        db = stack.enter_context(open_database(db_path))
        with nullcontext(db):
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE frames(split TEXT, ordinal INTEGER, image_path TEXT, label INTEGER, weight REAL, media_key TEXT, PRIMARY KEY(split, ordinal))")
            db.execute("CREATE TABLE groups_seen(group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
            db.execute("CREATE TABLE hashes_seen(sha256 TEXT PRIMARY KEY, group_id TEXT NOT NULL, split TEXT NOT NULL, label INTEGER NOT NULL)")
            db.execute("CREATE TABLE predictions(split TEXT, media_key TEXT, label INTEGER, score_sum REAL, frame_count INTEGER, PRIMARY KEY(split, media_key))")
            db.execute("CREATE INDEX frames_media_idx ON frames(split, media_key)")
            counts = {}
            for split, manifest in (("train", args.train_manifest), ("validation", args.validation_manifest), ("test", args.test_manifest)):
                counts[split] = add_manifest(db, manifest, split, frames_root)
                if not counts[split]:
                    raise ValueError(f"{split} manifest contains no frames")
            train_classes = {row[0] for row in db.execute("SELECT DISTINCT label FROM frames WHERE split='train'")}
            if len(train_classes) != 2:
                raise ValueError("Training split must contain frames from both real and synthetic classes")
            train_media_counts = apply_class_balanced_media_weights(db)
            train_data = FrameDataset(db_path, "train", frames_root, args.image_size)
            val_data = FrameDataset(db_path, "validation", frames_root, args.image_size)
            test_data = FrameDataset(db_path, "test", frames_root, args.image_size)
            stack.callback(train_data.close)
            stack.callback(val_data.close)
            stack.callback(test_data.close)
            sampler = BlockShuffleSampler(len(train_data), args.seed)
            pin = device.type == "cuda"
            train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=pin)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
            model = build_model().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            best_score = -math.inf
            history = []
            for epoch in range(args.epochs):
                sampler.set_epoch(epoch)
                model.train()
                total_loss = 0.0
                batches = 0
                for images, labels, weights, _keys in train_loader:
                    images, labels, weights = images.to(device), labels.float().to(device), weights.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = criterion(model(images).flatten(), labels)
                    weights = weights.to(dtype=losses.dtype)
                    # Normalize against the dataset's expected batch weight, not this
                    # batch's observed weight; per-batch normalization would cancel
                    # class balancing whenever a batch contains only one class.
                    expected_batch_weight = 2.0 * len(images) / len(train_data)
                    loss = (losses * weights).sum() / max(expected_batch_weight, 1e-12)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1
                predict_and_store(model, val_loader, device, db, "validation")
                metrics = report_metrics(db, "validation", args.threshold)
                selection = metrics["roc_auc"]
                if selection is None:
                    selection = metrics["balanced_accuracy"]
                if selection is None:
                    selection = metrics["accuracy"]
                history.append({"epoch": epoch + 1, "train_loss": total_loss / max(batches, 1), "validation": metrics})
                if selection > best_score:
                    best_score = selection
                    temporary = checkpoint_path.with_suffix(".tmp")
                    torch.save({
                        "schema_version": "1.0", "model_id": "resnet18-frame-baseline",
                        "model_state_dict": model.state_dict(), "image_size": args.image_size,
                        "mean": MEAN, "std": STD, "threshold": args.threshold,
                        "selection_metric": "roc_auc_or_balanced_accuracy",
                    }, temporary)
                    os.replace(temporary, checkpoint_path)
                print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / max(batches, 1), "validation": metrics}), flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            predict_and_store(model, val_loader, device, db, "validation")
            validation_metrics = report_metrics(db, "validation", args.threshold, args.output_dir / "validation_predictions.jsonl")
            predict_and_store(model, test_loader, device, db, "test")
            if args.defer_test_evaluation:
                write_stored_predictions(db, "test", args.output_dir / "test_predictions.jsonl")
                test_metrics = None
            else:
                test_metrics = report_metrics(db, "test", args.threshold, args.output_dir / "test_predictions.jsonl")
            report = {
                "model_id": checkpoint["model_id"], "device": str(device), "counts_frames": counts,
                "class_counts_frames": {
                    split: {"real": db.execute("SELECT COUNT(*) FROM frames WHERE split=? AND label=0", (split,)).fetchone()[0],
                            "synthetic": db.execute("SELECT COUNT(*) FROM frames WHERE split=? AND label=1", (split,)).fetchone()[0]}
                    for split in ("train", "validation", "test")
                },
                "config": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                           "image_size": args.image_size, "learning_rate": args.learning_rate,
                           "workers": args.workers, "threshold": args.threshold,
                           "loss_weighting": "equal total weight per class; equal weight per unique media within each class",
                           "train_unique_media_by_class": train_media_counts},
                "history": history, "validation_metrics": validation_metrics, "test_metrics": test_metrics,
                "test_evaluation_deferred": args.defer_test_evaluation,
                "limitations": ["Frame classifier baseline; media scores are mean frame probabilities.", "Scores are not calibrated and are not proof of authenticity."],
            }
            (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)

