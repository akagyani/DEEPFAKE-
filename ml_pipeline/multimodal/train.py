"""Train audio-visual late fusion using group-disjoint split manifests."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import build_model

CLASSES = {"real": 0, "synthetic": 1}


class VideoDataset(Dataset):
    def __init__(self, manifest: Path, root: Path, size: int):
        self.root, self.rows = root, []
        groups, hashes = set(), set()
        with manifest.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("label") not in CLASSES or not row.get("group_id") or not row.get("sha256"):
                    raise ValueError(f"{manifest}:{line_no}: expected binary label, group_id and sha256")
                digest = str(row["sha256"]).lower()
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    raise ValueError(f"{manifest}:{line_no}: invalid source SHA-256")
                if not isinstance(row.get("audio_present"), bool):
                    raise ValueError(f"{manifest}:{line_no}: audio_present must be boolean")
                if digest in hashes:
                    raise ValueError(f"{manifest}:{line_no}: duplicate source SHA-256")
                hashes.add(digest)
                groups.add(row["group_id"])
                for key in ("frame", "log_mel"):
                    path = (root / row[key]).resolve()
                    if root.resolve() not in path.parents or not path.is_file():
                        raise ValueError(f"{manifest}:{line_no}: missing/outside derived {key}: {path}")
                self.rows.append(row)
        self.groups = groups
        self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                             transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(self.root / row["frame"]) as image:
            image = self.transform(image.convert("RGB"))
        mel = torch.from_numpy(np.load(self.root / row["log_mel"], allow_pickle=False)).float().unsqueeze(0)
        if tuple(mel.shape) != (1, 64, 256):
            raise ValueError(f"Expected log-mel shape (1, 64, 256); got {tuple(mel.shape)}")
        return image, mel, torch.tensor(CLASSES[row["label"]], dtype=torch.float32), torch.tensor(float(row["audio_present"])), row["sha256"]


def evaluate(model, loader, device):
    model.eval()
    scores, labels, keys, audio_flags = [], [], [], []
    visual_scores, audio_scores = [], []
    with torch.no_grad():
        for image, mel, label, audio_present, digest in loader:
            outputs = model(image.to(device), mel.to(device), audio_present.to(device))
            scores.extend(torch.sigmoid(outputs["fused_logits"]).cpu().tolist())
            visual_scores.extend(torch.sigmoid(outputs["visual_logits"]).cpu().tolist())
            audio_scores.extend(torch.sigmoid(outputs["audio_logits"]).cpu().tolist())
            labels.extend(label.tolist())
            keys.extend(digest)
            audio_flags.extend(audio_present.to(torch.int).tolist())
    pairs = sorted(zip(scores, labels, keys), key=lambda x: x[0])
    positives, negatives = sum(labels), len(labels) - sum(labels)
    auc = None
    if positives and negatives:
        rank_sum, i = 0.0, 0
        while i < len(pairs):
            j = i + 1
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            average_rank = ((i + 1) + j) / 2.0
            rank_sum += average_rank * sum(pair[1] for pair in pairs[i:j])
            i = j
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    pred = [int(s >= .5) for s in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(pred, labels))
    fp, fn = negatives - tn, positives - tp
    return {"count": len(labels), "positive_synthetic": int(positives), "negative_real": int(negatives),
            "accuracy_at_0_5": (tp + tn) / len(labels) if labels else None,
            "balanced_accuracy_at_0_5": (((tp / positives) if positives else 0) + ((tn / negatives) if negatives else 0)) / 2 if labels else None,
            "roc_auc": auc, "threshold": 0.5,
            "predictions": [{"sha256": key, "label": "synthetic" if int(label) else "real",
                             "score_synthetic": float(fused), "score_visual": float(visual),
                             "score_audio": float(audio), "audio_present": bool(has_audio)}
                            for key, label, fused, visual, audio, has_audio in
                            zip(keys, labels, scores, visual_scores, audio_scores, audio_flags)]}


def write_predictions(path: Path, report: dict, score_column: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in report["predictions"]:
            if score_column == "score_audio" and not row["audio_present"]:
                continue
            stream.write(json.dumps({"sha256": row["sha256"], "label": row["label"],
                                     "score_synthetic": row[score_column],
                                     "audio_present": row["audio_present"]}) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--av-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--audio-seconds", type=float, default=8.0)
    p.add_argument("--pretrained-visual", action="store_true", help="Initialize the visual ResNet18 from ImageNet weights")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1 or args.workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.audio_seconds) or args.audio_seconds <= 0:
        p.error("epochs, batch-size, image-size must be positive; workers cannot be negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available")
    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        p.error("MPS was requested but is not available")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    root = args.av_root.expanduser().resolve()
    train = VideoDataset(args.train_manifest, root, args.image_size)
    val = VideoDataset(args.validation_manifest, root, args.image_size)
    test = VideoDataset(args.test_manifest, root, args.image_size)
    if not len(train) or not len(val) or not len(test):
        p.error("all three split manifests must contain at least one labeled video")
    if {row["label"] for row in val.rows} != set(CLASSES):
        p.error("validation split needs both real and synthetic videos for checkpoint selection and threshold calibration")
    if train.groups & val.groups or train.groups & test.groups or val.groups & test.groups:
        p.error("group_id leakage found across split manifests")
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    test_loader = DataLoader(test, shuffle=False, **loader_args)
    model = build_model(pretrained_visual=args.pretrained_visual).to(device)
    positives = sum(row["label"] == "synthetic" for row in train.rows)
    negatives = len(train) - positives
    if not positives or not negatives:
        p.error("training split needs both real and synthetic videos")
    pos_weight = torch.tensor([negatives / positives], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for image, mel, label, audio_present, _ in train_loader:
            image, mel, label, audio_present = image.to(device), mel.to(device), label.to(device), audio_present.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(image, mel, audio_present)
            audio_mask = audio_present.bool()
            audio_aux = criterion(result["audio_logits"][audio_mask], label[audio_mask]) if audio_mask.any() else result["audio_logits"].sum() * 0
            loss = criterion(result["fused_logits"], label) + .2 * criterion(result["visual_logits"], label) + .2 * audio_aux
            loss.backward(); optimizer.step(); loss_sum += loss.item() * len(label)
        validation = evaluate(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "train_loss": loss_sum / len(train), "validation": {k: v for k, v in validation.items() if k != "predictions"}}), flush=True)
        criterion_value = validation["roc_auc"]
        if criterion_value is None:
            criterion_value = validation["balanced_accuracy_at_0_5"] or 0.0
        if criterion_value > best_auc:
            best_auc = criterion_value
            torch.save({"state_dict": model.state_dict(), "trained": True, "epoch": epoch,
                        "selection_metric": "validation ROC AUC, fallback balanced accuracy", "image_size": args.image_size,
                        "audio_seconds": args.audio_seconds, "pretrained_visual": args.pretrained_visual,
                        "classes": {"real": 0, "synthetic": 1}}, out / "best_av_model.pt")
    checkpoint = torch.load(out / "best_av_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    validation = evaluate(model, val_loader, device)
    test_predictions = evaluate(model, test_loader, device)
    for split, report in (("validation", validation), ("test", test_predictions)):
        for branch, score_column in (("fused", "score_synthetic"), ("visual", "score_visual"), ("audio", "score_audio")):
            write_predictions(out / f"{split}_{branch}_predictions.jsonl", report, score_column)
    validation.pop("predictions", None)
    test_predictions.pop("predictions", None)
    (out / "validation_at_default_threshold.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (out / "training_summary.json").write_text(json.dumps({"trained": True, "best_epoch": checkpoint["epoch"],
        "validation_selection_metric": checkpoint["selection_metric"], "device": str(device), "train_count": len(train),
        "validation_count": len(val), "test_count": len(test), "seed": args.seed,
        "audio_seconds": args.audio_seconds, "image_size": args.image_size, "learning_rate": args.learning_rate,
        "pretrained_visual": args.pretrained_visual,
        "test_metrics_computed": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete. Validation/test score files and selected checkpoint: {out}; test metrics not computed by trainer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
