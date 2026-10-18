"""Run audio-visual inference; untrained random weights are explicitly disclosed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="AV manifest from multimodal/preprocess.py")
    p.add_argument("--av-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Omit to create an untrained random model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int)
    args = p.parse_args()
    root = args.av_root.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    trained = False
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get("state_dict"), dict):
            p.error("checkpoint must contain a state_dict")
        if ckpt.get("classes", {"real": 0, "synthetic": 1}) != {"real": 0, "synthetic": 1}:
            p.error("checkpoint class mapping is incompatible; expected real=0, synthetic=1")
        model.load_state_dict(ckpt["state_dict"])
        trained = bool(ckpt.get("trained", False))
        image_size = args.image_size if args.image_size is not None else int(ckpt.get("image_size", 224))
    else:
        image_size = args.image_size if args.image_size is not None else 224
    model.eval()
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.manifest.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8", newline="\n") as dst, torch.no_grad():
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                frame_path = (root / row["frame"]).resolve()
                mel_path = (root / row["log_mel"]).resolve()
                if root not in frame_path.parents or root not in mel_path.parents:
                    raise ValueError("derived path escapes AV root")
                with Image.open(frame_path) as image:
                    image = transform(image.convert("RGB")).unsqueeze(0).to(device)
                mel = torch.from_numpy(np.load(mel_path, allow_pickle=False)).float().unsqueeze(0).unsqueeze(0).to(device)
                mask = torch.tensor([float(row["audio_present"])], device=device)
                result = model(image, mel, mask)
                record = {"sha256": row.get("sha256"), "relative_path": row.get("relative_path"),
                          "visual_synthetic_score": torch.sigmoid(result["visual_logits"]).item(),
                          "audio_synthetic_score": torch.sigmoid(result["audio_logits"]).item() if row["audio_present"] else None,
                          "fused_synthetic_score": torch.sigmoid(result["fused_logits"]).item(),
                          "audio_present": bool(row["audio_present"]), "model_trained": trained,
                          "interpretation": "model output; untrained random weights, not meaningful" if not trained else "model score, not proof of authenticity or manipulation"}
                dst.write(json.dumps(record) + "\n"); count += 1
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                p.error(f"{args.manifest}:{line_no}: {exc}")
    print(f"Wrote {count} predictions to {args.output}; model_trained={trained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
