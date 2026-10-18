"""Save a random-initialized audio-visual checkpoint for interface smoke checks only."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import build_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained_visual=False)
    torch.save({"state_dict": model.state_dict(), "trained": False, "image_size": 224, "audio_seconds": 8.0,
                "classes": {"real": 0, "synthetic": 1}}, output)
    print(f"Saved UNTRAINED random initialization: {output}. Its scores are not meaningful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
