"""Package project source and docs for upload to Colab; excludes datasets and generated artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def included(path: Path) -> bool:
    parts = set(path.parts)
    if parts & {".venv", "venv", "__pycache__", "outputs", "derived_frames", "derived_av", ".git"}:
        return False
    if path.name.endswith((".pyc", ".pt", ".pth", ".zip", ".npy", ".jpg", ".jpeg", ".png", ".mp4", ".wav")):
        return False
    # The launcher notebook carries its own base64 source bundle. Including it
    # here would create a recursive ZIP -> notebook -> ZIP dependency.
    if path.suffix.lower() == ".ipynb":
        return False
    if path.suffix.lower() in {".py", ".md", ".txt", ".csv"}:
        return True
    return path.name == ".gitignore"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="Output ZIP (default: project root/poor_guys_deepfake_colab_source.zip)")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve() if args.output else root / "poor_guys_deepfake_colab_source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output and included(path.relative_to(root)))
    if not files:
        p.error("No source files found to package")
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("poor_guys_deepfake_project") / path.relative_to(root))
    print(f"Created source-only bundle: {output}")
    print(f"Included files: {len(files)}; dataset media and model outputs are excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
