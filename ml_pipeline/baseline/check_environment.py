"""Check local runtime requirements before starting an expensive pipeline run."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ffmpeg", action="store_true", help="Fail if FFmpeg is not available on PATH")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Device requested for training and inference")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": {"version": sys.version.split()[0], "executable": sys.executable,
                   "minimum_supported": "3.10", "supported": sys.version_info >= (3, 10)},
        "packages": {},
        "ffmpeg": {"available": False, "ffprobe_available": False, "required": args.require_ffmpeg},
        "accelerators": {"cuda": False, "mps": False},
        "requested_device": args.device,
        "limitations": [
            "This checks runtime availability only; it does not check dataset rights, labels, disk capacity, or model quality.",
            "CUDA and MPS availability does not guarantee enough accelerator memory for a chosen batch size.",
        ],
    }
    package_status: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    for distribution in ("torch", "torchvision", "Pillow", "numpy"):
        try:
            package_status[distribution] = {"installed": True, "version": importlib.metadata.version(distribution)}
        except importlib.metadata.PackageNotFoundError:
            package_status[distribution] = {"installed": False, "version": None}
            errors.append(f"Missing Python package: {distribution}")
    for module_name, distribution in (("torch", "torch"), ("torchvision", "torchvision"), ("PIL", "Pillow"), ("numpy", "numpy")):
        if package_status[distribution]["installed"]:
            try:
                probe = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if probe.returncode:
                    raise RuntimeError((probe.stderr or probe.stdout or f"exit code {probe.returncode}").strip()[-1200:])
                package_status[distribution]["importable"] = True
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                package_status[distribution]["importable"] = False
                package_status[distribution]["import_error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"Python package {distribution} is installed but cannot be imported")
    report["packages"] = package_status

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    ffmpeg_status: dict[str, object] = {"available": ffmpeg is not None, "required": args.require_ffmpeg, "path": ffmpeg}
    if ffmpeg:
        try:
            result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=10, check=False)
            ffmpeg_status["version_line"] = result.stdout.splitlines()[0] if result.stdout else None
            if result.returncode:
                ffmpeg_status["available"] = False
                errors.append("FFmpeg was found but could not report its version")
        except (OSError, subprocess.TimeoutExpired) as exc:
            ffmpeg_status["available"] = False
            ffmpeg_status["error"] = str(exc)
            errors.append("FFmpeg was found but could not be started")
    ffmpeg_status["ffprobe_available"] = ffprobe is not None
    ffmpeg_status["ffprobe_path"] = ffprobe
    if ffprobe:
        try:
            probe = subprocess.run([ffprobe, "-version"], capture_output=True, text=True, timeout=10, check=False)
            ffmpeg_status["ffprobe_version_line"] = probe.stdout.splitlines()[0] if probe.stdout else None
            if probe.returncode:
                ffmpeg_status["ffprobe_available"] = False
                errors.append("ffprobe was found but could not report its version")
        except (OSError, subprocess.TimeoutExpired) as exc:
            ffmpeg_status["ffprobe_available"] = False
            ffmpeg_status["ffprobe_error"] = str(exc)
            errors.append("ffprobe was found but could not be started")
    elif args.require_ffmpeg:
        errors.append("ffprobe is required for AV preprocessing and was not found on PATH")
    if not ffmpeg and args.require_ffmpeg:
        errors.append("FFmpeg is required for video preparation and inference but was not found on PATH")
    report["ffmpeg"] = ffmpeg_status

    if package_status.get("torch", {}).get("importable"):
        probe_code = "import json, torch; print(json.dumps({'cuda': bool(torch.cuda.is_available()), 'mps': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())}))"
        try:
            probe = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True, timeout=30, check=False)
            if probe.returncode == 0:
                report["accelerators"] = json.loads(probe.stdout.strip().splitlines()[-1])
            else:
                errors.append("PyTorch imports but accelerator detection failed")
                report["accelerators_error"] = (probe.stderr or probe.stdout).strip()[-1200:]
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            errors.append(f"PyTorch accelerator detection failed: {type(exc).__name__}: {exc}")

    if args.device in {"cuda", "mps"} and not report["accelerators"].get(args.device, False):
        errors.append(f"Requested device {args.device!r} is unavailable")

    report["ready"] = not errors
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
