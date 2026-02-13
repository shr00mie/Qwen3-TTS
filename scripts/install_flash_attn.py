#!/usr/bin/env python3
"""
Print environment info and install instructions for flash-attn.
Optionally attempt: uv pip install flash-attn --no-build-isolation

Usage:
  uv run python scripts/install_flash_attn.py           # print info only
  uv run python scripts/install_flash_attn.py --install # then try install
"""
from __future__ import annotations

import argparse
import subprocess
import sys


WHEEL_FINDER_URL = "https://flashattn.dev"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show flash-attn install info and optionally install with --no-build-isolation."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run: uv pip install flash-attn --no-build-isolation",
    )
    args = parser.parse_args()

    # Environment info
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python:  {py_ver}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        cuda = getattr(torch.version, "cuda", None) or "N/A"
        print(f"CUDA:    {cuda}")
    except ImportError:
        print("PyTorch: not installed")
        print("CUDA:    N/A (install PyTorch first, then run this again)")

    print()
    print("Prebuilt wheels (recommended):")
    print(f"  1. Open {WHEEL_FINDER_URL}")
    print("  2. Select Platform, Python, PyTorch (match above), and CUDA")
    print("  3. Copy the 'uv pip install <url>' or 'pip install <url>' command")
    print("  4. Run it in this environment")
    print()

    if not args.install:
        print("To attempt install from source (requires matching CUDA toolkit):")
        print("  uv run python scripts/install_flash_attn.py --install")
        return

    print("Attempting: uv pip install flash-attn --no-build-isolation")
    print()
    result = subprocess.run(
        ["uv", "pip", "install", "flash-attn", "--no-build-isolation"],
    )
    if result.returncode != 0:
        print()
        print("Install failed. Use a prebuilt wheel instead:")
        print(f"  {WHEEL_FINDER_URL}")
        print("Or ensure system CUDA matches PyTorch's CUDA and try again with MAX_JOBS=4 if low on RAM.")
        sys.exit(1)
    print("flash-attn installed successfully.")


if __name__ == "__main__":
    main()
