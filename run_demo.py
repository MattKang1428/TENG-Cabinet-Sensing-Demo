"""Run the electrical and optical contact-recognition examples for Fig. 3."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("all", "electrical", "optical"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    branches = ("electrical", "optical") if args.branch == "all" else (args.branch,)
    for branch in branches:
        print(f"\n=== {branch} ===", flush=True)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(root / branch / "demo.py")], cwd=root)
        if result.returncode:
            return result.returncode
        print(f"Completed {branch} in {time.perf_counter() - started:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
