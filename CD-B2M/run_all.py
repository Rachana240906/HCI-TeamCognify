"""
run_all.py
==========
Master runner for CD-B2M — executes all phases in sequence.

Usage
-----
    python run_all.py                   # full run
    python run_all.py --smoke_test      # fast smoke test
    python run_all.py --skip_preprocess # skip if data already processed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT   = Path(__file__).parent


def run(script: str, extra_args: list[str] = None, timeout: int = 7200) -> int:
    cmd = [PYTHON, str(ROOT / "src" / script), "--config", "configs/config.yaml"]
    if extra_args:
        cmd += extra_args
    print(f"\n{'='*60}")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"EXIT={result.returncode}"
    print(f"[{status}] {script} finished in {elapsed:.1f}s")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test",      action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing if data/processed/ already exists")
    args = parser.parse_args()

    extra = ["--smoke_test"] if args.smoke_test else []
    ok = True

    steps = []

    if not args.skip_preprocess:
        steps.append(("preprocess.py", extra, 7200))

    steps += [
        ("controls.py",  [],    120),
        ("train.py",     extra, 7200),
        ("evaluate.py",  extra, 600),
        ("statistics.py", [],   60),
        ("visualize.py",  [],   120),
    ]

    results = {}
    for script, xargs, timeout in steps:
        rc = run(script, xargs, timeout=timeout)
        results[script] = rc
        if rc not in (0, 1):   # 1 = logging to stderr only (not a real error)
            print(f"WARNING: {script} returned {rc} — continuing anyway")

    print(f"\n{'='*60}")
    print("RUN SUMMARY")
    print(f"{'='*60}")
    for script, rc in results.items():
        status = "✓ OK" if rc in (0, 1) else f"✗ FAILED (rc={rc})"
        print(f"  {script:25s}  {status}")
    print(f"{'='*60}")
    print("Done. Check results/ for outputs.")


if __name__ == "__main__":
    main()
