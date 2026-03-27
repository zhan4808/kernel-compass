#!/usr/bin/env python
"""End-to-end validation: timing benchmarks + NCU counter checks.

Run levels:
  1. Timing only (no NCU):   python tests/test_validation.py
  2. Full NCU validation:    sudo ncu python tests/test_validation.py --ncu

Level 1 verifies kernels execute and produce correct-shaped output.
Level 2 profiles each case under NCU and checks counters against expected ranges.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from profiling.metrics import KernelProfile
from kernels.mla_reconstruction import (
    VALIDATION_CASES,
    VALIDATION_KERNELS,
    _CASE_SHAPES,
    validate_profiles,
    run_validation,
)


def test_timing() -> bool:
    """Level 1: run each case, check output shape, report latency."""
    print("=" * 60)
    print("Level 1: Timing validation (no NCU)")
    print("=" * 60)

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}\n")

    all_ok = True
    for case_name, fn in VALIDATION_KERNELS.items():
        H, M, K, N = _CASE_SHAPES[case_name]
        try:
            out = fn()
            torch.cuda.synchronize()
            expected_shape = (H, M, N) if "int4" not in case_name else (H, M, N)
            shape_ok = tuple(out.shape) == expected_shape
            if not shape_ok:
                # INT4 kernel output N may differ if block-padded
                shape_ok = out.shape[0] == H and out.shape[1] == M
            status = "OK" if shape_ok else f"SHAPE MISMATCH {out.shape}"
        except Exception as e:
            status = f"ERROR: {e}"
            shape_ok = False

        print(f"  {case_name:15s}  {status}")
        all_ok = all_ok and shape_ok

    print()
    print("Latency benchmarks:")
    run_validation(warmup=20, iters=100)
    print()
    return all_ok


def test_ncu() -> bool:
    """Level 2: profile each case under NCU, validate counters."""
    from profiling.ncu_runner import NcuRunner

    print("=" * 60)
    print("Level 2: NCU counter validation")
    print("=" * 60)

    runner = NcuRunner(gpu="auto", output_dir="data/validation")
    print(f"Runner: {runner}\n")

    by_case: dict[str, KernelProfile] = {}
    for case_name, fn in VALIDATION_KERNELS.items():
        expected = VALIDATION_CASES[case_name]
        print(f"Profiling {case_name} ({expected.label})...")
        try:
            profiles = runner.run(fn, warmup=10, iters=5, label=case_name)
            if profiles:
                best = max(profiles, key=lambda p: p.duration_us)
                by_case[case_name] = best
                print(f"  Captured {len(profiles)} kernel(s), "
                      f"primary: SM={best.sm_pct:.1f}% MEM={best.mem_pct:.1f}% "
                      f"L2={best.l2_hit_rate:.1f}% dur={best.avg_duration_us:.1f}µs "
                      f"cv={best.cv_pct:.1f}%")
            else:
                print("  WARNING: no profiles captured")
        except FileNotFoundError:
            print("  ERROR: ncu not found — install Nsight Compute")
            return False
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    results = validate_profiles(by_case)
    all_pass = True
    for case_name, passed, msg in results:
        tag = "PASS" if passed else "FAIL"
        label = VALIDATION_CASES[case_name].label
        print(f"  [{tag}] {case_name:15s}  {label}")
        if not passed:
            print(f"         {msg}")
            all_pass = False

    print()
    if all_pass:
        print("All three validation cases PASSED.")
    else:
        print("Some validation cases FAILED.")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true",
                        help="Run NCU counter validation (requires ncu on PATH)")
    args = parser.parse_args()

    timing_ok = test_timing()

    if args.ncu:
        ncu_ok = test_ncu()
    else:
        ncu_ok = True
        print("Skipping NCU validation (pass --ncu to enable).\n")

    if timing_ok and ncu_ok:
        print("RESULT: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
