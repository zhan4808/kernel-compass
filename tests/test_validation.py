"""Validation harness for kernel-compass."""

from __future__ import annotations

from typing import Dict

from kernels.mla_reconstruction import (
    _CASE_SHAPES,
    _CASE_WEIGHT_DTYPE,
    VALIDATION_CASES,
    VALIDATION_KERNELS,
    run_validation,
    validate_profiles,
)
from profiling.bottleneck import BottleneckClass, classify_one
from profiling.metrics import KernelProfile


def test_timing() -> bool:
    print("=" * 60)
    print("Level 1: Timing and shape sanity")
    print("=" * 60)
    rows = run_validation()
    ok = len(rows) == len(VALIDATION_KERNELS)
    print(f"Collected {len(rows)} timing rows")
    return ok


def test_ncu() -> tuple[bool, dict[str, KernelProfile]]:
    """Level 2 validation with NCU, fallback to CUPTI estimation."""
    from profiling.ncu_runner import _CUPTI_WARNING, CuptiRunner, NcuRunner, ncu_has_permissions

    print("=" * 60)
    print("Level 2: Counter validation")
    print("=" * 60)
    use_cupti = not ncu_has_permissions()
    if use_cupti:
        print("NCU hardware counters unavailable.")
        print(f"{_CUPTI_WARNING}\n")
        runner = CuptiRunner(gpu="auto")
    else:
        runner = NcuRunner(gpu="auto", output_dir="data/validation")

    by_case: dict[str, KernelProfile] = {}
    all_pass = True
    for case_name, fn in VALIDATION_KERNELS.items():
        expected = VALIDATION_CASES[case_name]
        shapes = _CASE_SHAPES[case_name]
        wdtype = _CASE_WEIGHT_DTYPE.get(case_name, "bf16")
        print(f"Profiling {case_name} ({expected.label})")
        if use_cupti:
            profiles = runner.run(fn, warmup=10, iters=5, label=case_name, case_shapes=shapes, weight_dtype=wdtype)
        else:
            profiles = runner.run(fn, warmup=10, iters=5, label=case_name)
        if not profiles:
            print("  [FAIL] no kernels captured")
            all_pass = False
            continue
        best = max(profiles, key=lambda p: p.duration_us)
        by_case[case_name] = best
        ok, notes = validate_profiles(case_name, best.sm_pct, best.dram_bw_pct, best.l2_hit_rate)
        print(f"  SM={best.sm_pct:.1f}% DRAM={best.dram_bw_pct:.1f}% L2={best.l2_hit_rate:.1f}%")
        for n in notes:
            print(f"    - {n}")
        if not ok:
            all_pass = False
            print("  [FAIL] outside expected range")
        else:
            print("  [PASS]")
    print()
    return all_pass, by_case


_EXPECTED_CLASS: Dict[str, BottleneckClass] = {
    "bf16_16mb": BottleneckClass.L2_BOUND,
    "int4_16mb": BottleneckClass.COMPUTE_BOUND,
    "fp8_16mb": BottleneckClass.L2_BOUND,
    "bf16_128mb": BottleneckClass.MEMORY_BOUND,
    "fp8_128mb": BottleneckClass.MEMORY_BOUND,
}


def test_classifier(by_case: dict[str, KernelProfile]) -> bool:
    print("=" * 60)
    print("Level 3: Classifier validation")
    print("=" * 60)
    all_pass = True
    for case_name, expected_class in _EXPECTED_CLASS.items():
        p = by_case.get(case_name)
        if p is None:
            print(f"  [FAIL] {case_name}: missing profile")
            all_pass = False
            continue
        diag = classify_one(p)
        ok = diag.bottleneck == expected_class
        print(f"  [{'PASS' if ok else 'FAIL'}] {case_name:12s} -> {diag.bottleneck.value:18s} ({diag.confidence})")
        if not ok:
            print(f"    expected {expected_class.value}, got {diag.bottleneck.value}")
            all_pass = False
    print()
    return all_pass


def main(enable_ncu: bool = False) -> int:
    timing_ok = test_timing()
    ncu_ok = True
    by_case: dict[str, KernelProfile] = {}
    if enable_ncu:
        ncu_ok, by_case = test_ncu()
    classifier_ok = True
    if by_case:
        classifier_ok = test_classifier(by_case)
    elif enable_ncu:
        classifier_ok = False

    if timing_ok and ncu_ok and classifier_ok:
        print("RESULT: ALL TESTS PASSED")
        return 0
    print("RESULT: TESTS FAILED")
    return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu", action="store_true", help="Run level 2/3 counter validation")
    args = parser.parse_args()
    raise SystemExit(main(enable_ncu=args.ncu))
