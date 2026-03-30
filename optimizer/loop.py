"""Stages 3-4 optimization loop with counter-grounded verification."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from profiling.bottleneck import BottleneckClass, classify_one
from profiling.metrics import KernelProfile

_BC = BottleneckClass

# Baseline for experiments: BF16. FP8 row describes the *current* Triton path only.
PRECISION_TIERS: dict[str, dict] = {
    "bf16": {"bytes_per_param": 2, "hw_native": True, "dequant_overhead": False},
    # Current implementation: FP8 weights, FP16 activations, FP16 compute in Triton
    # (W8A16). Does not use H100 native FP8 tensor cores (W8A8); see cuBLASLt/TE TODO.
    "fp8": {"bytes_per_param": 1, "hw_native": False, "dequant_overhead": False},
    "int8": {"bytes_per_param": 1, "hw_native": True, "dequant_overhead": False},
    "int4": {"bytes_per_param": 0.5, "hw_native": False, "dequant_overhead": True},
    "nvfp4": {"bytes_per_param": 0.5, "hw_native": True, "dequant_overhead": False},
}


@dataclass
class OptimizationResult:
    kernel_name: str
    shape: dict
    before: KernelProfile
    after: KernelProfile
    bottleneck_before: BottleneckClass
    bottleneck_after: BottleneckClass
    config_tried: dict
    latency_delta_pct: float
    primary_metric_delta_pct: float
    accepted: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class GridConfig:
    precision: str
    label: str


@dataclass
class LoopState:
    iteration: int = 0
    history: list[OptimizationResult] = field(default_factory=list)
    best_profiles: list[KernelProfile] = field(default_factory=list)


def enumerate_configs(bottleneck: BottleneckClass, current_precision: str) -> list[GridConfig]:
    # L2-resident: counter-grounded path rejects quantization; do not enumerate wasted runs.
    if bottleneck == _BC.L2_BOUND:
        return []
    if bottleneck == _BC.MEMORY_BOUND:
        if current_precision in ("bf16", "fp16"):
            return [
                GridConfig("fp8", "FP8 weight-only W8A16 (recommended before INT4)"),
                GridConfig("int4", "INT4 W4A16 (aggressive)"),
            ]
        if current_precision == "fp8":
            return [GridConfig("int4", "INT4 W4A16 (aggressive)")]
    return []


def _make_kernel(shape: tuple, precision: str, tile_config: Optional[dict] = None) -> tuple[Callable, str]:
    import torch
    from kernels.baselines import batched_fp8_gemm, batched_int4_gemm, quantize_fp8, quantize_int4

    H, M, K, N = shape
    tc = tile_config or {}
    bm, bn, bk = tc.get("BLOCK_M", 16), tc.get("BLOCK_N", 64), tc.get("BLOCK_K", 128)

    if precision in ("bf16", "fp16"):
        dt = torch.bfloat16 if precision == "bf16" else torch.float16
        A = torch.randn(H, M, K, dtype=dt, device="cuda")
        W = torch.randn(H, K, N, dtype=dt, device="cuda")
        return (lambda: torch.bmm(A, W)), ("bf16" if precision == "bf16" else "fp16")
    if precision == "fp8":
        A = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
        W = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
        W_fp8, scales = quantize_fp8(W)
        return (lambda: batched_fp8_gemm(A, W_fp8, scales, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)), "fp8"
    if precision == "int4":
        A = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
        W = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
        W_packed, scales = quantize_int4(W)
        return (lambda: batched_int4_gemm(A, W_packed, scales, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)), "int4"
    raise ValueError(f"Unknown precision: {precision}")


def _profile_kernel(runner, fn: Callable, shape: tuple, weight_dtype: str, warmup: int, iters: int, label: str) -> list[KernelProfile]:
    from profiling.ncu_runner import CuptiRunner

    if isinstance(runner, CuptiRunner):
        return runner.run(fn, warmup=warmup, iters=iters, label=label, case_shapes=shape, weight_dtype=weight_dtype)
    return runner.run(fn, warmup=warmup, iters=iters, label=label)


def verify(before: KernelProfile, after: KernelProfile, bottleneck: BottleneckClass, config: GridConfig, shape: tuple) -> OptimizationResult:
    after_diag = classify_one(after)
    lat_b = before.avg_duration_us
    lat_a = after.avg_duration_us
    lat_delta = (lat_a - lat_b) / lat_b * 100 if lat_b > 0 else 0.0
    base = dict(
        kernel_name=before.kernel_name,
        shape={"H": shape[0], "M": shape[1], "K": shape[2], "N": shape[3]},
        before=before,
        after=after,
        bottleneck_before=bottleneck,
        bottleneck_after=after_diag.bottleneck,
        config_tried={"precision": config.precision, "label": config.label},
        latency_delta_pct=lat_delta,
        primary_metric_delta_pct=0.0,
    )
    if bottleneck == _BC.L2_BOUND:
        return OptimizationResult(**base, accepted=False, rejection_reason="L2-resident: quantization saves HBM traffic that is not the bottleneck.")
    if bottleneck == _BC.MEMORY_BOUND:
        ok = lat_delta < -2.0 and after.sm_pct < 80.0
        return OptimizationResult(**base, accepted=ok, rejection_reason=None if ok else f"No useful speedup ({lat_delta:+.1f}%).")
    if bottleneck == _BC.COMPUTE_BOUND:
        ok = after.sm_pct < before.sm_pct - 2.0 and lat_delta < 5.0
        return OptimizationResult(**base, accepted=ok, rejection_reason=None if ok else "SM did not improve enough.")
    if bottleneck == _BC.OCCUPANCY_LIMITED:
        ok = after.occupancy > before.occupancy + 2.0 and lat_delta < 5.0
        return OptimizationResult(**base, accepted=ok, rejection_reason=None if ok else "Occupancy did not improve enough.")
    return OptimizationResult(**base, accepted=False, rejection_reason="Unknown bottleneck class.")


def optimize_grid(kernel_fn: Callable, shape: tuple, current_precision: str = "bf16", warmup: int = 10, iters: int = 5) -> list[OptimizationResult]:
    from profiling.ncu_runner import CuptiRunner, NcuRunner, ncu_has_permissions

    runner = CuptiRunner(gpu="auto") if not ncu_has_permissions() else NcuRunner(gpu="auto")
    profiles = _profile_kernel(runner, kernel_fn, shape, current_precision, warmup, iters, "baseline")
    if not profiles:
        return []
    profile = max(profiles, key=lambda p: p.duration_us)
    diag = classify_one(profile)
    print(f"  Baseline: {diag.bottleneck.value} | SM={profile.sm_pct:.1f}% DRAM={profile.dram_bw_pct:.1f}% L2={profile.l2_hit_rate:.1f}%")
    results: list[OptimizationResult] = []
    for cfg in enumerate_configs(diag.bottleneck, current_precision):
        new_fn, wd = _make_kernel(shape, cfg.precision)
        new_profiles = _profile_kernel(runner, new_fn, shape, wd, warmup, iters, cfg.precision)
        if not new_profiles:
            continue
        r = verify(profile, max(new_profiles, key=lambda p: p.duration_us), diag.bottleneck, cfg, shape)
        results.append(r)
        print(f"    {'ACCEPTED' if r.accepted else 'REJECTED'} {cfg.label} ({r.latency_delta_pct:+.1f}%)")
    return results


def optimize_llm(
    kernel_fn: Callable,
    shape: tuple,
    current_precision: str = "bf16",
    n_candidates: int = 3,
    warmup: int = 10,
    iters: int = 5,
    force_heuristic: bool = False,
) -> list[OptimizationResult]:
    from optimizer.llm import generate_configs
    from profiling.ncu_runner import CuptiRunner, NcuRunner, ncu_has_permissions

    runner = CuptiRunner(gpu="auto") if not ncu_has_permissions() else NcuRunner(gpu="auto")
    profiles = _profile_kernel(runner, kernel_fn, shape, current_precision, warmup, iters, "baseline")
    if not profiles:
        return []
    profile = max(profiles, key=lambda p: p.duration_us)
    diag = classify_one(profile)
    sd = {"H": shape[0], "M": shape[1], "K": shape[2], "N": shape[3]}
    cfgs = generate_configs(diag, profile, sd, current_precision=current_precision, n_candidates=n_candidates, force_heuristic=force_heuristic)
    out: list[OptimizationResult] = []
    for raw in cfgs:
        prec = raw.get("precision", current_precision)
        tile = {k: v for k, v in raw.items() if k.startswith("BLOCK")}
        new_fn, wd = _make_kernel(shape, prec, tile)
        new_profiles = _profile_kernel(runner, new_fn, shape, wd, warmup, iters, prec)
        if not new_profiles:
            continue
        r = verify(profile, max(new_profiles, key=lambda p: p.duration_us), diag.bottleneck, GridConfig(prec, raw.get("reasoning", prec)), shape)
        out.append(r)
        print(f"    {'ACCEPTED' if r.accepted else 'REJECTED'} {raw.get('reasoning', prec)} ({r.latency_delta_pct:+.1f}%)")
    return out


_DEMO_SHAPES: dict[str, tuple] = {
    "MLA 16 MB (L2-resident)": (128, 1, 128, 512),
    "MLA 128 MB (HBM-bound)": (128, 1, 128, 4096),
    "FFN decode (HBM-bound)": (1, 1, 4096, 8192),
}


def demo() -> None:
    print("=== Stage 3 demo ===")
    for label, shape in [("16MB case", (128, 1, 128, 512)), ("128MB case", (128, 1, 128, 4096))]:
        print(f"\n{label} shape={shape}")
        fn, _ = _make_kernel(shape, "bf16")
        optimize_grid(fn, shape, "bf16")


def compare_modes(shapes: Optional[dict[str, tuple]] = None) -> None:
    def _stats_from_output(text: str) -> tuple[int, Optional[int], int]:
        lines = [ln for ln in text.splitlines() if "ACCEPTED" in ln or "REJECTED" in ln]
        tried = len(lines)
        first_accept: Optional[int] = None
        for i, ln in enumerate(lines, start=1):
            if "ACCEPTED" in ln:
                first_accept = i
                break
        wasted = tried if first_accept is None else max(0, first_accept - 1)
        return tried, first_accept, wasted

    shapes = shapes or _DEMO_SHAPES
    print("Shape | Grid tried | Grid first_accept | Grid wasted | LLM tried | LLM first_accept | LLM wasted")
    for label, shape in shapes.items():
        h, m, k, n = shape
        grid_cmd = [sys.executable, "-m", "optimizer.loop", "--grid", str(h), str(m), str(k), str(n), "--precision", "bf16"]
        llm_cmd = [sys.executable, "-m", "optimizer.loop", "--llm", str(h), str(m), str(k), str(n), "--precision", "bf16", "--heuristic"]

        grid_proc = subprocess.run(grid_cmd, capture_output=True, text=True)
        llm_proc = subprocess.run(llm_cmd, capture_output=True, text=True)

        if grid_proc.returncode != 0 or llm_proc.returncode != 0:
            g_err = f"grid rc={grid_proc.returncode}" if grid_proc.returncode != 0 else "grid ok"
            l_err = f"llm rc={llm_proc.returncode}" if llm_proc.returncode != 0 else "llm ok"
            print(f"{label} | SKIPPED ({g_err}, {l_err})")
            continue

        g_tried, g_first, g_wasted = _stats_from_output(grid_proc.stdout)
        l_tried, l_first, l_wasted = _stats_from_output(llm_proc.stdout)
        g_first_txt = str(g_first) if g_first is not None else "none"
        l_first_txt = str(l_first) if l_first is not None else "none"
        print(
            f"{label} | {g_tried} | {g_first_txt} | {g_wasted} | "
            f"{l_tried} | {l_first_txt} | {l_wasted}"
        )


def run_loop(target_script: str, script_args: str = "", ncu_output_dir: str = "data", max_iters: int = 3, min_speedup: float = 1.05) -> LoopState:
    from profiling.bottleneck import classify
    from profiling.ncu_runner import load_profiles, run_ncu

    state = LoopState()
    os.makedirs(ncu_output_dir, exist_ok=True)
    for i in range(max_iters):
        csv_path = os.path.join(ncu_output_dir, f"iter_{i:02d}.csv")
        csv_path = run_ncu(target_script, args=script_args, output=csv_path)
        profiles = classify(load_profiles(csv_path))
        state.best_profiles = profiles
        if not profiles:
            break
        state.iteration = i
        candidate = max(profiles, key=lambda p: p.duration_us)
        print(f"iter {i}: {candidate.summary()} (min_speedup={min_speedup})")
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stages 3-4: counter-grounded optimization")
    parser.add_argument("--demo", action="store_true", help="Stage 3 killer demo (L2-resident vs HBM-bound)")
    parser.add_argument("--compare", action="store_true", help="Stage 4 comparison: grid vs LLM on three shapes")
    parser.add_argument("--grid", nargs=4, type=int, metavar=("H", "M", "K", "N"), help="Grid search on a single shape")
    parser.add_argument("--llm", nargs=4, type=int, metavar=("H", "M", "K", "N"), help="LLM-guided search on a single shape")
    parser.add_argument("--precision", default="bf16", help="Baseline precision (default: bf16)")
    parser.add_argument("--heuristic", action="store_true", help="Force heuristic fallback (no API call)")
    parser.add_argument("--script", default="", help="Legacy loop target script")
    parser.add_argument("--args", default="", help="Legacy loop target args")
    parser.add_argument("--iters", type=int, default=3, help="Legacy loop iterations")
    parser.add_argument("--min-speedup", type=float, default=1.05, help="Legacy loop speedup threshold")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.compare:
        compare_modes()
    elif args.grid:
        shape = tuple(args.grid)
        fn, _ = _make_kernel(shape, args.precision)
        optimize_grid(fn, shape, current_precision=args.precision)
    elif args.llm:
        shape = tuple(args.llm)
        fn, _ = _make_kernel(shape, args.precision)
        optimize_llm(fn, shape, current_precision=args.precision, force_heuristic=args.heuristic)
    elif args.script:
        run_loop(args.script, args.args, max_iters=args.iters, min_speedup=args.min_speedup)
    else:
        parser.print_help()
