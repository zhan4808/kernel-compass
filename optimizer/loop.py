"""Stages 3-4 optimization loop with counter-grounded verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from profiling.bottleneck import BottleneckClass, classify, classify_one
from profiling.metrics import KernelProfile

_BC = BottleneckClass

PRECISION_TIERS: dict[str, dict] = {
    "bf16": {"bytes_per_param": 2, "hw_native": True, "dequant_overhead": False},
    "fp8": {"bytes_per_param": 1, "hw_native": True, "dequant_overhead": False},
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
    note: str = ""
    original_ms: float = 0.0
    optimized_ms: float = 0.0

    @property
    def speedup(self) -> float:
        if self.optimized_ms <= 0:
            return 0.0
        return self.original_ms / self.optimized_ms


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
    if bottleneck in (_BC.L2_BOUND, _BC.MEMORY_BOUND):
        out: list[GridConfig] = []
        l2 = bottleneck == _BC.L2_BOUND
        if current_precision in ("bf16", "fp16"):
            out.append(GridConfig("fp8", "FP8 W8A16 (hw-native)" if l2 else "FP8 W8A16 (hw-native, recommended)"))
            out.append(GridConfig("int4", "INT4 W4A16 (sw dequant)" if l2 else "INT4 W4A16 (aggressive)"))
        elif current_precision == "fp8" and not l2:
            out.append(GridConfig("int4", "INT4 W4A16 (aggressive)"))
        return out
    return []


def _make_kernel(shape: tuple, precision: str, tile_config: Optional[dict] = None) -> tuple[Callable, str]:
    import torch
    from kernels.baselines import batched_fp8_gemm, batched_int4_gemm, quantize_fp8, quantize_int4

    H, M, K, N = shape
    tc = tile_config or {}
    bm = tc.get("BLOCK_M", 16)
    bn = tc.get("BLOCK_N", 64)
    bk = tc.get("BLOCK_K", 128)

    if precision in ("bf16", "fp16"):
        dt = torch.bfloat16 if precision == "bf16" else torch.float16
        A = torch.randn(H, M, K, dtype=dt, device="cuda")
        W = torch.randn(H, K, N, dtype=dt, device="cuda")
        return (lambda: torch.bmm(A, W)), "bf16" if precision == "bf16" else "fp16"

    if precision == "fp8":
        A = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
        W_src = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
        W_fp8, scales = quantize_fp8(W_src)
        return (lambda: batched_fp8_gemm(A, W_fp8, scales, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)), "fp8"

    if precision == "int4":
        A = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
        W_src = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
        W_packed, scales = quantize_int4(W_src)
        return (lambda: batched_int4_gemm(A, W_packed, scales, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)), "int4"

    raise ValueError(f"Unknown precision: {precision}")


def _profile_kernel(runner, fn: Callable, shape: tuple, weight_dtype: str, warmup: int, iters: int, label: str) -> list[KernelProfile]:
    from profiling.ncu_runner import CuptiRunner

    if isinstance(runner, CuptiRunner):
        return runner.run(fn, warmup=warmup, iters=iters, label=label, case_shapes=shape, weight_dtype=weight_dtype)
    return runner.run(fn, warmup=warmup, iters=iters, label=label)


_PRIMARY_METRIC: dict[BottleneckClass, Optional[str]] = {
    _BC.MEMORY_BOUND: "dram_bw_pct",
    _BC.L2_BOUND: None,
    _BC.COMPUTE_BOUND: "sm_pct",
    _BC.OCCUPANCY_LIMITED: "occupancy_pct",
}


def verify(before: KernelProfile, after: KernelProfile, bottleneck: BottleneckClass, config: GridConfig, shape: tuple) -> OptimizationResult:
    after_diag = classify_one(after)
    lat_b = before.avg_duration_us
    lat_a = after.avg_duration_us
    lat_delta = (lat_a - lat_b) / lat_b * 100 if lat_b > 0 else 0.0
    metric_key = _PRIMARY_METRIC.get(bottleneck)
    m_b = getattr(before, metric_key, 0.0) if metric_key else 0.0
    m_a = getattr(after, metric_key, 0.0) if metric_key else 0.0
    m_delta = (m_a - m_b) / m_b * 100 if metric_key and m_b > 0 else 0.0
    base = dict(
        kernel_name=before.kernel_name,
        shape={"H": shape[0], "M": shape[1], "K": shape[2], "N": shape[3]},
        before=before,
        after=after,
        bottleneck_before=bottleneck,
        bottleneck_after=after_diag.bottleneck,
        config_tried={"precision": config.precision, "label": config.label},
        latency_delta_pct=lat_delta,
        primary_metric_delta_pct=m_delta,
    )

    if bottleneck == _BC.L2_BOUND:
        return OptimizationResult(
            **base,
            accepted=False,
            rejection_reason=(
                f"Weights are L2-resident (L2 hit rate {before.l2_hit_rate:.0f}%). "
                "Quantization saves HBM bandwidth that is not the bottleneck. "
                "Recommendation: cross-layer fusion to exceed L2 capacity, or keep BF16."
            ),
        )

    if bottleneck == _BC.MEMORY_BOUND:
        ok = lat_delta < -2.0 and after.sm_pct < 80.0
        reason = None if ok else (
            f"Latency did not improve ({lat_delta:+.1f}%). {config.label} is not faster than BF16 baseline."
            if lat_delta >= -2.0
            else f"SM spiked to {after.sm_pct:.0f}% (dequant overhead negated bandwidth savings)."
        )
        return OptimizationResult(**base, accepted=ok, rejection_reason=reason)

    if bottleneck == _BC.COMPUTE_BOUND:
        ok = m_delta < -2.0 and lat_delta < 5.0
        reason = None if ok else (f"SM changed {m_delta:+.1f}% (need <-2%)" if m_delta >= -2.0 else f"Latency regressed {lat_delta:+.1f}%")
        return OptimizationResult(**base, accepted=ok, rejection_reason=reason)

    if bottleneck == _BC.OCCUPANCY_LIMITED:
        ok = m_delta > 2.0 and lat_delta < 5.0
        reason = None if ok else (
            f"Occupancy changed {m_delta:+.1f}% (need >+2%)" if m_delta <= 2.0 else f"Latency regressed {lat_delta:+.1f}%"
        )
        return OptimizationResult(**base, accepted=ok, rejection_reason=reason)

    return OptimizationResult(**base, accepted=False, rejection_reason="Unknown bottleneck class")


def optimize_grid(
    kernel_fn: Callable,
    shape: tuple,
    current_precision: str = "bf16",
    max_configs: int = 10,
    warmup: int = 10,
    iters: int = 5,
) -> list[OptimizationResult]:
    from profiling.ncu_runner import CuptiRunner, NcuRunner, ncu_has_permissions

    use_cupti = not ncu_has_permissions()
    runner = CuptiRunner(gpu="auto") if use_cupti else NcuRunner(gpu="auto")
    profiles = _profile_kernel(runner, kernel_fn, shape, current_precision, warmup, iters, "baseline")
    if not profiles:
        print("  ERROR: no profiles captured for baseline")
        return []
    profile = max(profiles, key=lambda p: p.duration_us)
    diag = classify_one(profile)
    print(f"  Baseline: {diag.bottleneck.value} (confidence={diag.confidence})")
    print(f"    SM={profile.sm_pct:.1f}% DRAM={profile.dram_bw_pct:.1f}% L2={profile.l2_hit_rate:.1f}% lat={profile.avg_duration_us:.1f}us")
    configs = enumerate_configs(diag.bottleneck, current_precision)
    if not configs:
        print("  No candidate precision configs for this bottleneck class.")
        return []

    results: list[OptimizationResult] = []
    for i, cfg in enumerate(configs[:max_configs]):
        print(f"\n  [{i + 1}/{len(configs)}] Trying: {cfg.label}")
        new_fn, new_wdtype = _make_kernel(shape, cfg.precision)
        new_profiles = _profile_kernel(runner, new_fn, shape, new_wdtype, warmup, iters, cfg.precision)
        if not new_profiles:
            print("    ERROR: no profiles captured")
            continue
        new_profile = max(new_profiles, key=lambda p: p.duration_us)
        result = verify(profile, new_profile, diag.bottleneck, cfg, shape)
        results.append(result)
        print(f"    {'ACCEPTED' if result.accepted else 'REJECTED'}")
        print(f"    Latency: {profile.avg_duration_us:.1f}us -> {new_profile.avg_duration_us:.1f}us ({result.latency_delta_pct:+.1f}%)")
        if result.rejection_reason:
            print(f"    Reason: {result.rejection_reason}")
        if result.accepted:
            profile = new_profile
            current_precision = cfg.precision
            diag = classify_one(profile)
    return results


def optimize_llm(
    kernel_fn: Callable,
    shape: tuple,
    current_precision: str = "bf16",
    max_iters: int = 3,
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
    results: list[OptimizationResult] = []

    for _ in range(max_iters):
        cfgs = generate_configs(diag, profile, sd, current_precision=current_precision, n_candidates=n_candidates, force_heuristic=force_heuristic)
        if not cfgs:
            break
        accepted = False
        for raw in cfgs:
            prec = raw.get("precision", current_precision)
            tile = {k: v for k, v in raw.items() if k.startswith("BLOCK")}
            reason = raw.get("reasoning", prec.upper())
            new_fn, new_wdtype = _make_kernel(shape, prec, tile)
            new_profiles = _profile_kernel(runner, new_fn, shape, new_wdtype, warmup, iters, prec)
            if not new_profiles:
                continue
            new_profile = max(new_profiles, key=lambda p: p.duration_us)
            r = verify(profile, new_profile, diag.bottleneck, GridConfig(precision=prec, label=reason), shape)
            r.config_tried.update(tile)
            if reason:
                r.config_tried["reasoning"] = reason
            results.append(r)
            if r.accepted:
                profile = new_profile
                current_precision = prec
                diag = classify_one(profile)
                accepted = True
                break
        if not accepted:
            break
    return results


_DEMO_SHAPES: dict[str, tuple] = {
    "MLA 16 MB (L2-resident)": (128, 1, 128, 512),
    "MLA 128 MB (HBM-bound)": (128, 1, 128, 4096),
    "FFN decode (HBM-bound)": (1, 1, 4096, 8192),
}


def compare_modes(shapes: Optional[dict[str, tuple]] = None, warmup: int = 10, iters: int = 5) -> None:
    shapes = shapes or _DEMO_SHAPES
    rows = []
    for label, shape in shapes.items():
        print(f"\n=== {label} shape={shape} ===")
        fn_g, _ = _make_kernel(shape, "bf16")
        grid = optimize_grid(fn_g, shape, warmup=warmup, iters=iters)
        fn_l, _ = _make_kernel(shape, "bf16")
        llm = optimize_llm(fn_l, shape, warmup=warmup, iters=iters, force_heuristic=True)
        grid_accepted = [r for r in grid if r.accepted]
        llm_accepted = [r for r in llm if r.accepted]
        rows.append(
            {
                "label": label,
                "grid_tried": len(grid),
                "grid_best_us": min((r.after.avg_duration_us for r in grid_accepted), default=None),
                "llm_tried": len(llm),
                "llm_best_us": min((r.after.avg_duration_us for r in llm_accepted), default=None),
            }
        )
    print("\nShape                          Grid tried  Grid best   LLM tried   LLM best")
    print("-" * 72)
    for r in rows:
        gb = f"{r['grid_best_us']:.1f}us" if r["grid_best_us"] else "none"
        lb = f"{r['llm_best_us']:.1f}us" if r["llm_best_us"] else "none"
        print(f"{r['label']:<30s} {r['grid_tried']:>10d} {gb:>10s} {r['llm_tried']:>10d} {lb:>10s}")


def demo() -> None:
    print("=" * 72)
    print("kernel-compass Stage 3: Counter-Grounded Grid Search")
    print("=" * 72)
    s16 = (128, 1, 128, 512)
    s128 = (128, 1, 128, 4096)
    print("\nCase 1: 16MB BF16 (L2-resident)")
    fn_16, _ = _make_kernel(s16, "bf16")
    r16 = optimize_grid(fn_16, s16, "bf16")
    print("\nCase 2: 128MB BF16 (HBM-bound)")
    fn_128, _ = _make_kernel(s128, "bf16")
    r128 = optimize_grid(fn_128, s128, "bf16")
    print("\nSUMMARY")
    for label, rows in [("16MB / L2", r16), ("128MB / HBM", r128)]:
        print(f"  {label}:")
        if not rows:
            print("    (no configs)")
            continue
        for r in rows:
            tag = "ACCEPTED" if r.accepted else "REJECTED"
            print(f"    [{tag}] {r.config_tried.get('label','')} lat {r.latency_delta_pct:+.1f}%")
            if r.rejection_reason:
                print(f"      {r.rejection_reason}")


_CANDIDATE_PRIORITY = [
    _BC.MEMORY_BOUND.value,
    _BC.OCCUPANCY_LIMITED.value,
    _BC.COMPUTE_BOUND.value,
    _BC.L2_BOUND.value,
]


def select_candidate(profiles: list[KernelProfile]) -> Optional[KernelProfile]:
    ordered = sorted(profiles, key=lambda p: -p.duration_us)
    for bt in _CANDIDATE_PRIORITY:
        for p in ordered:
            if p.bottleneck == bt:
                return p
    return ordered[0] if ordered else None


Strategy = Callable[[KernelProfile], str]


def strategy_quantize(p: KernelProfile) -> str:
    return (
        f"Kernel '{p.kernel_name}' is memory-bound (DRAM {p.dram_bw_pct:.0f}%). "
        "Try FP8 first; INT4/NVFP4 only for aggressive compression."
    )


def strategy_l2_skip(p: KernelProfile) -> str:
    return (
        f"Kernel '{p.kernel_name}' is L2-resident (L2 {p.l2_hit_rate:.0f}%). "
        "Skip quantization; consider fusion or larger batches."
    )


def strategy_tile_size(p: KernelProfile) -> str:
    return f"Kernel '{p.kernel_name}' is occupancy-limited. Reduce tile sizes to improve occupancy."


def strategy_reduce_flops(p: KernelProfile) -> str:
    return f"Kernel '{p.kernel_name}' is compute-bound (SM {p.sm_pct:.0f}%). Reduce FLOPs or improve TC utilization."


_STRATEGY_MAP: dict[str, Strategy] = {
    _BC.MEMORY_BOUND.value: strategy_quantize,
    _BC.L2_BOUND.value: strategy_l2_skip,
    _BC.COMPUTE_BOUND.value: strategy_reduce_flops,
    _BC.OCCUPANCY_LIMITED.value: strategy_tile_size,
}


def propose(candidate: KernelProfile) -> str:
    return _STRATEGY_MAP.get(candidate.bottleneck, strategy_tile_size)(candidate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stages 3-4: counter-grounded optimization")
    parser.add_argument("--demo", action="store_true", help="Stage 3 killer demo (L2-resident vs HBM-bound)")
    parser.add_argument("--compare", action="store_true", help="Stage 4 comparison: grid vs LLM on three shapes")
    parser.add_argument("--grid", nargs=4, type=int, metavar=("H", "M", "K", "N"), help="Grid search on a single shape")
    parser.add_argument("--llm", nargs=4, type=int, metavar=("H", "M", "K", "N"), help="LLM-guided search on a single shape")
    parser.add_argument("--precision", default="bf16", help="Baseline precision (default: bf16)")
    parser.add_argument("--heuristic", action="store_true", help="Force heuristic fallback (no API call)")
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
    else:
        parser.print_help()
