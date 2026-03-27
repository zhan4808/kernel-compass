"""MLA reconstruction BMM kernels, profiler, and validation cases.

Ported from cache-barrier/profiling/profile_mla_reconstruction.py.

Architecture (per decode layer):
  BMM1 — Q absorption:    (H, bs, d_nope) @ (H, d_nope, kv_lora_rank)
  BMM2 — V reconstruction: (H, bs, kv_lora_rank) @ (H, kv_lora_rank, d_v)

Validation cases (from cache-barrier paper Section 5.5 / Table 11):
  1. cuBLAS FP16, weight=16 MB  → low DRAM (~35%), L2-resident
  2. INT4 Triton,  weight=16 MB  → high SM (~70%), low DRAM (~20%)
  3. cuBLAS FP16, weight=128 MB → high DRAM (~83%), HBM-bound

Usage:
    python -m kernels.mla_reconstruction --model deepseek-v3
    python -m kernels.mla_reconstruction --validate          # run three cases
    python -m kernels.mla_reconstruction --validate --check data/validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from profiling.metrics import BMMProfile, KernelProfile


# ── Model configs ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MLAConfig:
    name: str
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    num_layers: int


CONFIGS: dict[str, MLAConfig] = {
    "deepseek-v2-lite": MLAConfig(
        name="DeepSeek-V2-Lite",
        num_heads=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        num_layers=27,
    ),
    "deepseek-v3": MLAConfig(
        name="DeepSeek-V3",
        num_heads=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        num_layers=61,
    ),
}


# ── Benchmarking primitives ───────────────────────────────────────────────────

def bench_bmm(A: torch.Tensor, B: torch.Tensor,
              warmup: int = 20, iters: int = 100) -> float:
    """Return median latency (ms) for torch.bmm(A, B)."""
    for _ in range(warmup):
        torch.bmm(A, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        torch.bmm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


# ── Main profiler ─────────────────────────────────────────────────────────────

def profile_reconstruction(
    cfg: MLAConfig,
    batch_sizes: List[int],
    dtype: torch.dtype = torch.float16,
    warmup: int = 20,
    iters: int = 100,
) -> list[dict]:
    """Profile BMM1 and BMM2 for each batch size.  Returns list of result dicts."""
    device = "cuda"
    H, d_nope, d_v, d_lora = cfg.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    w_kc = torch.randn(H, d_nope, d_lora, dtype=dtype, device=device)
    w_vc = torch.randn(H, d_lora, d_v,    dtype=dtype, device=device)

    w_kc_bytes = w_kc.nelement() * w_kc.element_size()
    w_vc_bytes = w_vc.nelement() * w_vc.element_size()

    print("=" * 72)
    print(f"MLA Reconstruction: {cfg.name}")
    print(f"  H={H}  d_nope={d_nope}  d_v={d_v}  kv_lora_rank={d_lora}")
    print(f"  w_kc: {w_kc_bytes/1e6:.1f} MB   w_vc: {w_vc_bytes/1e6:.1f} MB")
    print(f"  dtype={dtype}  warmup={warmup}  iters={iters}")
    print("=" * 72)

    results = []
    for bs in batch_sizes:
        q_nope  = torch.randn(H, bs, d_nope, dtype=dtype, device=device)
        attn_out = torch.randn(H, bs, d_lora, dtype=dtype, device=device)

        bmm1_ms = bench_bmm(q_nope, w_kc, warmup, iters)
        bmm2_ms = bench_bmm(attn_out, w_vc, warmup, iters)

        p1 = BMMProfile.from_timing("bmm1", H, bs, d_nope, d_lora, "fp16", bmm1_ms, w_kc_bytes)
        p2 = BMMProfile.from_timing("bmm2", H, bs, d_lora, d_v,    "fp16", bmm2_ms, w_vc_bytes)

        total_ms = bmm1_ms + bmm2_ms
        full_recon_ms = total_ms * cfg.num_layers

        print(f"  bs={bs:>4}  BMM1={bmm1_ms:.4f}ms ({p1.tflops:.1f}T, {p1.bandwidth_gbs:.0f}GB/s)"
              f"  BMM2={bmm2_ms:.4f}ms ({p2.tflops:.1f}T, {p2.bandwidth_gbs:.0f}GB/s)"
              f"  total={total_ms:.4f}ms  full-model={full_recon_ms:.2f}ms")

        results.append({
            "model": cfg.name,
            "batch_size": bs,
            "bmm1_ms": bmm1_ms,
            "bmm2_ms": bmm2_ms,
            "recon_total_ms": total_ms,
            "bmm1_tflops": p1.tflops,
            "bmm2_tflops": p2.tflops,
            "bmm1_bw_gbs": p1.bandwidth_gbs,
            "bmm2_bw_gbs": p2.bandwidth_gbs,
            "full_model_recon_ms": full_recon_ms,
        })

    return results


def roofline_analysis(cfg: MLAConfig, batch_sizes: List[int]) -> None:
    """Print INT4 speedup estimate from roofline model (no GPU needed)."""
    H, d_nope, d_v, d_lora = cfg.num_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    w_vc_fp16 = H * d_lora * d_v * 2
    w_vc_int4 = H * d_lora * d_v // 2

    peak_tflops = 989e12
    peak_bw = 3.35e12
    crossover_ai = peak_tflops / peak_bw

    print(f"\nRoofline (H100 SXM5: {peak_tflops/1e12:.0f} TFLOPS, {peak_bw/1e9:.0f} GB/s)")
    print(f"  Crossover AI = {crossover_ai:.0f} FLOPS/byte")
    print(f"  {'BS':>6} {'FLOPs':>12} {'W_vc(FP16)':>12} {'AI':>8} {'Regime':>14} {'INT4 est':>10}")
    for bs in batch_sizes:
        flops = 2 * H * bs * d_lora * d_v
        act_bytes = (H * bs * d_lora + H * bs * d_v) * 2
        mem_fp16 = w_vc_fp16 + act_bytes
        ai = flops / mem_fp16
        regime = "MEMORY" if ai < crossover_ai else "COMPUTE"
        if ai < crossover_ai:
            speedup = mem_fp16 / (w_vc_int4 + act_bytes)
        else:
            speedup = 1.0
        print(f"  {bs:>6} {flops/1e9:>10.1f}G {w_vc_fp16/1e6:>10.1f}MB "
              f"{ai:>8.1f} {regime:>14} {speedup:>8.2f}x")


# ── Validation cases ──────────────────────────────────────────────────────────
#
# Three cases from cache-barrier paper (Section 5.5, Table 11) that establish
# ground truth for the counter-grounded classifier.  Each case is a
# self-contained callable that NcuRunner.run() can profile.
#
# Weight size formula: H * K * N * element_size
#   16 MB  → H=128, K=128, N=512,  FP16 → 128*128*512*2   = 16 777 216
#   128 MB → H=128, K=128, N=4096, FP16 → 128*128*4096*2  = 134 217 728

@dataclass(frozen=True)
class ValidationExpected:
    """Expected NCU counter ranges for a validation case."""
    dram_bw_pct: Tuple[float, float]   # (min, max) % of peak
    sm_pct: Tuple[float, float]
    l2_hit_rate: Tuple[float, float]
    label: str


VALIDATION_CASES: Dict[str, ValidationExpected] = {
    "fp16_16mb": ValidationExpected(
        dram_bw_pct=(15.0, 55.0),
        sm_pct=(10.0, 60.0),
        l2_hit_rate=(60.0, 100.0),
        label="cuBLAS FP16 BMM, weight=16 MB (L2-resident)",
    ),
    "int4_16mb": ValidationExpected(
        dram_bw_pct=(5.0, 35.0),
        sm_pct=(40.0, 90.0),
        l2_hit_rate=(30.0, 100.0),
        label="INT4 Triton W4A16 BMM, weight=16 MB (SM-saturated)",
    ),
    "fp16_128mb": ValidationExpected(
        dram_bw_pct=(60.0, 100.0),
        sm_pct=(10.0, 60.0),
        l2_hit_rate=(0.0, 50.0),
        label="cuBLAS FP16 BMM, weight=128 MB (HBM-bound)",
    ),
}

# Shapes: (H, M, K, N) per case.
_CASE_SHAPES: Dict[str, Tuple[int, int, int, int]] = {
    "fp16_16mb":  (128, 1, 128, 512),    # 128*128*512*2  = 16 MB
    "int4_16mb":  (128, 1, 128, 512),
    "fp16_128mb": (128, 1, 128, 4096),   # 128*128*4096*2 = 128 MB
}

# Per-process tensor cache so repeated calls (warmup + iters) don't re-allocate.
_CASE_CACHE: Dict[str, dict] = {}


def _ensure(case: str) -> dict:
    if case in _CASE_CACHE:
        return _CASE_CACHE[case]
    H, M, K, N = _CASE_SHAPES[case]
    A = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    t: dict = {"A": A, "W": W}
    if "int4" in case:
        from kernels.baselines import quantize_int4
        t["W_packed"], t["scales"] = quantize_int4(W)
        t["K"] = K
    _CASE_CACHE[case] = t
    return t


def case_fp16_16mb() -> torch.Tensor:
    """Validation case 1: cuBLAS FP16, weight ≈ 16 MB (L2-resident on H100)."""
    t = _ensure("fp16_16mb")
    return torch.bmm(t["A"], t["W"])


def case_int4_16mb() -> torch.Tensor:
    """Validation case 2: INT4 Triton W4A16, weight ≈ 16 MB (SM-saturated)."""
    t = _ensure("int4_16mb")
    from kernels.baselines import batched_int4_gemm
    return batched_int4_gemm(t["A"], t["W_packed"], t["scales"], t["K"])


def case_fp16_128mb() -> torch.Tensor:
    """Validation case 3: cuBLAS FP16, weight ≈ 128 MB (HBM-bound)."""
    t = _ensure("fp16_128mb")
    return torch.bmm(t["A"], t["W"])


VALIDATION_KERNELS: Dict[str, callable] = {
    "fp16_16mb":  case_fp16_16mb,
    "int4_16mb":  case_int4_16mb,
    "fp16_128mb": case_fp16_128mb,
}


def validate_profiles(
    profiles_by_case: Dict[str, KernelProfile],
) -> List[Tuple[str, bool, str]]:
    """Check profiled counters against expected ranges.

    Returns a list of (case_name, passed, message) tuples.
    """
    results: List[Tuple[str, bool, str]] = []
    for case_name, expected in VALIDATION_CASES.items():
        if case_name not in profiles_by_case:
            results.append((case_name, False, "no profile data"))
            continue

        p = profiles_by_case[case_name]
        fails: list[str] = []

        for field, (lo, hi) in [
            ("dram_bw_pct", expected.dram_bw_pct),
            ("sm_pct", expected.sm_pct),
            ("l2_hit_rate", expected.l2_hit_rate),
        ]:
            val = getattr(p, field)
            if not (lo <= val <= hi):
                fails.append(f"{field}={val:.1f}% not in [{lo:.0f}, {hi:.0f}]")

        if fails:
            results.append((case_name, False, "; ".join(fails)))
        else:
            results.append((case_name, True, "OK"))

    return results


def run_validation(
    warmup: int = 20,
    iters: int = 100,
) -> list[dict]:
    """Run all three validation cases as timing benchmarks (no NCU required).

    Returns per-case latency results.  For counter-level validation,
    use NcuRunner to profile the case_* functions and feed the results
    to validate_profiles().
    """
    results = []
    for case_name, fn in VALIDATION_KERNELS.items():
        H, M, K, N = _CASE_SHAPES[case_name]
        w_bytes = H * K * N * (1 if "int4" in case_name else 2)
        if "int4" in case_name:
            w_bytes = H * K * N // 2  # packed 4-bit

        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        flops = 2 * H * M * K * N
        bw = (H * K * N * 2 / 1e9) / (median_ms / 1e3) if median_ms > 0 else 0.0
        tf = (flops / 1e12) / (median_ms / 1e3) if median_ms > 0 else 0.0

        expected = VALIDATION_CASES[case_name]
        print(f"  {case_name:15s}  {median_ms:.4f} ms  {tf:.1f} TFLOPS  "
              f"{bw:.0f} GB/s  [{expected.label}]")

        results.append({
            "case": case_name,
            "median_ms": median_ms,
            "tflops": tf,
            "bandwidth_gbs": bw,
            "weight_mb": H * K * N * 2 / 1e6,
        })

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Profile MLA reconstruction BMMs")
    parser.add_argument("--model", choices=list(CONFIGS), default="deepseek-v2-lite")
    parser.add_argument("--batch-sizes", default="1,4,16,64,128,256,512")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ncu-mode", action="store_true",
                        help="Reduce iterations for NCU profiling")
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--validate", action="store_true",
                        help="Run three validation cases (timing only)")
    parser.add_argument("--check", metavar="CSV",
                        help="Validate NCU profiles against expected counter ranges")
    args = parser.parse_args()

    if args.check:
        _check_from_csv(args.check)
        return

    if args.validate:
        print("Validation cases (timing only — use NcuRunner for counter validation):")
        run_validation(
            warmup=3 if args.ncu_mode else args.warmup,
            iters=5 if args.ncu_mode else args.iters,
        )
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        return

    cfg = CONFIGS[args.model]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    warmup, iters = (3, 5) if args.ncu_mode else (args.warmup, args.iters)

    results = profile_reconstruction(cfg, batch_sizes, warmup=warmup, iters=iters)
    roofline_analysis(cfg, batch_sizes)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")


def _check_from_csv(csv_path: str) -> None:
    """Load NCU profiles from CSV and validate against expected ranges."""
    from profiling.ncu_runner import load_profiles
    from profiling.bottleneck import classify

    profiles = classify(load_profiles(csv_path))
    if not profiles:
        print(f"No profiles found in {csv_path}")
        return

    # Heuristic: match profiles to cases by kernel name patterns and shape
    by_case: Dict[str, KernelProfile] = {}
    for p in profiles:
        name_lower = p.kernel_name.lower()
        is_triton = "triton" in name_lower or "w4a16" in name_lower or "kernel_batched" in name_lower

        if is_triton and p.mem_pct < 40:
            by_case.setdefault("int4_16mb", p)
        elif not is_triton and p.mem_pct > 55:
            by_case.setdefault("fp16_128mb", p)
        elif not is_triton:
            by_case.setdefault("fp16_16mb", p)

    results = validate_profiles(by_case)
    all_pass = True
    for case_name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        label = VALIDATION_CASES[case_name].label
        print(f"  [{status}] {case_name:15s}  {label}")
        if not passed:
            print(f"         {msg}")
            all_pass = False

    if all_pass:
        print("\nAll validation cases passed.")
    else:
        print("\nSome validation cases failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
