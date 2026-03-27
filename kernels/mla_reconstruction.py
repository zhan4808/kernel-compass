"""MLA reconstruction BMM kernels and profiler.

Ported from cache-barrier/profiling/profile_mla_reconstruction.py.

Architecture (per decode layer):
  BMM1 — Q absorption:    (H, bs, d_nope) @ (H, d_nope, kv_lora_rank)
  BMM2 — V reconstruction: (H, bs, kv_lora_rank) @ (H, kv_lora_rank, d_v)

Usage:
    python -m kernels.mla_reconstruction --model deepseek-v3
    python -m kernels.mla_reconstruction --model deepseek-v2-lite --ncu-mode
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional

import torch

from profiling.metrics import BMMProfile


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

    # Reconstruction weights (shared across batch sizes — same as runtime)
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

    # H100 SXM5 specs
    peak_tflops = 989e12
    peak_bw = 3.35e12
    crossover_ai = peak_tflops / peak_bw  # ~295

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
        print(f"  {bs:>6} {flops/1e9:>10.1f}G {w_vc_fp16/1e6:>10.1f}MB {ai:>8.1f} {regime:>14} {speedup:>8.2f}x")


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
    args = parser.parse_args()

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


if __name__ == "__main__":
    _cli()
