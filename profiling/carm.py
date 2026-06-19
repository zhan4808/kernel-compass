"""Measured cache-aware roofline model (CARM) + CUDA-graph timing.

Latency model (cache-barrier paper, Section 5.6; all parameters measured on
the target GPU, H100 defaults from profiling/measure_carm_params.py):

    t = t0 + max(B / BW(WS), F / P_peak)
    BW(WS) = BW_L2_eff   if WS < C_eff   (capacity-gated, LRU-effective)
             BW_HBM_eff  otherwise

plus an explicit per-launch fixed cost t0 (graph-captured vs eager), which
dominates microsecond-scale kernels and which per-launch CUDA-event timing
cannot see past (~15.5 us floor).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from profiling.metrics import GPU_SPECS, GpuSpec


@dataclass(frozen=True)
class CarmParams:
    bw_hbm_tbs: float       # effective HBM bandwidth (measured, not datasheet)
    bw_l2_tbs: float        # effective L2 serving bandwidth (GEMM pattern)
    c_eff_mb: float         # effective residency capacity (LRU), < nominal
    peak_tflops: float
    t0_graph_us: float      # fixed cost per launch under CUDA graphs
    t0_eager_us: float      # fixed cost per eager launch (incl. eventing)
    # GPU-parameterization (cache-barrier CUDA validation, 2026-06-19): which
    # precisions have native tensor-core MMA, and that MMA's peak relative to
    # bf16. A weight-only-quant kernel (W8A16/W4A16, bf16 compute) is bounded by
    # the in-core dequant ceiling regardless of native_mma; a matched kernel
    # (W8A8/W4A4) reaches native_peak_mult x peak only if its precision is
    # native, else it falls back to the dequant ceiling. This is what moves the
    # quant-vs-dense crossover across hardware (H100 has no native FP4; B200 does).
    native_mma: tuple[str, ...] = ()
    native_peak_mult: tuple[tuple[str, float], ...] = ()
    dequant_ceiling_tflops: float = 0.0   # in-core dequant ceiling (weight-only)


# H100: measured (cache-barrier profiling/measure_carm_params.py 2026-06-10;
#       ceilings from the CUDA MoE sweep profiling/cuda_validation 2026-06-19).
# bw_l2_tbs is reduction-slope L2 read BW; GEMM-pattern serving can be higher (~6 TB/s).
# A100: scaled estimates.  B200: PROJECTED from public Blackwell specs (native FP4,
#       ~96 MB L2, ~8 TB/s HBM3e) -- the Blackwell hook, NOT measured.
CARM_PARAMS: dict[str, CarmParams] = {
    "h100": CarmParams(bw_hbm_tbs=3.146, bw_l2_tbs=5.331, c_eff_mb=36.0,
                       peak_tflops=989.4, t0_graph_us=2.802, t0_eager_us=18.048,
                       native_mma=("int8", "fp8"),
                       native_peak_mult=(("int8", 2.0), ("fp8", 2.0)),
                       dequant_ceiling_tflops=422.9),
    "a100": CarmParams(bw_hbm_tbs=1.94, bw_l2_tbs=4.0, c_eff_mb=29.0,
                       peak_tflops=312.0, t0_graph_us=3.5, t0_eager_us=18.0,
                       native_mma=("int8",),
                       native_peak_mult=(("int8", 2.0),),
                       dequant_ceiling_tflops=140.0),
    "b200": CarmParams(bw_hbm_tbs=8.0, bw_l2_tbs=10.0, c_eff_mb=96.0,
                       peak_tflops=2250.0, t0_graph_us=2.5, t0_eager_us=16.0,
                       native_mma=("int8", "fp8", "fp4"),
                       native_peak_mult=(("int8", 2.0), ("fp8", 2.0), ("fp4", 4.0)),
                       dequant_ceiling_tflops=960.0),
}

# MLA W4A16 dequant packed-byte throughput (fitted, cache-barrier plot_cache_aware_roofline.py)
INT4_BLOCK_M = 16
INT4_WT_PACKED_BYTES = 128 * 128 * 512 // 2  # H*K*N/2 for MLA shape
INT4_R_DQ_BYTES_PER_S = 0.496e12  # fitted packed-byte dequant throughput

# FlagGems fused_moe W8A16 in-core conversion ceiling (warm NCU, Mixtral shape)
MOE_W8A16_PEAK_TFLOPS = 305.0


class CarmRegime(Enum):
    LAUNCH_BOUND = "launch_bound"    # t0 dominates; no data-path change helps
    L2_SERVED = "l2_served"          # WS < C_eff: quantization does not help
    HBM_BOUND = "hbm_bound"          # weight-byte-bound: W8A8 INT8-MMA wins
    COMPUTE_BOUND = "compute_bound"  # FP16/BF16 (or tuned INT8 pipelines)


@dataclass(frozen=True)
class CarmAdvice:
    regime: CarmRegime
    predicted_us: float
    recommendation: str


def params_for(gpu: str = "h100") -> CarmParams:
    return CARM_PARAMS[gpu]


def bw_ws_tbs(ws_bytes: float, p: CarmParams) -> float:
    """Capacity-gated bandwidth: a working set AT C_eff already thrashes (LRU)."""
    return p.bw_l2_tbs if ws_bytes < p.c_eff_mb * 1024 * 1024 else p.bw_hbm_tbs


def predict_fp16_recon_us(flops: float, fp16_bytes: float, gpu: str = "h100", graphed: bool = True) -> float:
    """FP16 reconstruction BMM: weight+act+output bytes, capacity-gated BW."""
    return predict_us(flops, fp16_bytes, gpu=gpu, graphed=graphed)


def predict_int4_recon_us(bs: int, gpu: str = "h100", graphed: bool = True) -> float:
    """INT4 W4A16 MLA kernel: dequant in-core ceiling (not bandwidth)."""
    import math

    p = params_for(gpu)
    t0 = p.t0_graph_us if graphed else p.t0_eager_us
    tiles = math.ceil(bs / INT4_BLOCK_M)
    return t0 + INT4_WT_PACKED_BYTES * tiles / INT4_R_DQ_BYTES_PER_S * 1e6


def validate_recon_mape(recon_points: list[dict], gpu: str = "h100") -> dict:
    """MAPE for FP16/INT4 vs carm_params.json recon_points."""
    rows = []
    fp16_errs, int4_errs = [], []
    for pt in recon_points:
        pred_f = predict_fp16_recon_us(pt["flops"], pt["fp16_bytes"], gpu=gpu)
        pred_i = predict_int4_recon_us(pt["bs"], gpu=gpu)
        rows.append({
            "bs": pt["bs"],
            "fp16_us": pt["fp16_us"],
            "fp16_pred": round(pred_f, 2),
            "int4_us": pt["int4_us"],
            "int4_pred": round(pred_i, 2),
        })
        fp16_errs.append(abs(pred_f - pt["fp16_us"]) / pt["fp16_us"])
        int4_errs.append(abs(pred_i - pt["int4_us"]) / pt["int4_us"])
    n = max(len(fp16_errs), 1)
    return {
        "fp16_mape_pct": round(sum(fp16_errs) / n * 100, 1),
        "int4_mape_pct": round(sum(int4_errs) / n * 100, 1),
        "rows": rows,
    }


@dataclass(frozen=True)
class MoeShape:
    E: int
    H: int
    I: int
    topk: int = 2


def moe_flops(tokens: int, shape: MoeShape) -> float:
    """SwiGLU MoE: GEMM1 (gate+up) + GEMM2 per routed token."""
    T, k, H, I = tokens, shape.topk, shape.H, shape.I
    gemm1 = 2 * T * k * (2 * H * (2 * I))
    gemm2 = 2 * T * k * (I * H)
    return float(gemm1 + gemm2)


# Graph-timed Mixtral anchors (cache-barrier fused_moe extended sweep, W8A16 fix)
_MOE_BF16_ANCHORS: list[tuple[int, float]] = [(16, 968), (64, 1224), (128, 1000), (256, 1242), (512, 5311)]
_MOE_W816_ANCHORS: list[tuple[int, float]] = [(16, 564), (64, 711), (128, 833), (256, 1209), (512, 1990)]

# Tuned-CUDA Marlin anchors (cache-barrier cuda_validation, vLLM 0.20.2, 2026-06-19).
# Unlike the Triton bf16 baseline (which scaled super-linearly and made quant look
# like a high-T win), the tuned-CUDA bf16 fused_experts is competent at high T, so
# weight-only quant LOSES once compute-bound -- the crossover the paper predicts.
_MOE_BF16_CUDA_ANCHORS: list[tuple[int, float]] = [
    (16, 953), (64, 962), (128, 999), (256, 1386), (512, 2210),
    (1024, 2404), (1536, 2641), (2048, 3377)]
_MOE_FP8_CUDA_ANCHORS: list[tuple[int, float]] = [
    (16, 502), (64, 525), (128, 568), (256, 1013), (512, 1791),
    (1024, 3071), (1536, 4475), (2048, 5690)]


def _interp_us(tokens: int, anchors: list[tuple[int, float]]) -> float:
    """Log-linear interpolation between measured graph-timed anchors."""
    import math

    if tokens <= anchors[0][0]:
        return anchors[0][1] * (tokens / anchors[0][0])
    if tokens >= anchors[-1][0]:
        t0, u0 = anchors[-2]
        t1, u1 = anchors[-1]
        slope = (math.log(u1) - math.log(u0)) / (math.log(t1) - math.log(t0))
        return u1 * math.exp(slope * (math.log(tokens) - math.log(t1)))
    for (t0, u0), (t1, u1) in zip(anchors, anchors[1:]):
        if t0 <= tokens <= t1:
            if t0 == t1:
                return u0
            frac = (math.log(tokens) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return u0 * ((u1 / u0) ** frac)
    return anchors[-1][1]


def predict_moe_bf16_us(tokens: int, shape: MoeShape | None = None, gpu: str = "h100") -> float:
    """Graph-timed Mixtral bf16 latency (interpolated measured anchors)."""
    del shape, gpu
    return _interp_us(tokens, _MOE_BF16_ANCHORS)


def predict_moe_w8a16_us(tokens: int, shape: MoeShape | None = None, gpu: str = "h100") -> float:
    """Graph-timed fixed W8A16 latency (interpolated measured anchors)."""
    del shape, gpu
    return _interp_us(tokens, _MOE_W816_ANCHORS)


def moe_crossover_tokens(
    shape: MoeShape | None = None,
    gpu: str = "h100",
    t_max: int = 2048,
) -> int | None:
    """Smallest T where interpolated W8A16 latency >= bf16 (quantized stops winning)."""
    del shape, gpu
    for T in range(1, t_max + 1):
        if predict_moe_w8a16_us(T) >= predict_moe_bf16_us(T):
            return T
    return None


def moe_crossover_tokens_measured(rows: list[dict]) -> int | None:
    """Crossover from measured sweep rows with w8a16_vs_bf16 (or w8a16/bf16 fields)."""
    for r in sorted(rows, key=lambda x: x["T"]):
        ratio = r.get("w8a16_vs_bf16")
        if ratio is None and r.get("bf16") and r.get("w8a16"):
            ratio = r["bf16"] / r["w8a16"]
        if ratio is not None and ratio < 1.0:
            return int(r["T"])
    return None


def predict_moe_bf16_cuda_us(tokens: int) -> float:
    """Graph-timed tuned-CUDA (Marlin) bf16 MoE latency (interpolated anchors)."""
    return _interp_us(tokens, _MOE_BF16_CUDA_ANCHORS)


def predict_moe_fp8_cuda_us(tokens: int) -> float:
    """Graph-timed tuned-CUDA fp8 W8A16 MoE latency (interpolated anchors)."""
    return _interp_us(tokens, _MOE_FP8_CUDA_ANCHORS)


def moe_crossover_tokens_cuda(t_max: int = 2048) -> int | None:
    """Smallest T where tuned-CUDA fp8 W8A16 stops beating bf16 (measured ~600).

    Unlike the Triton path (quant wins at all T because the bf16 baseline was
    pathologically slow at high T), the tuned-CUDA fp8 path crosses bf16 in the
    few-hundred-token range -- the cache-aware-roofline crossover, finally visible
    once the dense baseline is competent."""
    for T in range(1, t_max + 1):
        if predict_moe_fp8_cuda_us(T) >= predict_moe_bf16_cuda_us(T):
            return T
    return None


def predict_us(flops: float, bytes_moved: float, gpu: str = "h100", graphed: bool = True) -> float:
    p = params_for(gpu)
    t0 = p.t0_graph_us if graphed else p.t0_eager_us
    t_mem = bytes_moved / (bw_ws_tbs(bytes_moved, p) * 1e12) * 1e6
    t_comp = flops / (p.peak_tflops * 1e12) * 1e6
    return t0 + max(t_mem, t_comp)


def advise(flops: float, weight_bytes: float, gpu: str = "h100", graphed: bool = True) -> CarmAdvice:
    """Three-way measurable deployment rule from the cache-barrier paper."""
    p = params_for(gpu)
    t0 = p.t0_graph_us if graphed else p.t0_eager_us
    t_mem = weight_bytes / (bw_ws_tbs(weight_bytes, p) * 1e12) * 1e6
    t_comp = flops / (p.peak_tflops * 1e12) * 1e6
    pred = t0 + max(t_mem, t_comp)

    if t0 > 2 * max(t_mem, t_comp):
        return CarmAdvice(CarmRegime.LAUNCH_BOUND, pred,
                          "Fixed launch cost dominates: fuse/batch launches or use CUDA graphs; "
                          "precision changes cannot help.")
    if t_comp > t_mem:
        return CarmAdvice(CarmRegime.COMPUTE_BOUND, pred,
                          "Compute-bound: stay FP16/BF16; weight quantization saves bytes that "
                          "are not the bottleneck and dequant variants add in-core cost.")
    if weight_bytes < p.c_eff_mb * 1024 * 1024:
        return CarmAdvice(CarmRegime.L2_SERVED, pred,
                          f"Working set fits effective L2 ({p.c_eff_mb:.0f} MB): weights are "
                          "L2-served, so quantization does not pay (measured 0.7x for W8A8). "
                          "Keep FP16/BF16.")
    return CarmAdvice(CarmRegime.HBM_BOUND, pred,
                      "Weight-byte-bound from HBM: W8A8 with INT8 tensor-core MMA "
                      "(no inner-loop dequant) wins 1.4-1.5x measured. Avoid W4A16/W8A16 "
                      "Triton dequant kernels (in-core ceiling).")


def graph_time_us(fn: Callable, n_inner: int = 20, reps: int = 50, warmup: int = 10) -> float:
    """Median per-launch latency under CUDA graphs.

    This is the only event-based timing that sees past the ~15.5 us eager
    launch/eventing floor; use it for any accept/reject latency decision.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) / n_inner * 1000.0


def spec_for(gpu: str = "h100") -> GpuSpec:
    return GPU_SPECS[gpu]
