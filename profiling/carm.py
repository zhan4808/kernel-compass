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


# H100: measured (cache-barrier carm_params.json). A100: scaled estimates.
CARM_PARAMS: dict[str, CarmParams] = {
    "h100": CarmParams(bw_hbm_tbs=3.146, bw_l2_tbs=6.3, c_eff_mb=36.0,
                       peak_tflops=989.0, t0_graph_us=2.795, t0_eager_us=15.4),
    "a100": CarmParams(bw_hbm_tbs=1.94, bw_l2_tbs=4.0, c_eff_mb=29.0,
                       peak_tflops=312.0, t0_graph_us=3.5, t0_eager_us=18.0),
}


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
