"""KernelProfile dataclass — the shared currency between all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KernelProfile:
    """One profiled kernel invocation as seen by NCU."""

    kernel_name: str

    # Throughput (% of peak)
    sm_pct: float = 0.0       # SM / compute throughput
    mem_pct: float = 0.0      # DRAM throughput

    # Classification (filled by bottleneck.py)
    bottleneck: str = ""       # "COMPUTE-BOUND" | "MEMORY-BOUND" | "LATENCY-BOUND" | "BALANCED"

    # Occupancy
    occupancy: float = 0.0    # achieved occupancy %
    active_warps_pct: float = 0.0

    # Cache hit rates (0–100)
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0

    # Launch geometry
    block_size: int = 0
    grid_size: int = 0
    registers_per_thread: int = 0

    # Timing
    duration_us: float = 0.0  # microseconds, summed over all invocations
    invocation_count: int = 1

    # Tensor core activity
    tensor_core_insts: float = 0.0

    # FP op mix (instruction counts)
    fp_fadd: float = 0.0
    fp_fmul: float = 0.0
    fp_ffma: float = 0.0

    # Raw NCU row (preserved for downstream inspection)
    raw: dict = field(default_factory=dict)

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def fp_total(self) -> float:
        return self.fp_fadd + self.fp_fmul + self.fp_ffma

    @property
    def tensor_core_ratio(self) -> float:
        """Fraction of FP work done on tensor cores (0–1)."""
        denom = self.tensor_core_insts + self.fp_total
        return self.tensor_core_insts / denom if denom > 0 else 0.0

    @property
    def avg_duration_us(self) -> float:
        return self.duration_us / self.invocation_count if self.invocation_count else 0.0

    def summary(self) -> str:
        name = self.kernel_name if len(self.kernel_name) <= 60 else self.kernel_name[:28] + "…" + self.kernel_name[-29:]
        return (
            f"{name}\n"
            f"  [{self.bottleneck or '?'}]  SM={self.sm_pct:.1f}%  MEM={self.mem_pct:.1f}%  "
            f"occ={self.occupancy:.1f}%  L2={self.l2_hit_rate:.1f}%  "
            f"dur={self.duration_us:.1f}µs  n={self.invocation_count}"
        )


@dataclass
class BMMProfile:
    """Timing result for a single batched-matmul variant (used by kernels/)."""

    label: str          # e.g. "fp16_bmm1", "int4_bmm2"
    H: int
    M: int              # batch size
    K: int
    N: int
    dtype: str          # "fp16" | "int4"

    median_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gbs: float = 0.0

    @classmethod
    def from_timing(cls, label: str, H: int, M: int, K: int, N: int,
                    dtype: str, median_ms: float, weight_bytes: int) -> "BMMProfile":
        flops = 2 * H * M * K * N
        bw = (weight_bytes / 1e9) / (median_ms / 1e3) if median_ms > 0 else 0.0
        tf = (flops / 1e12) / (median_ms / 1e3) if median_ms > 0 else 0.0
        return cls(label=label, H=H, M=M, K=K, N=N, dtype=dtype,
                   median_ms=median_ms, tflops=tf, bandwidth_gbs=bw)
