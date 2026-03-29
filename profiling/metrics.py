"""Core profiling data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    hbm_bw_gbs: float
    l2_cache_mb: float
    fp16_tflops: float


GPU_SPECS: dict[str, GpuSpec] = {
    "h100": GpuSpec(name="NVIDIA H100 SXM5", hbm_bw_gbs=3350.0, l2_cache_mb=50.0, fp16_tflops=989.0),
    "a100": GpuSpec(name="NVIDIA A100 SXM4", hbm_bw_gbs=2039.0, l2_cache_mb=40.0, fp16_tflops=312.0),
}


@dataclass
class KernelProfile:
    """One profiled kernel aggregate."""

    kernel_name: str
    sm_pct: float = 0.0
    mem_pct: float = 0.0
    occupancy: float = 0.0
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    tensor_core_ratio: float = 0.0
    registers_per_thread: int = 0
    block_size: int = 0
    grid_size: int = 0
    duration_us: float = 0.0
    invocation_count: int = 1
    cv_pct: float = 0.0
    bottleneck: str = ""
    source: str = "ncu"

    @property
    def avg_duration_us(self) -> float:
        return self.duration_us / max(self.invocation_count, 1)

    @property
    def sm_utilization(self) -> float:
        return self.sm_pct

    @property
    def occupancy_pct(self) -> float:
        return self.occupancy

    @property
    def dram_bw_pct(self) -> float:
        return self.mem_pct

    @property
    def is_estimated(self) -> bool:
        return self.source != "ncu"

    def summary(self) -> str:
        est = " [estimated]" if self.is_estimated else ""
        return (
            f"{self.kernel_name}: sm={self.sm_pct:.1f}% dram={self.mem_pct:.1f}% "
            f"l2={self.l2_hit_rate:.1f}% lat={self.avg_duration_us:.1f}us{est}"
        )


@dataclass
class BMMProfile:
    """Timing-level summary for reconstruction BMM."""

    label: str
    H: int
    M: int
    K: int
    N: int
    dtype: str
    median_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gbs: float = 0.0

    @classmethod
    def from_timing(
        cls,
        label: str,
        H: int,
        M: int,
        K: int,
        N: int,
        dtype: str,
        median_ms: float,
        weight_bytes: float,
    ) -> "BMMProfile":
        sec = max(median_ms, 1e-9) / 1e3
        flops = 2.0 * H * M * K * N
        tflops = flops / sec / 1e12
        bandwidth_gbs = weight_bytes / sec / 1e9
        return cls(
            label=label,
            H=H,
            M=M,
            K=K,
            N=N,
            dtype=dtype,
            median_ms=median_ms,
            tflops=tflops,
            bandwidth_gbs=bandwidth_gbs,
        )
