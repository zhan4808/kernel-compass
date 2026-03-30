"""Counter-grounded bottleneck classification."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from profiling.metrics import KernelProfile


class BottleneckClass(Enum):
    MEMORY_BOUND = "memory_bound"
    L2_BOUND = "l2_bound"
    COMPUTE_BOUND = "compute_bound"
    OCCUPANCY_LIMITED = "occupancy_limited"


@dataclass(frozen=True)
class Diagnosis:
    bottleneck: BottleneckClass
    confidence: str
    explanation: str


def classify_one(p: KernelProfile) -> Diagnosis:
    if p.dram_bw_pct < 40 and p.sm_pct < 40 and p.l2_hit_rate > 60:
        return Diagnosis(
            BottleneckClass.L2_BOUND,
            "high",
            (
                f"Weights are L2-resident (L2 hit rate {p.l2_hit_rate:.0f}%). "
                f"DRAM utilization is only {p.dram_bw_pct:.0f}%."
            ),
        )

    if p.dram_bw_pct > 60 and p.sm_pct < 50:
        return Diagnosis(
            BottleneckClass.MEMORY_BOUND,
            "high",
            (
                f"DRAM at {p.dram_bw_pct:.0f}% of peak. Kernel is HBM-bound. "
                "Use FP8 weight-only (W8A16) before INT4. "
                "Native FP8 W8A8 tensor-core path is future work."
            ),
        )

    if p.sm_pct > 65:
        return Diagnosis(
            BottleneckClass.COMPUTE_BOUND,
            "high",
            (
                f"SM utilization at {p.sm_pct:.0f}%. "
                "Kernel is compute-bound; dequant overhead can hurt."
            ),
        )

    if 0 < p.occupancy < 40:
        return Diagnosis(
            BottleneckClass.OCCUPANCY_LIMITED,
            "medium",
            f"Low occupancy ({p.occupancy:.0f}%). Tile/register pressure likely limits active warps.",
        )

    return Diagnosis(
        BottleneckClass.MEMORY_BOUND,
        "low",
        "No dominant bottleneck identified; defaulting to memory_bound.",
    )


def classify(profiles: list[KernelProfile]) -> list[KernelProfile]:
    for p in profiles:
        p.bottleneck = classify_one(p).bottleneck.value
    return profiles


_OBSERVATION: Dict[str, str] = {
    BottleneckClass.MEMORY_BOUND.value: (
        "**HBM-bandwidth limited.** Prefer FP8 weight-only (W8A16) first, then INT4/NVFP4 when footprint dominates. "
        "Native FP8 W8A8 path is not yet wired."
    ),
    BottleneckClass.L2_BOUND.value: (
        "**L2-resident.** HBM traffic is not the bottleneck; quantization usually does not help latency."
    ),
    BottleneckClass.COMPUTE_BOUND.value: (
        "**Compute limited.** Reduce FLOPs or improve tensor-core efficiency."
    ),
    BottleneckClass.OCCUPANCY_LIMITED.value: (
        "**Occupancy limited.** Reduce tile/register pressure to raise active warps."
    ),
}


def report(profiles: list[KernelProfile], label: str = "analysis") -> str:
    lines: list[str] = [f"# Bottleneck Report: {label}", ""]
    ordered = sorted(profiles, key=lambda p: -p.duration_us)
    lines.append(f"**Total unique kernels:** {len(ordered)}")
    lines.append("")
    for rank, p in enumerate(ordered, start=1):
        short = p.kernel_name if len(p.kernel_name) <= 80 else p.kernel_name[:40] + "..." + p.kernel_name[-37:]
        est = " *(estimated)*" if p.is_estimated else ""
        lines.append(f"### #{rank}: `{short}`")
        lines.append(f"- **Classification**: **{p.bottleneck}**{est}")
        lines.append(f"- SM throughput: {p.sm_pct:.1f}%")
        lines.append(f"- DRAM throughput: {p.mem_pct:.1f}%")
        lines.append(f"- L2 hit rate: {p.l2_hit_rate:.1f}%")
        lines.append(f"- Avg latency: {p.avg_duration_us:.1f} us (n={p.invocation_count})")
        lines.append(f"> {_OBSERVATION.get(p.bottleneck, 'No observation available.')}")
        lines.append("")

    counts: Dict[str, int] = defaultdict(int)
    for p in ordered:
        counts[p.bottleneck] += 1
    lines.append("## Summary Counts")
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def export_csv(profiles: list[KernelProfile], out_path: str) -> None:
    fields = [
        "kernel_name",
        "bottleneck",
        "source",
        "sm_pct",
        "mem_pct",
        "occupancy",
        "l1_hit_rate",
        "l2_hit_rate",
        "duration_us",
        "invocation_count",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in profiles:
            w.writerow({k: getattr(p, k) for k in fields})
