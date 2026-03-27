"""Stage 2 — bottleneck classifier.

Consumes a list of KernelProfile objects (produced by ncu_runner.py) and
annotates each with a bottleneck label.  Also generates a markdown report.

Usage:
    from profiling.bottleneck import classify, report
    profiles = classify(profiles)
    print(report(profiles, label="decode_bs64"))
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Dict, List

from profiling.metrics import KernelProfile


# ── Classification ────────────────────────────────────────────────────────────

# H100 SXM5 roofline crossover: 989 TFLOPS FP16 / 3350 GB/s ≈ 295
# Use a tunable threshold dict keyed by GPU name substring.
_CROSSOVER_AI: Dict[str, float] = {
    "H100": 295.0,
    "A100": 312.0,   # 312 TFLOPS / 2000 GB/s
}
_DEFAULT_CROSSOVER_AI = 295.0


def _crossover_for(gpu_name: str) -> float:
    for key, val in _CROSSOVER_AI.items():
        if key.upper() in gpu_name.upper():
            return val
    return _DEFAULT_CROSSOVER_AI


def classify_one(p: KernelProfile) -> str:
    """Return bottleneck string for a single profile."""
    if p.sm_pct < 10 and p.mem_pct < 10:
        return "LATENCY-BOUND"
    if p.sm_pct > p.mem_pct * 1.3:
        return "COMPUTE-BOUND"
    if p.mem_pct > p.sm_pct * 1.3:
        return "MEMORY-BOUND"
    return "BALANCED"


def classify(profiles: List[KernelProfile]) -> List[KernelProfile]:
    """Annotate each KernelProfile with its bottleneck in-place, return list."""
    for p in profiles:
        p.bottleneck = classify_one(p)
    return profiles


# ── Markdown report ───────────────────────────────────────────────────────────

def report(profiles: List[KernelProfile], label: str = "analysis") -> str:
    lines: List[str] = []
    lines.append(f"# Kernel Bottleneck Analysis: {label}")
    lines.append("")

    if not profiles:
        lines.append("No profiles.")
        return "\n".join(lines)

    # Sort by duration descending
    ordered = sorted(profiles, key=lambda p: -p.duration_us)

    lines.append(f"**Total unique kernels:** {len(profiles)}")
    lines.append("")

    for rank, p in enumerate(ordered, 1):
        name = p.kernel_name
        short = name if len(name) <= 80 else name[:40] + "…" + name[-37:]
        lines.append(f"### #{rank}: `{short}`")
        lines.append("")
        lines.append(f"- **Classification**: **{p.bottleneck}**")
        lines.append(f"- SM throughput: {p.sm_pct:.1f}% of peak")
        lines.append(f"- Memory throughput: {p.mem_pct:.1f}% of peak")
        lines.append(f"- Invocations: {p.invocation_count}  |  Total time: {p.duration_us:.1f} µs  |  Avg: {p.avg_duration_us:.1f} µs")
        if p.occupancy > 0:
            lines.append(f"- Achieved occupancy: {p.occupancy:.1f}%")
        if p.l2_hit_rate > 0:
            lines.append(f"- L1 hit rate: {p.l1_hit_rate:.1f}%  |  L2 hit rate: {p.l2_hit_rate:.1f}%")
        if p.tensor_core_ratio > 0:
            lines.append(f"- Tensor core usage: {100 * p.tensor_core_ratio:.1f}%")
        if p.registers_per_thread > 0:
            lines.append(f"- Registers/thread: {p.registers_per_thread}  |  Block: {p.block_size}  |  Grid: {p.grid_size}")

        lines.append("")
        lines.append(_observation(p))
        lines.append("")

    # Summary counts
    counts = defaultdict(int)
    for p in profiles:
        counts[p.bottleneck] += 1
    lines.append("## Summary")
    lines.append("")
    for bt, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {bt}: {n} kernel(s)")
    lines.append("")

    return "\n".join(lines)


def _observation(p: KernelProfile) -> str:
    if p.bottleneck == "MEMORY-BOUND":
        suffix = ""
        if p.l2_hit_rate > 80:
            suffix = " **Note**: High L2 hit rate — weights may be L2-resident; quantization may not help."
        return (
            f"> **Memory-bandwidth limited.**  Reduce traffic (quantization, compression) "
            f"or improve cache locality.{suffix}"
        )
    if p.bottleneck == "COMPUTE-BOUND":
        return (
            "> **Compute limited.**  Reduce FLOPs (smaller dims, approximate ops) "
            "or increase tensor core utilization / use FP8/INT8 compute."
        )
    if p.bottleneck == "LATENCY-BOUND":
        return (
            "> **Low utilization** of both compute and memory.  Likely launch overhead "
            "or tiny problem size.  Consider kernel fusion or larger batch."
        )
    return "> **Balanced.**  Neither compute nor memory clearly dominates."


# ── CSV export ────────────────────────────────────────────────────────────────

def to_csv(profiles: List[KernelProfile]) -> str:
    """Serialize profiles to CSV string."""
    fields = [
        "kernel_name", "bottleneck", "sm_pct", "mem_pct",
        "occupancy", "l1_hit_rate", "l2_hit_rate",
        "duration_us", "invocation_count",
        "tensor_core_ratio", "block_size", "grid_size", "registers_per_thread",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for p in profiles:
        row = {f: getattr(p, f, "") for f in fields}
        row["tensor_core_ratio"] = f"{p.tensor_core_ratio:.4f}"
        writer.writerow(row)
    return buf.getvalue()
