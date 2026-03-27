"""Stages 3–5 — optimization loop (skeleton).

Pipeline:
  Stage 1  ncu_runner.py   — collect NCU profiles → List[KernelProfile]
  Stage 2  bottleneck.py   — classify each kernel
  Stage 3  (this file)     — select optimization candidate
  Stage 4  (this file)     — propose + apply a kernel variant
  Stage 5  (this file)     — validate speedup; accept or revert

Entry point: run_loop(target_script, args)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from profiling.metrics import KernelProfile
from profiling.bottleneck import classify
from profiling.ncu_runner import run_ncu, load_profiles


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    kernel_name: str
    original_ms: float
    optimized_ms: float
    accepted: bool
    note: str = ""

    @property
    def speedup(self) -> float:
        return self.original_ms / self.optimized_ms if self.optimized_ms > 0 else 0.0


@dataclass
class LoopState:
    iteration: int = 0
    history: List[OptimizationResult] = field(default_factory=list)
    best_profiles: List[KernelProfile] = field(default_factory=list)


# ── Stage 3: candidate selection ─────────────────────────────────────────────

def select_candidate(profiles: List[KernelProfile]) -> Optional[KernelProfile]:
    """Return the kernel most worth optimizing.

    Heuristic: pick the slowest MEMORY-BOUND kernel (highest total duration).
    Falls back to slowest COMPUTE-BOUND, then anything.
    """
    ordered = sorted(profiles, key=lambda p: -p.duration_us)
    for bt in ("MEMORY-BOUND", "COMPUTE-BOUND", "BALANCED", "LATENCY-BOUND"):
        for p in ordered:
            if p.bottleneck == bt:
                return p
    return ordered[0] if ordered else None


# ── Stage 4: optimization proposal ───────────────────────────────────────────

# A Strategy takes a candidate KernelProfile and returns a human-readable
# description of what to try.  Real implementations would patch kernel source.
Strategy = Callable[[KernelProfile], str]


def strategy_quantize(p: KernelProfile) -> str:
    return (
        f"Kernel '{p.kernel_name}' is MEMORY-BOUND with L2 hit rate {p.l2_hit_rate:.1f}%. "
        "Proposal: apply INT4 weight quantization to reduce memory traffic."
    )


def strategy_tile_size(p: KernelProfile) -> str:
    return (
        f"Kernel '{p.kernel_name}' has occupancy {p.occupancy:.1f}% and "
        f"{p.registers_per_thread} registers/thread. "
        "Proposal: reduce tile size to increase occupancy."
    )


def strategy_fusion(p: KernelProfile) -> str:
    return (
        f"Kernel '{p.kernel_name}' is LATENCY-BOUND (low utilization). "
        "Proposal: fuse with adjacent elementwise ops to amortize launch overhead."
    )


_STRATEGY_MAP: dict[str, Strategy] = {
    "MEMORY-BOUND":  strategy_quantize,
    "COMPUTE-BOUND": strategy_tile_size,
    "LATENCY-BOUND": strategy_fusion,
    "BALANCED":      strategy_tile_size,
}


def propose(candidate: KernelProfile) -> str:
    strategy = _STRATEGY_MAP.get(candidate.bottleneck, strategy_tile_size)
    return strategy(candidate)


# ── Stage 5: validation ───────────────────────────────────────────────────────

def validate(
    original: KernelProfile,
    run_baseline_ms: Callable[[], float],
    run_optimized_ms: Callable[[], float],
    min_speedup: float = 1.05,
) -> OptimizationResult:
    """Run both variants, accept if speedup >= min_speedup."""
    orig_ms = run_baseline_ms()
    opt_ms  = run_optimized_ms()
    speedup = orig_ms / opt_ms if opt_ms > 0 else 0.0
    accepted = speedup >= min_speedup
    return OptimizationResult(
        kernel_name=original.kernel_name,
        original_ms=orig_ms,
        optimized_ms=opt_ms,
        accepted=accepted,
        note=f"speedup={speedup:.2f}x ({'accepted' if accepted else 'rejected, min=' + str(min_speedup) + 'x'})",
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop(
    target_script: str,
    script_args: str = "",
    ncu_output_dir: str = "data",
    max_iters: int = 5,
    min_speedup: float = 1.05,
) -> LoopState:
    """End-to-end optimization loop (stub — Stages 4/5 are not automated yet).

    Each iteration:
      1. Run NCU on target_script → CSV
      2. Parse + classify → List[KernelProfile]
      3. Select worst bottleneck
      4. Print optimization proposal
      5. (Manual step) apply the proposed change and re-run to validate
    """
    state = LoopState()

    for i in range(max_iters):
        state.iteration = i
        csv_path = os.path.join(ncu_output_dir, f"iter_{i:02d}.csv")

        print(f"\n{'='*60}")
        print(f"Iteration {i}")
        print(f"{'='*60}")

        # Stage 1
        csv_path = run_ncu(target_script, args=script_args, output=csv_path)

        # Stage 2
        profiles = classify(load_profiles(csv_path))
        state.best_profiles = profiles

        # Stage 3
        candidate = select_candidate(profiles)
        if candidate is None:
            print("No candidate found — stopping.")
            break

        print(f"\nCandidate: {candidate.summary()}")

        # Stage 4
        proposal = propose(candidate)
        print(f"\nProposal:\n  {proposal}")

        # Stage 5 — placeholder: manual validation required
        print("\n[Stage 5] Apply the proposal manually, then press Enter to re-profile...")
        input()

    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimization loop (Stages 3–5)")
    parser.add_argument("--script", required=True)
    parser.add_argument("--args", default="")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    args = parser.parse_args()

    run_loop(args.script, args.args, max_iters=args.iters, min_speedup=args.min_speedup)
