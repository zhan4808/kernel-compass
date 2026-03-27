"""Stage 1 — NCU runner.

Launches a target script under Nsight Compute, writes a CSV report, then
parses it into a list of KernelProfile objects ready for Stage 2.

Typical usage:

    python -m profiling.ncu_runner \\
        --script kernels/mla_reconstruction.py \\
        --args "--model deepseek-v3 --ncu-mode" \\
        --output data/mla_v3_bs1.csv \\
        --label "mla_v3_decode_bs1"

Or from Python:

    from profiling.ncu_runner import run_ncu, parse_ncu_csv, load_profiles
    csv_path = run_ncu("kernels/mla_reconstruction.py", args="--ncu-mode",
                       output="data/out.csv")
    profiles = load_profiles(csv_path)
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import List, Optional

from profiling.metrics import KernelProfile


# ── NCU invocation ────────────────────────────────────────────────────────────

# Metrics we ask NCU to collect.  Extend as needed.
_NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_pct",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "gpu__time_duration.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
])


def _ncu_binary() -> str:
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if ncu is None:
        raise FileNotFoundError(
            "ncu not found on PATH.  Install Nsight Compute or activate the "
            "CUDA toolkit environment."
        )
    return ncu


def run_ncu(
    script: str,
    args: str = "",
    output: str = "data/ncu_out.csv",
    python: str = sys.executable,
    extra_ncu_flags: str = "",
    kernel_regex: Optional[str] = None,
) -> str:
    """Run *script* under NCU and write a CSV report to *output*.

    Returns the path to the written CSV.
    """
    ncu = _ncu_binary()

    cmd = [
        ncu,
        "--csv",
        "--metrics", _NCU_METRICS,
        "--target-processes", "all",
    ]
    if kernel_regex:
        cmd += ["--kernel-name", kernel_regex]
    if extra_ncu_flags:
        cmd += extra_ncu_flags.split()
    cmd += [python, script] + (args.split() if args else [])

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"[ncu_runner] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ncu_runner] stderr:\n{result.stderr[-2000:]}", file=sys.stderr)

    with open(output, "w") as f:
        f.write(result.stdout)

    print(f"[ncu_runner] CSV written to {output} ({len(result.stdout)} bytes)")
    return output


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_ncu_csv(csv_path: str) -> list[dict]:
    """Parse NCU --csv output into a list of raw row dicts."""
    with open(csv_path) as f:
        content = f.read()

    lines = [l for l in content.split("\n") if not l.startswith("==") and l.strip()]
    if not lines:
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    headers: list[str] = []
    rows: list[dict] = []
    for row in reader:
        if not headers:
            headers = [h.strip().strip('"') for h in row]
            continue
        if len(row) == len(headers):
            rows.append(dict(zip(headers, [v.strip().strip('"') for v in row])))
    return rows


def _safe_float(val: str, default: float = 0.0) -> float:
    if not val or val.lower() in ("n/a", ""):
        return default
    val = val.replace(",", "").replace("%", "")
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if val.endswith(suffix):
            try:
                return float(val[:-1]) * mult
            except ValueError:
                return default
    try:
        return float(val)
    except ValueError:
        return default


def _find_col(col_names: list[str], patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        for c in col_names:
            if pat.lower() in c.lower():
                return c
    return None


def rows_to_profiles(rows: list[dict]) -> List[KernelProfile]:
    """Convert raw NCU row dicts into KernelProfile objects.

    Multiple invocations of the same kernel are aggregated: duration is
    summed, other metrics are taken from the first invocation.
    """
    if not rows:
        return []

    col_names = list(rows[0].keys())
    fc = lambda pats: _find_col(col_names, pats)

    name_col   = fc(["Kernel Name", "kernel_name"])
    sm_col     = fc(["sm__throughput.avg.pct_of_peak", "Compute (SM) Throughput"])
    mem_col    = fc(["dram__throughput.avg.pct_of_peak", "Memory Throughput"])
    occ_col    = fc(["launch__occupancy", "Achieved Occupancy"])
    warp_col   = fc(["sm__warps_active.avg.pct_of_peak", "Warp Occupancy"])
    reg_col    = fc(["launch__registers_per_thread", "Registers Per Thread"])
    dur_col    = fc(["gpu__time_duration.sum", "Duration"])
    block_col  = fc(["launch__block_size", "Block Size"])
    grid_col   = fc(["launch__grid_size", "Grid Size"])
    l1h_col    = fc(["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit"])
    l1m_col    = fc(["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss"])
    l2h_col    = fc(["lts__t_sectors_op_read_lookup_hit"])
    l2m_col    = fc(["lts__t_sectors_op_read_lookup_miss"])
    tc_col     = fc(["smsp__inst_executed_pipe_tensor"])
    fadd_col   = fc(["sm__sass_thread_inst_executed_op_fadd"])
    fmul_col   = fc(["sm__sass_thread_inst_executed_op_fmul"])
    ffma_col   = fc(["sm__sass_thread_inst_executed_op_ffma"])

    # Aggregate by kernel name
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        name = row.get(name_col or "", "unknown") if name_col else "unknown"
        by_name[name].append(row)

    profiles: List[KernelProfile] = []
    for name, invs in by_name.items():
        rep = invs[0]
        g = lambda col: _safe_float(rep.get(col, "0")) if col else 0.0

        total_dur = sum(_safe_float(inv.get(dur_col, "0")) for inv in invs) if dur_col else 0.0
        # NCU reports duration in nanoseconds; convert to microseconds
        dur_us = total_dur / 1000.0

        l1h, l1m = g(l1h_col), g(l1m_col)
        l2h, l2m = g(l2h_col), g(l2m_col)
        l1_rate = 100.0 * l1h / (l1h + l1m) if (l1h + l1m) > 0 else 0.0
        l2_rate = 100.0 * l2h / (l2h + l2m) if (l2h + l2m) > 0 else 0.0

        profiles.append(KernelProfile(
            kernel_name=name,
            sm_pct=g(sm_col),
            mem_pct=g(mem_col),
            occupancy=g(occ_col),
            active_warps_pct=g(warp_col),
            registers_per_thread=int(g(reg_col)),
            block_size=int(g(block_col)),
            grid_size=int(g(grid_col)),
            duration_us=dur_us,
            invocation_count=len(invs),
            l1_hit_rate=l1_rate,
            l2_hit_rate=l2_rate,
            tensor_core_insts=g(tc_col),
            fp_fadd=g(fadd_col),
            fp_fmul=g(fmul_col),
            fp_ffma=g(ffma_col),
            raw=rep,
        ))

    return profiles


def load_profiles(csv_path: str) -> List[KernelProfile]:
    """Parse an NCU CSV and return KernelProfile list (no classification yet)."""
    return rows_to_profiles(parse_ncu_csv(csv_path))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: run NCU and emit KernelProfiles")
    parser.add_argument("--script", required=True, help="Python script to profile")
    parser.add_argument("--args", default="", help="Arguments to pass to the script")
    parser.add_argument("--output", default="data/ncu_out.csv", help="CSV output path")
    parser.add_argument("--label", default="", help="Label for the report (printed only)")
    parser.add_argument("--kernel-regex", default=None, help="Filter kernels by name regex")
    parser.add_argument("--parse-only", metavar="CSV",
                        help="Skip ncu, just parse an existing CSV and print summary")
    args = parser.parse_args()

    if args.parse_only:
        profiles = load_profiles(args.parse_only)
    else:
        csv_path = run_ncu(args.script, args=args.args, output=args.output,
                           kernel_regex=args.kernel_regex)
        profiles = load_profiles(csv_path)

    from profiling.bottleneck import classify, report
    profiles = classify(profiles)
    print(report(profiles, label=args.label or args.script))


if __name__ == "__main__":
    _cli()
