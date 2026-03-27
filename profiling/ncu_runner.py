"""Stage 1 — NCU runner.

Launches a target script under Nsight Compute, writes a CSV report, then
parses it into a list of KernelProfile objects ready for Stage 2.

Two interfaces:

  1. Functional (existing):
       csv_path = run_ncu("kernels/mla_reconstruction.py", args="--ncu-mode")
       profiles = load_profiles(csv_path)

  2. Class-based (NcuRunner):
       runner  = NcuRunner(gpu="h100")
       profiles = runner.run(kernel_fn, warmup=10, iters=5)
       profiles = runner.profile_script("kernels/mla_reconstruction.py",
                                        args="--ncu-mode")
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
from dataclasses import dataclass
from typing import Callable, List, Optional

from profiling.metrics import KernelProfile


# ── GPU specs ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GpuSpec:
    name: str
    l2_cache_mb: float
    hbm_bw_gbs: float      # GB/s
    fp16_tflops: float      # TFLOPS

    @property
    def crossover_ai(self) -> float:
        """Arithmetic intensity (FLOP/byte) where roofline transitions."""
        return (self.fp16_tflops * 1e3) / self.hbm_bw_gbs


GPU_SPECS: dict[str, GpuSpec] = {
    "h100": GpuSpec("H100 SXM5", l2_cache_mb=50.0, hbm_bw_gbs=3350.0, fp16_tflops=989.0),
    "a100": GpuSpec("A100 SXM",  l2_cache_mb=40.0, hbm_bw_gbs=2039.0, fp16_tflops=312.0),
}


# ── NCU metrics we collect ────────────────────────────────────────────────────

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


# ── NCU binary discovery ─────────────────────────────────────────────────────

def _ncu_binary() -> str:
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if ncu is None:
        raise FileNotFoundError(
            "ncu not found on PATH.  Install Nsight Compute or activate the "
            "CUDA toolkit environment."
        )
    return ncu


# ── Functional interface (kept for backward compat) ───────────────────────────

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

        # Per-invocation durations for CV computation
        inv_durs = [_safe_float(inv.get(dur_col, "0")) for inv in invs] if dur_col else []
        total_dur_ns = sum(inv_durs)
        dur_us = total_dur_ns / 1000.0

        cv_pct = _cv_pct(inv_durs)

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
            cv_pct=cv_pct,
            l1_hit_rate=l1_rate,
            l2_hit_rate=l2_rate,
            tensor_core_insts=g(tc_col),
            fp_fadd=g(fadd_col),
            fp_fmul=g(fmul_col),
            fp_ffma=g(ffma_col),
            raw=rep,
        ))

    return profiles


def _cv_pct(values: list[float]) -> float:
    """Coefficient of variation (%) from a list of measurements."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return 100.0 * (var ** 0.5) / mean


def load_profiles(csv_path: str) -> List[KernelProfile]:
    """Parse an NCU CSV and return KernelProfile list (no classification yet)."""
    return rows_to_profiles(parse_ncu_csv(csv_path))


# ── NcuRunner class ───────────────────────────────────────────────────────────

def _resolve_gpu(gpu: str) -> str:
    """Map user-supplied gpu string to a key in GPU_SPECS."""
    if gpu != "auto":
        key = gpu.lower().replace(" ", "").replace("-", "")
        return key if key in GPU_SPECS else "h100"
    try:
        import torch
        name = torch.cuda.get_device_name(0).lower()
        for key in GPU_SPECS:
            if key in name:
                return key
    except Exception:
        pass
    return "h100"


class NcuRunner:
    """High-level interface for profiling GPU kernels via Nsight Compute.

    Usage::

        runner = NcuRunner(gpu="h100")

        # Profile a callable (must be importable, not a closure)
        profiles = runner.run(my_kernel_fn, warmup=10, iters=5)

        # Profile an existing script
        profiles = runner.profile_script("kernels/mla_reconstruction.py",
                                         args="--ncu-mode")

    Each returned KernelProfile includes cv_pct (coefficient of variation
    across invocations) and the full counter set from NCU.
    """

    def __init__(
        self,
        gpu: str = "auto",
        output_dir: str = "data",
    ):
        self.gpu_key = _resolve_gpu(gpu)
        self.spec = GPU_SPECS.get(self.gpu_key, GPU_SPECS["h100"])
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        kernel_fn: Callable,
        *args,
        warmup: int = 10,
        iters: int = 5,
        kernel_regex: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[KernelProfile]:
        """Profile *kernel_fn(*args)* under NCU.

        The callable must be a module-level function (importable by its
        ``__module__`` and ``__name__``).  It should allocate any required
        tensors internally; tensor arguments are not serialisable across
        the subprocess boundary.

        Returns a list of ``KernelProfile`` objects (one per unique GPU
        kernel captured by NCU).
        """
        label = label or kernel_fn.__name__
        script_path = self._gen_script(kernel_fn, args, warmup, iters)
        csv_path = os.path.join(self.output_dir, f"{label}.csv")

        run_ncu(
            script_path,
            output=csv_path,
            kernel_regex=kernel_regex,
            extra_ncu_flags="--profile-from-start off",
        )
        return load_profiles(csv_path)

    def profile_script(
        self,
        script: str,
        args: str = "",
        kernel_regex: Optional[str] = None,
        label: Optional[str] = None,
    ) -> List[KernelProfile]:
        """Run an existing Python script under NCU and return profiles."""
        label = label or os.path.splitext(os.path.basename(script))[0]
        csv_path = os.path.join(self.output_dir, f"{label}.csv")

        run_ncu(script, args=args, output=csv_path, kernel_regex=kernel_regex)
        return load_profiles(csv_path)

    # ── Internals ─────────────────────────────────────────────────────────

    def _gen_script(
        self,
        fn: Callable,
        args: tuple,
        warmup: int,
        iters: int,
    ) -> str:
        """Generate a temporary Python script that calls *fn* under profiler
        brackets so NCU only captures the measured iterations."""
        mod = fn.__module__
        name = fn.__name__
        args_repr = ", ".join(repr(a) for a in args)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        code = (
            f"import sys, os\n"
            f"sys.path.insert(0, {project_root!r})\n"
            f"os.chdir({project_root!r})\n"
            f"import torch\n"
            f"from {mod} import {name}\n"
            f"\n"
            f"for _ in range({warmup}):\n"
            f"    {name}({args_repr})\n"
            f"torch.cuda.synchronize()\n"
            f"\n"
            f"torch.cuda.cudart().cudaProfilerStart()\n"
            f"for _ in range({iters}):\n"
            f"    {name}({args_repr})\n"
            f"    torch.cuda.synchronize()\n"
            f"torch.cuda.cudart().cudaProfilerStop()\n"
        )

        path = os.path.join(self.output_dir, f"_ncu_tmp_{fn.__name__}.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def __repr__(self) -> str:
        return f"NcuRunner(gpu={self.gpu_key!r}, spec={self.spec.name})"


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
