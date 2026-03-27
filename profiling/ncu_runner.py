"""Nsight Compute runner + CUPTI fallback."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from profiling.metrics import GPU_SPECS, KernelProfile

_ERR_NVGPUCTRPERM = "ERR_NVGPUCTRPERM"
_CUPTI_WARNING = (
    "WARNING: Counter data is ESTIMATED from workload geometry, not measured by "
    "hardware counters. Do not use this mode for final optimization decisions."
)

_METRICS = {
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1_hit_rate": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    "l2_hit_rate": "lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct",
}


def _resolve_gpu(gpu: str) -> str:
    if gpu != "auto":
        return gpu.lower()
    try:
        import torch

        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return "h100"
    if "a100" in name:
        return "a100"
    return "h100"


def _ncu_binary() -> str:
    ncu = shutil.which("ncu")
    if not ncu:
        raise FileNotFoundError("ncu not found on PATH")
    return ncu


def ncu_has_permissions() -> bool:
    """Probe if NCU can read GPU counters in this environment."""
    try:
        ncu = _ncu_binary()
    except FileNotFoundError:
        return False

    q = subprocess.run(
        [ncu, "--query-metrics-mode", "set"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    out = (q.stdout or "") + (q.stderr or "")
    if _ERR_NVGPUCTRPERM in out:
        return False

    probe = subprocess.run(
        [
            ncu,
            "--target-processes",
            "all",
            "--metrics",
            _METRICS["sm_pct"],
            sys.executable,
            "-c",
            "import torch; torch.zeros(1,device='cuda'); torch.cuda.synchronize()",
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    out = (probe.stdout or "") + (probe.stderr or "")
    return _ERR_NVGPUCTRPERM not in out


def _to_float(value: str) -> float:
    text = (value or "").replace("%", "").replace(",", "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_profiles(csv_path: str) -> list[KernelProfile]:
    if not os.path.exists(csv_path):
        return []
    rows: dict[str, KernelProfile] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row.get("Kernel Name", "").strip()
            metric = row.get("Metric Name", "").strip()
            val = _to_float(row.get("Metric Value", "0"))
            if not k:
                continue
            prof = rows.setdefault(k, KernelProfile(kernel_name=k))
            if metric == _METRICS["sm_pct"]:
                prof.sm_pct = val
            elif metric == _METRICS["mem_pct"]:
                prof.mem_pct = val
            elif metric == _METRICS["occupancy"]:
                prof.occupancy = val
            elif metric == _METRICS["l1_hit_rate"]:
                prof.l1_hit_rate = val
            elif metric == _METRICS["l2_hit_rate"]:
                prof.l2_hit_rate = val
            elif metric == "gpu__time_duration.sum":
                prof.duration_us = val / 1000.0
                prof.invocation_count = 1
    return list(rows.values())


class NcuRunner:
    """Run callable under Nsight Compute."""

    def __init__(self, gpu: str = "auto", output_dir: str = "data/validation"):
        self.gpu_key = _resolve_gpu(gpu)
        self.spec = GPU_SPECS.get(self.gpu_key, GPU_SPECS["h100"])
        self.output_dir = output_dir

    def __repr__(self) -> str:
        return f"NcuRunner(gpu={self.gpu_key!r}, spec={self.spec.name})"

    def run(
        self,
        kernel_fn: Callable,
        *args,
        warmup: int = 10,
        iters: int = 5,
        label: Optional[str] = None,
    ) -> list[KernelProfile]:
        os.makedirs(self.output_dir, exist_ok=True)
        ncu = _ncu_binary()
        label = label or getattr(kernel_fn, "__name__", "kernel")
        out_csv = os.path.join(self.output_dir, f"{label}.csv")

        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "run_kernel.py"
            script.write_text(
                (
                    "import cloudpickle\n"
                    "import torch\n"
                    f"fn = cloudpickle.load(open('{td}/fn.pkl','rb'))\n"
                    f"args = cloudpickle.load(open('{td}/args.pkl','rb'))\n"
                    f"warmup = {warmup}\n"
                    f"iters = {iters}\n"
                    "for _ in range(warmup):\n"
                    "    fn(*args)\n"
                    "torch.cuda.synchronize()\n"
                    "for _ in range(iters):\n"
                    "    fn(*args)\n"
                    "torch.cuda.synchronize()\n"
                ),
                encoding="utf-8",
            )
            import cloudpickle

            with open(f"{td}/fn.pkl", "wb") as f:
                cloudpickle.dump(kernel_fn, f)
            with open(f"{td}/args.pkl", "wb") as f:
                cloudpickle.dump(args, f)

            cmd = [
                ncu,
                "--target-processes",
                "all",
                "--csv",
                "--page",
                "raw",
                "--log-file",
                out_csv,
                "--metrics",
                ",".join(list(_METRICS.values()) + ["gpu__time_duration.sum"]),
                sys.executable,
                str(script),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            full = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0:
                raise RuntimeError(full.strip() or "NCU run failed")
            if _ERR_NVGPUCTRPERM in full:
                raise RuntimeError(_ERR_NVGPUCTRPERM)

        return load_profiles(out_csv)


class CuptiRunner:
    """Fallback profiler using torch.profiler device timings + estimation."""

    _WEIGHT_BPE = {"bf16": 2.0, "fp16": 2.0, "int4": 0.5, "fp8": 1.0}
    _HAS_SCALES = {"bf16": False, "fp16": False, "int4": True, "fp8": True}

    def __init__(self, gpu: str = "auto"):
        self.gpu_key = _resolve_gpu(gpu)
        self.spec = GPU_SPECS.get(self.gpu_key, GPU_SPECS["h100"])

    def __repr__(self) -> str:
        return f"CuptiRunner(gpu={self.gpu_key!r}, spec={self.spec.name})"

    def run(
        self,
        kernel_fn: Callable,
        *args,
        warmup: int = 10,
        iters: int = 5,
        label: Optional[str] = None,
        case_shapes: Optional[tuple] = None,
        weight_dtype: str = "bf16",
        is_int4: bool = False,
    ) -> list[KernelProfile]:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if is_int4 and weight_dtype == "bf16":
            weight_dtype = "int4"

        label = label or getattr(kernel_fn, "__name__", "kernel")
        for _ in range(warmup):
            kernel_fn(*args)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                kernel_fn(*args)
            torch.cuda.synchronize()

        cuda_events = [
            e
            for e in prof.key_averages()
            if e.self_device_time_total > 0
            and "memcpy" not in (e.key or "").lower()
            and "memset" not in (e.key or "").lower()
        ]
        if not cuda_events:
            return []

        best = max(cuda_events, key=lambda e: e.self_device_time_total)
        avg_dur_us = best.self_device_time_total / max(best.count, 1)
        if case_shapes is None:
            return [
                KernelProfile(
                    kernel_name=best.key or label,
                    duration_us=best.self_device_time_total,
                    invocation_count=best.count,
                    source="cupti_estimated",
                )
            ]
        return [self._estimate(best, avg_dur_us, case_shapes, weight_dtype, label)]

    def _estimate(
        self,
        event,
        avg_dur_us: float,
        shapes: tuple,
        weight_dtype: str,
        label: str,
    ) -> KernelProfile:
        H, M, K, N = shapes
        dur_s = max(avg_dur_us * 1e-6, 1e-9)
        peak_bw = self.spec.hbm_bw_gbs * 1e9
        l2_bytes = self.spec.l2_cache_mb * 1024 * 1024

        bpe = self._WEIGHT_BPE.get(weight_dtype, 2.0)
        weight_bytes = float(H * K * N) * bpe
        act_bytes = H * M * K * 2.0
        out_bytes = H * M * N * 2.0
        scales_bytes = H * N * 2.0 if self._HAS_SCALES.get(weight_dtype, False) else 0.0
        total_bytes = weight_bytes + act_bytes + out_bytes + scales_bytes

        if weight_bytes < l2_bytes * 0.9:
            ratio = weight_bytes / l2_bytes
            mem_pct = 20.0 + 25.0 * ratio
            if weight_dtype == "int4":
                mem_pct *= 0.6
            elif weight_dtype == "fp8":
                mem_pct *= 0.8
        else:
            mem_pct = min(100.0, total_bytes / dur_s / peak_bw * 100.0)

        if weight_bytes < l2_bytes * 0.9:
            l2_hit = min(95.0, 65.0 + 30.0 * (1.0 - weight_bytes / l2_bytes))
        else:
            l2_hit = max(5.0, 45.0 * l2_bytes / max(weight_bytes, 1.0))

        if weight_dtype == "int4":
            sm_pct = min(85.0, max(45.0, 70.0 * (1.0 - 0.3 * mem_pct / 100.0)))
        elif weight_dtype == "fp8":
            sm_pct = max(20.0, min(65.0, 0.5 * mem_pct + 15.0))
        else:
            sm_pct = max(15.0, min(55.0, 0.4 * mem_pct + 12.0))

        return KernelProfile(
            kernel_name=event.key or label,
            sm_pct=sm_pct,
            mem_pct=mem_pct,
            l2_hit_rate=l2_hit,
            duration_us=event.self_device_time_total,
            invocation_count=event.count,
            source="cupti_estimated",
        )
