"""MLA reconstruction profiling and validation cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from kernels.baselines import batched_fp8_gemm, batched_int4_gemm, quantize_fp8, quantize_int4
from profiling.metrics import BMMProfile


@dataclass(frozen=True)
class ValidationExpected:
    dram_bw_pct: tuple[float, float]
    sm_pct: tuple[float, float]
    l2_hit_rate: tuple[float, float]
    label: str


VALIDATION_CASES: Dict[str, ValidationExpected] = {
    "bf16_16mb": ValidationExpected(
        dram_bw_pct=(15.0, 55.0),
        sm_pct=(10.0, 60.0),
        l2_hit_rate=(60.0, 100.0),
        label="cuBLAS BF16 BMM, weight=16 MB (L2-resident)",
    ),
    "int4_16mb": ValidationExpected(
        dram_bw_pct=(5.0, 35.0),
        sm_pct=(40.0, 90.0),
        l2_hit_rate=(30.0, 100.0),
        label="INT4 Triton W4A16 BMM, packed weight=4 MB (SM-saturated)",
    ),
    "fp8_16mb": ValidationExpected(
        dram_bw_pct=(5.0, 45.0),
        sm_pct=(15.0, 80.0),
        l2_hit_rate=(50.0, 100.0),
        label="FP8 Triton W8A16 BMM, weight=8 MB actual (L2-resident)",
    ),
    "bf16_128mb": ValidationExpected(
        dram_bw_pct=(60.0, 100.0),
        sm_pct=(10.0, 60.0),
        l2_hit_rate=(0.0, 50.0),
        label="cuBLAS BF16 BMM, weight=128 MB (HBM-bound)",
    ),
    "fp8_128mb": ValidationExpected(
        dram_bw_pct=(30.0, 100.0),
        sm_pct=(10.0, 70.0),
        l2_hit_rate=(0.0, 55.0),
        label="FP8 Triton W8A16 BMM, weight=64 MB actual (HBM-bound)",
    ),
}

# (H, M, K, N)
_CASE_SHAPES: Dict[str, Tuple[int, int, int, int]] = {
    "bf16_16mb": (128, 1, 128, 512),
    "int4_16mb": (128, 1, 128, 512),
    "fp8_16mb": (128, 1, 128, 512),
    "bf16_128mb": (128, 1, 128, 4096),
    "fp8_128mb": (128, 1, 128, 4096),
}

_CASE_WEIGHT_DTYPE: Dict[str, str] = {
    "bf16_16mb": "bf16",
    "int4_16mb": "int4",
    "fp8_16mb": "fp8",
    "bf16_128mb": "bf16",
    "fp8_128mb": "fp8",
}

_WEIGHT_BPE = {"bf16": 2.0, "int4": 0.5, "fp8": 1.0}
_CASE_CACHE: dict[str, dict] = {}


def _ensure(case: str) -> dict:
    if case in _CASE_CACHE:
        return _CASE_CACHE[case]
    H, M, K, N = _CASE_SHAPES[case]
    dt = torch.bfloat16 if case.startswith("bf16") else torch.float16
    A = torch.randn(H, M, K, dtype=dt, device="cuda")
    W = torch.randn(H, K, N, dtype=dt, device="cuda")
    out: dict = {"A": A, "W": W, "H": H, "M": M, "K": K, "N": N}
    if "int4" in case:
        out["W_packed"], out["scales"] = quantize_int4(W.float())
    elif "fp8" in case:
        out["W_fp8"], out["scales"] = quantize_fp8(W.float())
    _CASE_CACHE[case] = out
    return out


def case_bf16_16mb() -> torch.Tensor:
    t = _ensure("bf16_16mb")
    return torch.bmm(t["A"], t["W"])


def case_int4_16mb() -> torch.Tensor:
    t = _ensure("int4_16mb")
    return batched_int4_gemm(t["A"].half(), t["W_packed"], t["scales"], t["K"])


def case_fp8_16mb() -> torch.Tensor:
    t = _ensure("fp8_16mb")
    return batched_fp8_gemm(t["A"].half(), t["W_fp8"], t["scales"])


def case_bf16_128mb() -> torch.Tensor:
    t = _ensure("bf16_128mb")
    return torch.bmm(t["A"], t["W"])


def case_fp8_128mb() -> torch.Tensor:
    t = _ensure("fp8_128mb")
    return batched_fp8_gemm(t["A"].half(), t["W_fp8"], t["scales"])


VALIDATION_KERNELS: Dict[str, callable] = {
    "bf16_16mb": case_bf16_16mb,
    "int4_16mb": case_int4_16mb,
    "fp8_16mb": case_fp8_16mb,
    "bf16_128mb": case_bf16_128mb,
    "fp8_128mb": case_fp8_128mb,
}


def validate_profiles(case_name: str, sm: float, dram: float, l2: float) -> tuple[bool, list[str]]:
    exp = VALIDATION_CASES[case_name]
    checks = [
        ("SM", sm, exp.sm_pct),
        ("DRAM", dram, exp.dram_bw_pct),
        ("L2", l2, exp.l2_hit_rate),
    ]
    notes: list[str] = []
    ok = True
    for label, val, (lo, hi) in checks:
        if lo <= val <= hi:
            notes.append(f"{label}={val:.1f}% in [{lo:.1f},{hi:.1f}]")
        else:
            notes.append(f"{label}={val:.1f}% NOT in [{lo:.1f},{hi:.1f}]")
            ok = False
    return ok, notes


def run_validation(warmup: int = 20, iters: int = 100) -> list[dict]:
    rows: list[dict] = []
    print(f"{'Case':<12} {'Latency':>10} {'TFLOPS':>10} {'BW(GB/s)':>10} {'Wt MB':>8}")
    for case_name, fn in VALIDATION_KERNELS.items():
        H, M, K, N = _CASE_SHAPES[case_name]
        wdtype = _CASE_WEIGHT_DTYPE.get(case_name, "bf16")
        bpe = _WEIGHT_BPE.get(wdtype, 2.0)
        w_bytes = H * K * N * bpe

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        med_ms = samples[len(samples) // 2]

        flops = 2 * H * M * K * N
        tf = (flops / 1e12) / (med_ms / 1e3) if med_ms > 0 else 0.0
        bw = (w_bytes / 1e9) / (med_ms / 1e3) if med_ms > 0 else 0.0
        rows.append(
            {
                "case": case_name,
                "latency_ms": med_ms,
                "tflops": tf,
                "bandwidth_gbs": bw,
                "weight_mb": w_bytes / 1e6,
            }
        )
        print(f"{case_name:<12} {med_ms:>10.4f} {tf:>10.2f} {bw:>10.1f} {w_bytes / 1e6:>8.1f}")
    return rows


def profile_reconstruction(
    batch_sizes: list[int],
    H: int = 128,
    d_nope: int = 128,
    d_lora: int = 512,
    d_v: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 20,
    iters: int = 100,
) -> list[BMMProfile]:
    profiles: list[BMMProfile] = []
    for bs in batch_sizes:
        x = torch.randn(H, bs, d_nope, dtype=dtype, device="cuda")
        w = torch.randn(H, d_nope, d_lora, dtype=dtype, device="cuda")

        for _ in range(warmup):
            torch.bmm(x, w)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []
        for _ in range(iters):
            start.record()
            torch.bmm(x, w)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        med = times[len(times) // 2]

        p = BMMProfile.from_timing(
            "reconstruction_bmm",
            H,
            bs,
            d_nope,
            d_lora,
            "bf16" if dtype == torch.bfloat16 else "fp16",
            med,
            H * d_nope * d_lora * 2,
        )
        profiles.append(p)
    return profiles
