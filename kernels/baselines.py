"""Baseline BMM wrappers: FP16 cuBLAS and INT4 Triton W4A16.

Ported from cache-barrier/profiling/bench_l2_barrier.py.

All public functions return median latency in milliseconds.

Usage:
    from kernels.baselines import bench_fp16_bmm, bench_int4_bmm, l2_barrier_sweep
"""

from __future__ import annotations

from typing import List

import torch
import triton
import triton.language as tl


# ── INT4 Triton kernel (W4A16 — weights packed, activations FP16) ─────────────

@triton.jit
def _kernel_w4a16_bmm(
    A_ptr, B_ptr, Scale_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_ah, stride_am, stride_ak,
    stride_bh, stride_bk, stride_bn,
    stride_sh, stride_sn,
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched W4A16 GEMM: C[h] = A[h] @ dequant(B[h])."""
    pid_h  = tl.program_id(0)
    pid_mn = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m  = pid_mn // grid_n
    pid_n  = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh
    HALF_BK: tl.constexpr = BLOCK_K // 2

    for k_start in range(0, K, BLOCK_K):
        offs_kp = (k_start // 2) + tl.arange(0, HALF_BK)
        b_ptrs  = b_base + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask  = (offs_kp[:, None] < K // 2) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        b_lo = (b_packed & 0x0F).to(tl.int32)
        b_hi = ((b_packed >> 4) & 0x0F).to(tl.int32)
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo).to(tl.float16)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi).to(tl.float16)

        offs_k_even = k_start + tl.arange(0, HALF_BK) * 2
        offs_k_odd  = offs_k_even + 1
        a_even = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_even[None, :] < K), other=0.0,
        ).to(tl.float16)
        a_odd = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_odd[None, :] < K), other=0.0,
        ).to(tl.float16)
        acc = acc + tl.dot(a_even, b_lo).to(tl.float32)
        acc = acc + tl.dot(a_odd,  b_hi).to(tl.float32)

    scale_ptrs = Scale_ptr + pid_h * stride_sh + offs_n * stride_sn
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scales[None, :]

    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ── Quantization helpers ──────────────────────────────────────────────────────

def quantize_int4(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column absmax INT4 quantization of W: (H, K, N) → (packed, scales).

    packed: (H, K//2, N) uint8 — two int4 values per byte (even=lo nibble, odd=hi)
    scales: (H, N) float16
    """
    w_max = W.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scale = w_max / 7.0
    w_q = (W / scale).round().clamp(-8, 7).to(torch.int8)
    packed = (w_q[..., 0::2, :] & 0x0F).to(torch.uint8) | \
             ((w_q[..., 1::2, :] & 0x0F).to(torch.uint8) << 4)
    return packed, scale.squeeze(-2).to(torch.float16)


def batched_int4_gemm(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """Batched W4A16 GEMM.  A: (H, M, K) fp16, B_packed: (H, K//2, N) uint8."""
    H, M, _ = A.shape
    _, _, N = B_packed.shape
    BLOCK_M = max(16, 1 << (max(16, M) - 1).bit_length())
    BLOCK_N = min(BLOCK_N, N)
    BLOCK_K = max(2, min(BLOCK_K, K))

    C = torch.empty((H, M, N), device=A.device, dtype=torch.float16)
    grid = (H, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    _kernel_w4a16_bmm[grid](
        A, B_packed, scales, C, M, N, K,
        A.stride(0),       A.stride(1),       A.stride(2),
        B_packed.stride(0), B_packed.stride(1), B_packed.stride(2),
        scales.stride(0),  scales.stride(1),
        C.stride(0),       C.stride(1),       C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _time_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_fp16_bmm(H: int, BS: int, K: int, N: int,
                   warmup: int = 50, iters: int = 200) -> float:
    """Median ms for torch.bmm on [H, BS, K] @ [H, K, N] in FP16."""
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K,  N, dtype=torch.float16, device="cuda")
    return _time_fn(lambda: torch.bmm(x, w), warmup, iters)


def bench_int4_bmm(H: int, BS: int, K: int, N: int,
                   warmup: int = 50, iters: int = 200) -> float:
    """Median ms for INT4 Triton W4A16 BMM on [H, BS, K] @ dequant([H, K//2, N])."""
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K,  N, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_int4(w)
    BLOCK_M = max(1, min(16, BS))
    if BLOCK_M > 1:
        BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    return _time_fn(
        lambda: batched_int4_gemm(x, w_packed, scales, K, BLOCK_M=BLOCK_M),
        warmup, iters,
    )


# ── L2 barrier sweep ──────────────────────────────────────────────────────────

def infer_l2_mb(gpu_name: str) -> float:
    name = gpu_name.upper()
    if "H100" in name:
        return 50.0
    if "A100" in name:
        return 40.0
    return 40.0


def l2_barrier_sweep(
    H: int = 128,
    d_nope: int = 128,
    d_lora_sweep: List[int] | None = None,
    batch_sizes: List[int] | None = None,
    warmup: int = 50,
    iters: int = 200,
) -> list[dict]:
    """Sweep d_lora across the L2 cache boundary and record INT4/FP16 ratio.

    Weight bytes per BMM = H * d_nope * d_lora * 2 (FP16).
    H100 L2 = 50 MB; crossover expected around d_lora ≈ 1536–1792.
    """
    if d_lora_sweep is None:
        d_lora_sweep = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]
    if batch_sizes is None:
        batch_sizes = [1, 4]

    gpu_name = torch.cuda.get_device_name(0)
    l2_mb = infer_l2_mb(gpu_name)
    print(f"GPU: {gpu_name}  (L2 ≈ {l2_mb:.0f} MB)")
    print(f"H={H}  d_nope={d_nope}  warmup={warmup}  iters={iters}\n")

    results = []
    for bs in batch_sizes:
        print(f"── BS={bs} ──────────────────────────────────────")
        print(f"{'d_lora':>8} {'Wt MB':>8} {'FP16 ms':>10} {'INT4 ms':>10} {'Ratio':>8} {'L2?':>5}")
        for d_lora in d_lora_sweep:
            wt_mb = H * d_nope * d_lora * 2 / (1024 ** 2)
            fits  = wt_mb < l2_mb
            try:
                fp16_ms = bench_fp16_bmm(H, bs, d_nope, d_lora, warmup, iters)
                int4_ms = bench_int4_bmm(H, bs, d_nope, d_lora, warmup, iters)
            except torch.cuda.OutOfMemoryError:
                print(f"{d_lora:>8} {wt_mb:>8.1f}  OOM")
                continue
            ratio = int4_ms / fp16_ms
            print(f"{d_lora:>8} {wt_mb:>8.1f} {fp16_ms:>10.4f} {int4_ms:>10.4f} {ratio:>7.2f}x {'yes' if fits else 'NO':>5}")
            results.append({
                "batch_size": bs, "d_lora": d_lora,
                "weight_mb": round(wt_mb, 2), "fits_l2": fits,
                "fp16_ms": round(fp16_ms, 4), "int4_ms": round(int4_ms, 4),
                "ratio": round(ratio, 4),
            })
        print()
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="L2 barrier sweep benchmark")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = l2_barrier_sweep()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")
