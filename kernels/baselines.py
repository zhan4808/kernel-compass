"""Baseline BMM wrappers: BF16 cuBLAS, INT4 Triton W4A16, FP8 weight-only W8A16 (Triton).

FP8 path: FP8 weights dequantized to FP16 for tl.dot — not native W8A8 tensor-core GEMM.
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_w4a16_bmm(
    A_ptr,
    Bp_ptr,
    Scale_ptr,
    C_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk2,
    stride_bn,
    stride_sh,
    stride_sn,
    stride_ch,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_mn = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = A_ptr + pid_h * stride_ah
    b_base = Bp_ptr + pid_h * stride_bh

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        kk2 = offs_k // 2
        packed = tl.load(
            b_base + kk2[:, None] * stride_bk2 + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        even = (offs_k % 2) == 0
        lo = (packed & 0x0F).to(tl.int8)
        hi = ((packed >> 4) & 0x0F).to(tl.int8)
        q = tl.where(even[:, None], lo, hi) - 8
        b = q.to(tl.float16)
        acc += tl.dot(a, b).to(tl.float32)

    scales = tl.load(
        Scale_ptr + pid_h * stride_sh + offs_n * stride_sn,
        mask=offs_n < N,
        other=1.0,
    ).to(tl.float32)
    acc = acc * scales[None, :]

    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def quantize_int4(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 7.0
    w_max = W.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scales = (w_max / qmax).to(torch.float16)
    q = torch.round((W / scales).clamp(-8, 7)).to(torch.int8) + 8
    q_u8 = q.to(torch.uint8)
    lo = q_u8[:, 0::2, :] & 0x0F
    hi = (q_u8[:, 1::2, :] & 0x0F) << 4
    packed = lo | hi
    return packed.contiguous(), scales.squeeze(-2).contiguous()


def batched_int4_gemm(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    H, M, _ = A.shape
    _, _, N = B_packed.shape
    C = torch.empty((H, M, N), device=A.device, dtype=torch.float16)
    grid = (H, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    _kernel_w4a16_bmm[grid](
        A,
        B_packed,
        scales,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B_packed.stride(0),
        B_packed.stride(1),
        B_packed.stride(2),
        scales.stride(0),
        scales.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


@triton.jit
def _kernel_w8a16_bmm(
    A_ptr,
    B_ptr,
    Scale_ptr,
    C_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_sh,
    stride_sn,
    stride_ch,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_mn = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)
        b = tl.load(
            b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a, b).to(tl.float32)
    scales = tl.load(Scale_ptr + pid_h * stride_sh + offs_n * stride_sn, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scales[None, :]
    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def quantize_fp8(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    w_max = W.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scale = w_max / fp8_max
    w_scaled = W / scale
    w_fp8 = w_scaled.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return w_fp8.contiguous(), scale.squeeze(-2).to(torch.float16).contiguous()


def batched_fp8_gemm(
    A: torch.Tensor,
    B_fp8: torch.Tensor,
    scales: torch.Tensor,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    H, M, K = A.shape
    _, _, N = B_fp8.shape
    C = torch.empty((H, M, N), device=A.device, dtype=torch.float16)
    grid = (H, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    _kernel_w8a16_bmm[grid](
        A,
        B_fp8,
        scales,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B_fp8.stride(0),
        B_fp8.stride(1),
        B_fp8.stride(2),
        scales.stride(0),
        scales.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def _time_fn(fn: Callable[[], torch.Tensor], warmup: int = 50, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return float(times[len(times) // 2])


def bench_bf16_bmm(H: int, BS: int, K: int, N: int, warmup: int = 50, iters: int = 200) -> float:
    x = torch.randn(H, BS, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.bfloat16, device="cuda")
    return _time_fn(lambda: torch.bmm(x, w), warmup, iters)


def bench_int4_bmm(H: int, BS: int, K: int, N: int, warmup: int = 50, iters: int = 200) -> float:
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_int4(w)
    return _time_fn(lambda: batched_int4_gemm(x, w_packed, scales, K), warmup, iters)


def bench_fp8_bmm(H: int, BS: int, K: int, N: int, warmup: int = 50, iters: int = 200) -> float:
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    w_fp8, scales = quantize_fp8(w)
    return _time_fn(lambda: batched_fp8_gemm(x, w_fp8, scales), warmup, iters)


def l2_barrier_sweep(
    H: int = 128,
    bs: int = 1,
    d_nope: int = 128,
    d_lora_sweep: list[int] | None = None,
    warmup: int = 30,
    iters: int = 100,
) -> list[dict]:
    d_lora_sweep = d_lora_sweep or [256, 512, 1024, 1536, 2048, 3072, 4096]
    out: list[dict] = []
    print(f"{'d_lora':>8} {'Wt MB':>8} {'BF16 ms':>10} {'INT4 ms':>10} {'Ratio':>8} {'L2?':>5}")
    for d_lora in d_lora_sweep:
        wt_mb = H * d_nope * d_lora * 2 / 1e6
        fits = wt_mb < 50
        bf16_ms = bench_bf16_bmm(H, bs, d_nope, d_lora, warmup, iters)
        int4_ms = bench_int4_bmm(H, bs, d_nope, d_lora, warmup, iters)
        ratio = int4_ms / max(bf16_ms, 1e-9)
        print(f"{d_lora:>8} {wt_mb:>8.1f} {bf16_ms:>10.4f} {int4_ms:>10.4f} {ratio:>7.2f}x {'yes' if fits else 'NO':>5}")
        out.append(
            {
                "d_lora": d_lora,
                "weight_mb": round(wt_mb, 2),
                "fits_l2": fits,
                "bf16_ms": round(bf16_ms, 4),
                "int4_ms": round(int4_ms, 4),
                "ratio": round(ratio, 4),
            }
        )
    return out
