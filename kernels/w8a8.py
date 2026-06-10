"""
W8A8 batched BMM with INT8 tensor-core MMA for MLA reconstruction.

The constructive answer to the cache-barrier audit: W4A16/W8A16 Triton kernels
lose because per-element dequantization saturates the SMs (in-core ceiling
~30 TF). This kernel removes dequant from the inner loop entirely:

  - weights:     static symmetric per-(head, out-channel) INT8
  - activations: dynamic symmetric per-(head, token) INT8 (one small kernel)
  - inner loop:  tl.dot(int8, int8) -> int32 accumulator  (Hopper IMMA)
  - epilogue:    acc_i32 * a_scale[m] * w_scale[n]  (one multiply per output)

Ported from cache-barrier profiling/w8a8/w8a8_bmm.py. Shapes: A [H, M, K] fp16, W [H, K, N] fp16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_act_kernel(
    a_ptr, q_ptr, s_ptr,
    M, K,
    stride_ah, stride_am, stride_ak,
    stride_qh, stride_qm, stride_qk,
    BLOCK_K: tl.constexpr,
):
    """Per-(head, token) symmetric INT8 quantization of activations."""
    pid = tl.program_id(0)
    h = pid // M
    m = pid % M
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    a = tl.load(a_ptr + h * stride_ah + m * stride_am + offs_k * stride_ak,
                mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(a), axis=0)
    scale = tl.maximum(amax / 127.0, 1e-8)
    q = (a / scale + 0.5 * tl.where(a >= 0, 1.0, -1.0)).to(tl.int8)
    tl.store(q_ptr + h * stride_qh + m * stride_qm + offs_k * stride_qk, q, mask=mask)
    tl.store(s_ptr + pid, scale)


@triton.jit
def _w8a8_bmm_kernel(
    aq_ptr, wq_ptr, c_ptr, as_ptr, ws_ptr,
    M, N, K,
    stride_ah, stride_am, stride_ak,
    stride_wh, stride_wk, stride_wn,
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[h] = (Aq[h] @ Wq[h]) * a_scale[h,m] * w_scale[h,n]; INT8 dot, int32 acc."""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = aq_ptr + pid_h * stride_ah + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = wq_ptr + pid_h * stride_wh + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        acc = tl.dot(a, w, acc=acc)  # int8 x int8 -> int32 (IMMA)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    a_s = tl.load(as_ptr + pid_h * M + offs_m, mask=m_mask, other=0.0)
    w_s = tl.load(ws_ptr + pid_h * N + offs_n, mask=n_mask, other=0.0)
    out = acc.to(tl.float32) * a_s[:, None] * w_s[None, :]

    c_ptrs = c_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


def quantize_weights_w8(w: torch.Tensor):
    """[H, K, N] fp16 -> int8 + per-(H, N) fp32 scale."""
    s = w.float().abs().amax(dim=1) / 127.0          # [H, N]
    s = s.clamp(min=1e-8)
    q = torch.round(w.float() / s[:, None, :]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.contiguous()


def quantize_acts_w8(a: torch.Tensor, q=None, s=None):
    """[H, M, K] fp16 -> int8 + per-(H, M) fp32 scale. Buffers reusable for CUDA graphs."""
    H, M, K = a.shape
    if q is None:
        q = torch.empty_like(a, dtype=torch.int8)
    if s is None:
        s = torch.empty(H * M, dtype=torch.float32, device=a.device)
    BLOCK_K = triton.next_power_of_2(K)
    _quant_act_kernel[(H * M,)](
        a, q, s, M, K,
        a.stride(0), a.stride(1), a.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        BLOCK_K=BLOCK_K,
    )
    return q, s


def _pick_config(M):
    # Autotuned on H100 at the MLA shape (K=128, N=512).
    if M <= 16:
        return dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4)
    if M <= 64:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)


def w8a8_bmm(aq, wq, a_scale, w_scale, out=None,
             BLOCK_M=None, BLOCK_N=None, BLOCK_K=None, num_warps=None, num_stages=None):
    H, M, K = aq.shape
    N = wq.shape[2]
    cfg = _pick_config(M)
    BLOCK_M = BLOCK_M or cfg["BLOCK_M"]
    BLOCK_N = BLOCK_N or cfg["BLOCK_N"]
    BLOCK_K = BLOCK_K or cfg["BLOCK_K"]
    num_warps = num_warps or cfg["num_warps"]
    num_stages = num_stages or cfg["num_stages"]
    if out is None:
        out = torch.empty(H, M, N, dtype=torch.float16, device=aq.device)
    grid = (H, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _w8a8_bmm_kernel[grid](
        aq, wq, out, a_scale, w_scale,
        M, N, K,
        aq.stride(0), aq.stride(1), aq.stride(2),
        wq.stride(0), wq.stride(1), wq.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def w8a8_bmm_full(a, wq, w_scale, bufs=None):
    """Dynamic act quant + INT8 BMM. `bufs=(q, s, out)` for graph capture."""
    q = s = out = None
    if bufs is not None:
        q, s, out = bufs
    q, s = quantize_acts_w8(a, q, s)
    return w8a8_bmm(q, wq, s, w_scale, out)


if __name__ == "__main__":
    torch.manual_seed(0)
    H, M, K, N = 128, 4, 128, 512
    a = torch.randn(H, M, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
    ref = torch.bmm(a, w).float()
    wq, ws = quantize_weights_w8(w)
    out = w8a8_bmm_full(a, wq, ws).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"rel_err = {rel:.5f}  ({'ok' if rel < 0.02 else 'FAIL'})")
