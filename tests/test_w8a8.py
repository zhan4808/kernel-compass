"""W8A8 INT8-MMA kernel tests: numerical accuracy, graph capture, perf sanity."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from kernels.baselines import make_w8a8_bmm_fn  # noqa: E402
from kernels.w8a8 import quantize_weights_w8, w8a8_bmm_full  # noqa: E402


@pytest.mark.parametrize("shape", [(128, 1, 128, 512), (128, 4, 128, 512), (8, 64, 256, 1024)])
def test_numerical_accuracy(shape):
    torch.manual_seed(0)
    H, M, K, N = shape
    a = torch.randn(H, M, K, dtype=torch.float16, device="cuda") / 4
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda") / 8
    ref = torch.bmm(a, w).float()
    wq, ws = quantize_weights_w8(w)
    out = w8a8_bmm_full(a, wq, ws).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 0.02, f"rel_err {rel:.4f} exceeds 0.02 for shape {shape}"


def test_graph_capturable():
    """The act-quant + BMM pipeline must capture into a CUDA graph (used by verify())."""
    fn, _bufs = make_w8a8_bmm_fn(128, 1, 128, 512)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()


def test_w8a8_wins_when_hbm_bound():
    """Graph-timed: W8A8 must beat cuBLAS FP16 on the 128 MB HBM-bound shape."""
    from profiling.carm import graph_time_us

    H, M, K, N = 128, 1, 128, 4096
    a = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    fp16_us = graph_time_us(lambda: torch.bmm(a, w))
    fn, _bufs = make_w8a8_bmm_fn(H, M, K, N)
    w8a8_us = graph_time_us(fn)
    assert fp16_us is not None and w8a8_us is not None
    assert w8a8_us < fp16_us, f"W8A8 {w8a8_us:.1f}us not faster than FP16 {fp16_us:.1f}us at 128 MB"


def test_w8a8_loses_when_l2_served():
    """Graph-timed: at the 16 MB L2-resident shape FP16 must win (regime check)."""
    from profiling.carm import graph_time_us

    H, M, K, N = 128, 1, 128, 512
    a = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    fp16_us = graph_time_us(lambda: torch.bmm(a, w))
    fn, _bufs = make_w8a8_bmm_fn(H, M, K, N)
    w8a8_us = graph_time_us(fn)
    assert fp16_us is not None and w8a8_us is not None
    assert fp16_us < w8a8_us, f"FP16 {fp16_us:.1f}us not faster than W8A8 {w8a8_us:.1f}us at 16 MB"


if __name__ == "__main__":
    test_numerical_accuracy((128, 1, 128, 512))
    test_graph_capturable()
    test_w8a8_wins_when_hbm_bound()
    test_w8a8_loses_when_l2_served()
    print("ALL W8A8 TESTS PASSED")
