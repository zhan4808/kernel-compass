"""CARM model sanity tests (no GPU required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from profiling.bottleneck import diagnose_shape  # noqa: E402
from profiling.carm import CarmRegime, advise, bw_ws_tbs, params_for, predict_us  # noqa: E402

P = params_for("h100")
MB = 1024 * 1024


def test_capacity_gate():
    assert bw_ws_tbs(16 * MB, P) == P.bw_l2_tbs
    assert bw_ws_tbs(128 * MB, P) == P.bw_hbm_tbs
    # AT the effective capacity already thrashes (LRU)
    assert bw_ws_tbs(P.c_eff_mb * MB, P) == P.bw_hbm_tbs


def test_regimes():
    # MLA 16 MB bs=1: L2-served
    assert diagnose_shape((128, 1, 128, 512)).regime == CarmRegime.L2_SERVED
    # MLA 128 MB bs=1: HBM-bound
    assert diagnose_shape((128, 1, 128, 4096)).regime == CarmRegime.HBM_BOUND
    # Large-batch: compute-bound
    assert diagnose_shape((128, 512, 128, 4096)).regime == CarmRegime.COMPUTE_BOUND
    # Tiny op: launch-bound
    assert advise(flops=1e6, weight_bytes=1e4).regime == CarmRegime.LAUNCH_BOUND


def test_prediction_matches_measured():
    # Graph-timed measurements from cache-barrier (H100):
    # 16 MB / bs=1 fp16 bmm: 4.9 us; 128 MB: 49.1 us
    p16 = predict_us(2 * 128 * 1 * 128 * 512, 16 * MB)
    p128 = predict_us(2 * 128 * 1 * 128 * 4096, 128 * MB)
    assert abs(p16 - 4.9) / 4.9 < 0.25
    assert abs(p128 - 49.1) / 49.1 < 0.25


if __name__ == "__main__":
    test_capacity_gate()
    test_regimes()
    test_prediction_matches_measured()
    print("ALL CARM TESTS PASSED")
