"""Optimizer regime-gating and LLM config-validation tests (no GPU for most)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.llm import SEARCH_SPACE, _heuristic_configs, parse_configs  # noqa: E402
from optimizer.loop import enumerate_configs  # noqa: E402
from profiling.bottleneck import BottleneckClass, Diagnosis  # noqa: E402

_BC = BottleneckClass


def _diag(bc: BottleneckClass) -> Diagnosis:
    return Diagnosis(bottleneck=bc, confidence="high", explanation="test")


def test_enumerate_configs_regime_gating():
    # Memory-bound bf16 baseline must offer W8A8 first
    cfgs = enumerate_configs(_BC.MEMORY_BOUND, "bf16")
    assert cfgs, "memory-bound should enumerate candidates"
    assert cfgs[0].precision == "w8a8"
    # L2-bound: no quantization candidates
    assert enumerate_configs(_BC.L2_BOUND, "bf16") == []


def test_llm_search_space_includes_w8a8():
    assert "w8a8" in SEARCH_SPACE[_BC.MEMORY_BOUND]["precision"]
    assert SEARCH_SPACE[_BC.L2_BOUND] == {}


def test_heuristic_prefers_w8a8_when_memory_bound():
    shape = {"H": 128, "M": 1, "K": 128, "N": 4096}
    cfgs = _heuristic_configs(_diag(_BC.MEMORY_BOUND), shape, "bf16", SEARCH_SPACE[_BC.MEMORY_BOUND])
    assert cfgs and cfgs[0]["precision"] == "w8a8"


def test_heuristic_empty_for_l2_bound():
    shape = {"H": 128, "M": 1, "K": 128, "N": 512}
    assert _heuristic_configs(_diag(_BC.L2_BOUND), shape, "bf16", {}) == []


def test_parse_configs_validates_against_space():
    space = SEARCH_SPACE[_BC.MEMORY_BOUND]
    text = """Here are the configs:
    [
      {"precision": "w8a8", "BLOCK_M": 16, "reasoning": "ok"},
      {"precision": "nvfp4", "BLOCK_M": 16},
      {"precision": "int4", "BLOCK_K": 999},
      "not a dict"
    ]"""
    out = parse_configs(text, space)
    assert len(out) == 2
    assert out[0]["precision"] == "w8a8" and out[0]["BLOCK_M"] == 16
    # invalid BLOCK_K dropped but precision keeps the config alive
    assert out[1]["precision"] == "int4" and "BLOCK_K" not in out[1]


def test_parse_configs_garbage_returns_empty():
    assert parse_configs("no json here", SEARCH_SPACE[_BC.MEMORY_BOUND]) == []
    assert parse_configs("[{broken", SEARCH_SPACE[_BC.MEMORY_BOUND]) == []


def test_validate_recon_mape_thresholds():
    """Model error against the measured recon points stays within accepted bounds."""
    import json

    from profiling.carm import validate_recon_mape

    params_path = os.path.join(os.path.dirname(__file__), "..", "profiling", "carm_params.json")
    with open(params_path) as f:
        recon_points = json.load(f)["recon_points"]
    res = validate_recon_mape(recon_points, gpu="h100")
    assert res["fp16_mape_pct"] < 25.0, res
    assert res["int4_mape_pct"] < 25.0, res


if __name__ == "__main__":
    test_enumerate_configs_regime_gating()
    test_llm_search_space_includes_w8a8()
    test_heuristic_prefers_w8a8_when_memory_bound()
    test_heuristic_empty_for_l2_bound()
    test_parse_configs_validates_against_space()
    test_parse_configs_garbage_returns_empty()
    test_validate_recon_mape_thresholds()
    print("ALL OPTIMIZER TESTS PASSED")
