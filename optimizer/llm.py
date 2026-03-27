"""Stage 4 LLM-guided config generation."""

from __future__ import annotations

import json
import os
from typing import Optional

from profiling.bottleneck import BottleneckClass, Diagnosis
from profiling.metrics import KernelProfile

_BC = BottleneckClass

SEARCH_SPACE: dict[BottleneckClass, dict] = {
    _BC.MEMORY_BOUND: {
        "precision": ["fp8", "int4"],
        "BLOCK_M": [16, 32, 64],
        "BLOCK_N": [32, 64, 128],
        "BLOCK_K": [64, 128, 256],
    },
    _BC.L2_BOUND: {},
    _BC.COMPUTE_BOUND: {
        "BLOCK_M": [16, 32, 64],
        "BLOCK_N": [32, 64, 128],
        "BLOCK_K": [64, 128, 256],
    },
    _BC.OCCUPANCY_LIMITED: {
        "BLOCK_M": [16, 32],
        "BLOCK_N": [32, 64],
        "BLOCK_K": [64, 128],
    },
}

_BPE = {"bf16": 2, "fp16": 2, "fp8": 1, "int4": 0.5}


def build_prompt(
    diagnosis: Diagnosis,
    profile: KernelProfile,
    shape: dict,
    current_precision: str,
    search_space: dict,
    n_candidates: int = 3,
) -> str:
    bpe = _BPE.get(current_precision, 2)
    w_bytes = shape["H"] * shape["K"] * shape["N"] * bpe
    space_json = json.dumps(search_space, indent=2)
    return f"""You are optimizing a GPU GEMM kernel on an NVIDIA H100.

Current bottleneck: {diagnosis.bottleneck.value}
Confidence: {diagnosis.confidence}
Diagnosis: {diagnosis.explanation}

Current precision: {current_precision}
Shape: H={shape["H"]}, M={shape["M"]}, K={shape["K"]}, N={shape["N"]}
Weight size: {w_bytes / 1e6:.1f} MB

Counters:
- SM throughput: {profile.sm_pct:.1f}%
- DRAM throughput: {profile.dram_bw_pct:.1f}%
- L2 hit rate: {profile.l2_hit_rate:.1f}%
- Latency: {profile.avg_duration_us:.1f} us

Allowed search space:
{space_json}

Rules:
- For MEMORY_BOUND: prefer FP8 before INT4.
- For L2_BOUND with empty space, return [].
- Do not output configs outside the search space.

Generate exactly {n_candidates} candidates.
Output ONLY a JSON array of objects with optional keys:
precision, BLOCK_M, BLOCK_N, BLOCK_K, reasoning
"""


def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=2048, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def _validate_one(raw: dict, space: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    out: dict = {}
    if "precision" in raw:
        if raw["precision"] in space.get("precision", []):
            out["precision"] = raw["precision"]
        else:
            return None
    for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
        if key in raw:
            if isinstance(raw[key], (int, float)) and int(raw[key]) in space.get(key, []):
                out[key] = int(raw[key])
    if "reasoning" in raw:
        out["reasoning"] = str(raw["reasoning"])
    has_content = "precision" in out or any(k.startswith("BLOCK_") for k in out)
    return out if has_content else None


def parse_configs(text: str, space: dict) -> list[dict]:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [c for raw in arr if (c := _validate_one(raw, space)) is not None]


def _heuristic_configs(diagnosis: Diagnosis, shape: dict, current_precision: str, space: dict) -> list[dict]:
    if not space:
        return []
    from optimizer.loop import PRECISION_TIERS

    H, K, N = shape["H"], shape["K"], shape["N"]
    out: list[dict] = []
    if diagnosis.bottleneck == _BC.MEMORY_BOUND:
        for p in space.get("precision", []):
            tier = PRECISION_TIERS.get(p, {})
            w_mb = H * K * N * tier.get("bytes_per_param", 1) / 1e6
            out.append(
                {
                    "precision": p,
                    "BLOCK_M": space.get("BLOCK_M", [16])[0],
                    "BLOCK_N": space.get("BLOCK_N", [64])[-1],
                    "BLOCK_K": space.get("BLOCK_K", [128])[-1],
                    "reasoning": f"{p.upper()} shrinks weight to {w_mb:.0f} MB",
                }
            )
    else:
        for bm in space.get("BLOCK_M", [16]):
            out.append(
                {
                    "precision": current_precision,
                    "BLOCK_M": bm,
                    "BLOCK_N": space.get("BLOCK_N", [64])[0],
                    "BLOCK_K": space.get("BLOCK_K", [128])[0],
                    "reasoning": "smaller tile to reduce pressure",
                }
            )
            if len(out) >= 3:
                break
    return out[:3]


def generate_configs(
    diagnosis: Diagnosis,
    profile: KernelProfile,
    shape: dict,
    current_precision: str = "bf16",
    n_candidates: int = 3,
    model: str = "claude-sonnet-4-20250514",
    force_heuristic: bool = False,
) -> list[dict]:
    space = SEARCH_SPACE.get(diagnosis.bottleneck, {})
    if not space:
        return []
    use_llm = not force_heuristic and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm:
        try:
            prompt = build_prompt(diagnosis, profile, shape, current_precision, space, n_candidates)
            resp = call_llm(prompt, model=model)
            parsed = parse_configs(resp, space)
            if parsed:
                return parsed
        except Exception as exc:
            print(f"    LLM call failed ({exc}); falling back to heuristic")
    return _heuristic_configs(diagnosis, shape, current_precision, space)
