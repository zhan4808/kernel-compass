# kernel-compass

Counter-grounded kernel optimization for GPU GEMMs.

## What it does

- Profiles kernels with Nsight Compute (`profiling/ncu_runner.py`)
- Falls back to CUPTI-based estimates when hardware counters are blocked
- Classifies bottlenecks into:
  - `memory_bound`
  - `l2_bound`
  - `compute_bound`
  - `occupancy_limited`
- Runs optimization loops:
  - Grid search (`optimizer/loop.py`)
  - LLM-guided proposal mode (`optimizer/llm.py`)

## Install

```bash
pip install -r requirements.txt
```

## Validation

Level 1 (timing only):

```bash
python tests/test_validation.py
```

Level 2 + 3 (counter validation + classifier validation):

```bash
python tests/test_validation.py --ncu
```

Validation cases:

- `bf16_16mb` (L2-resident baseline)
- `int4_16mb` (SM-heavy dequant path)
- `fp8_16mb` (L2-resident FP8)
- `bf16_128mb` (HBM-bound baseline)
- `fp8_128mb` (HBM-bound FP8)

## Optimization loop

Run killer demo (same loop, opposite recommendation):

```bash
python -m optimizer.loop --demo
```

Compare grid vs LLM-guided mode:

```bash
python -m optimizer.loop --compare
```

Single-shape runs:

```bash
python -m optimizer.loop --grid 128 1 128 4096 --precision bf16
python -m optimizer.loop --llm 128 1 128 4096 --precision bf16 --heuristic
```

## Notes

- If `ncu` counters are blocked (`ERR_NVGPUCTRPERM`), the runner uses `CuptiRunner`.
- CUPTI mode is useful for development, but final decisions should be made with real NCU counters.
