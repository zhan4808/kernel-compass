# Direction

## Core thesis

Latency-only optimization can misdiagnose kernels. Counter-grounded diagnosis is required to distinguish:

- true HBM pressure (`memory_bound`)
- L2-resident kernels (`l2_bound`)

Those two require opposite optimization actions.

## Confirmed behavior to preserve

The project tracks a bottleneck flip result:

- Baseline large-weight BF16 case starts as `memory_bound`
- Aggressive quantized variant can move into `l2_bound` or `compute_bound`

This means an optimization can improve latency while also changing the operating regime. The loop must re-profile and re-classify after each accepted change.

## Optimization policy

- `l2_bound`: reject precision-reduction proposals by default
- `memory_bound`: prefer FP8 first; consider INT4 only when memory footprint is the main constraint
- `compute_bound`: reduce FLOPs / improve tensor-core utilization
- `occupancy_limited`: tune tile sizes for occupancy

## Stage roadmap

1. Stage 1: counter collection (`NcuRunner`, `CuptiRunner`)
2. Stage 2: bottleneck classification (`profiling/bottleneck.py`)
3. Stage 3: counter-grounded grid search (`optimize_grid`)
4. Stage 4: LLM config generation + strict validation (`optimize_llm`)
5. Evaluation: grid vs LLM on matched profiling budget
