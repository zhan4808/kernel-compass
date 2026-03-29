# kernel-compass

Counter-grounded kernel optimization loop using hardware counters (or CUPTI
fallback) as the feedback signal.

## Current status

- Stage 1 profiling: working (`NcuRunner` + `CuptiRunner` fallback)
- Stage 2 classification: working (4-class taxonomy)
- Stage 3 optimization loop: working (`--demo`, `--compare`)
- Stage 4 LLM generation: scaffolded (`optimizer/llm.py`) with heuristic fallback

Validation currently passes end-to-end:

- `python tests/test_validation.py`
- `python tests/test_validation.py --ncu`

## Core result (demo-ready)

The loop distinguishes `l2_bound` from `memory_bound` and makes opposite
recommendations automatically:

- L2-resident MLA shape (16MB) -> reject quantization
- HBM-bound MLA shape (128MB) -> accept quantization when it helps

This is the key contrast missing from latency-only optimization loops.

## Compare-mode metrics

`python -m optimizer.loop --compare` now reports:

- iterations to first acceptance
- wasted profiling runs before first acceptance

This supports the central claim: diagnosis-conditioned generation can avoid
wasted runs on shapes where counter evidence already rules out the optimization.

## Immediate focus

1. Finalize and submit cache-barrier arXiv paper.
2. Send Dr. Lin outreach/proposal with:
   - cache-barrier result
   - kernel-compass end-to-end demo status
   - planned Stage 4 LLM contribution
3. Continue Stage 4 in parallel after outreach:
   - robust Claude API integration
   - side-by-side evaluation vs grid baseline.