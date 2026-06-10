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

### Corrections from the 2026-06 methodology audit (see cache-barrier `profiling/validation/`)

- **NCU must run with `--cache-control none` for residency diagnosis.** The
  default (`all`) flushes L2 before every replayed launch; under it,
  `l2_hit_rate` is always near-zero and the `L2_BOUND` class can never fire.
  `ncu_runner.run_ncu` / `NcuRunner.run` now default to `none`.
- **Effective L2 capacity on H100 is ~32-40 MB, not the nominal 50 MB**
  (LRU + partitioning). Residency-based decisions should use the measured
  effective capacity, and effective L2-era serving bandwidth is ~4-6 TB/s
  for GEMM access patterns, not the ~12 TB/s aggregate figure.
- **"L2-resident -> reject weight-only quantization" remains the right
  recommendation**, but for a stronger reason: even with L2 residency
  removed (weight-rotation intervention), the dequant-bound INT4 kernel
  only reaches parity with FP16 — it never wins. Reject INT4 in this regime
  on kernel-efficiency grounds as well as residency grounds.
- **Per-launch CUDA-event timing has a ~15 us floor** on this class of
  µs-scale kernels; validation timing for accept/revert decisions should use
  CUDA-graph timing (or compare medians of graph-captured loops).

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

## Precision implementation (accurate claims)

- **Baseline for experiments:** BF16 (default in `optimizer.loop` and demos).
- **Implemented alternatives:**
  - FP8 **weight-only W8A16** (Triton: FP8 weights, FP16 activations/compute). Reduces weight bytes vs BF16 but does **not** use H100 native FP8 tensor cores.
  - INT4 W4A16 (Triton, dequant overhead).
- **Not yet implemented:** native FP8 **W8A8** (cuBLASLt / Transformer Engine) — add and verify via NCU before claiming “native FP8 inference” benchmarks. NVFP4 kernel path: taxonomy may exist; implementation as needed.

Paper framing: benchmark **precision strategies for MLA reconstruction weights** (BF16 baseline, FP8 weight-only W8A16, INT4 weight-only) and show the counter-grounded optimizer picks the right tool per regime — without claiming full hardware-native FP8 inference until W8A8 is wired.