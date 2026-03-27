# kernel-compass

Counter-grounded kernel optimization loop. Uses Nsight Compute hardware
counters as the feedback signal — diagnosing what's wrong before generating
a fix, and verifying the fix improved the right metric, not just latency.

## Motivation

The MLA L2 cache barrier finding (see cache-barrier repo) is the proof-of-concept:
a latency-only optimization loop would accept INT4 regression at bs=1 because
INT4 runs in finite time. A counter-grounded loop would catch it because
DRAM utilization did not improve and SM saturation increased.

Existing LLM kernel generation systems (KernelGen, KernelEvolve) use throughput
as the sole reward signal — blind to why a kernel is slow. This project builds
the counter-grounded feedback loop those systems lack.

## Architecture

Stage 1 — Profile:     NcuRunner → KernelProfile (counters + latency)
Stage 2 — Diagnose:    heuristic classifier → BottleneckClass
Stage 3 — Generate:    LLM generates configs conditioned on bottleneck class
Stage 4 — Verify:      accept only if target counter metric improves
Stage 5 — Iterate:     loop until convergence or budget

## The killer demo

Target: MLA reconstruction GEMM at M=1, weight=16MB (L2-resident)
Expected: latency-only loop accepts INT4 (0.49x cuBLAS)
Goal: counter-grounded loop diagnoses L2 residency, rejects INT4,
      reports "DRAM utilization unchanged, SM saturation increased"

Ground truth data is in the cache-barrier repo (Section 5.5 of paper).

## Status

Stage 1 (NcuRunner): IN PROGRESS
Stage 2 (Classifier): TODO
Stage 3-5 (Optimizer loop): TODO

## Next immediate task

Implement NcuRunner in profiling/ncu_runner.py.

Target interface:
    runner = NcuRunner(gpu="h100")
    profile = runner.run(kernel_fn, *args, warmup=10, iters=5)
    # returns KernelProfile with:
    #   dram_bw_pct, sm_utilization, l2_hit_rate,
    #   occupancy_pct, registers_per_thread, duration_ms, cv_pct

Validate on three known cases from cache-barrier paper:
  1. cuBLAS FP16 bmm, weight=16MB  → low DRAM (~35%), L2-resident
  2. INT4 Triton kernel, weight=16MB → high SM (~70%), low DRAM (~20%)
  3. cuBLAS FP16 bmm, weight=128MB  → high DRAM (~83%), HBM-bound

All three expected outputs documented in cache-barrier/paper/sglang_mla.pdf
Section 5.5 and Table 11.

## Key reference

cache-barrier repo: github.com/zhan4808/cache-barrier
Paper: arxiv link (pending)
```

**.gitignore:**
```
*.ncu-rep
*.nsys-rep
*.sqlite
__pycache__/
*.pyc
*.csv
data/*.json