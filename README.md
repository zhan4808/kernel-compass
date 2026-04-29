# kernel-compass

Automated GPU kernel optimization loop for transformer inference, seeded from
the [cache-barrier](https://github.com/zhan4808/cache-barrier) paper on MLA reconstruction GEMMs.

Clone with the pinned companion checkout:

```bash
git clone --recursive git@github.com:zhan4808/kernel-compass.git
# or after a plain clone:  git submodule update --init --recursive
```

## Pipeline

```
target script
     │
     ▼ Stage 1
ncu_runner.py  ──→  NCU CSV  ──→  List[KernelProfile]
                                         │
                                         ▼ Stage 2
                                   bottleneck.py  ──→  KernelProfile.bottleneck
                                         │
                                         ▼ Stage 3
                                   select_candidate()
                                         │
                                         ▼ Stage 4
                                   propose()  ──→  optimization description
                                         │
                                         ▼ Stage 5
                                   validate()  ──→  accept / revert
```

## Structure

```
profiling/
  metrics.py       KernelProfile and BMMProfile dataclasses
  ncu_runner.py    Stage 1 — run ncu, parse CSV, emit KernelProfiles
  bottleneck.py    Stage 2 — classify + markdown report

kernels/
  mla_reconstruction.py  DeepSeek-V2/V3 MLA reconstruction BMM profiler
  baselines.py           FP16 cuBLAS and INT4 Triton W4A16 BMM wrappers

optimizer/
  loop.py          Stages 3–5 — candidate selection, proposal, validation

data/              NCU CSVs and benchmark results (gitignored)
paper/             LaTeX draft + GPU data checklist (see paper/README.md)
cache-barrier/     Git submodule — reference experiments/paper artifact (optional checkout)
DIRECTION.md       Roadmap and design notes
```

## Requirements

- NVIDIA GPU (H100 or A100 recommended for MLA experiments)
- PyTorch ≥ 2.1 with CUDA
- Triton ≥ 3.0
- Nsight Compute (`ncu`) for Stage 1

```bash
pip install torch triton
```

## Quick start

**Profile MLA reconstruction BMMs:**
```bash
python -m kernels.mla_reconstruction --model deepseek-v3
```

**L2 barrier sweep (INT4 vs FP16 across L2 boundary):**
```bash
python -m kernels.baselines --output data/l2_sweep.json
```

**Run NCU and classify kernels:**
```bash
python -m profiling.ncu_runner \
    --script kernels/mla_reconstruction.py \
    --args "--model deepseek-v3 --ncu-mode" \
    --output data/mla_v3.csv \
    --label "mla_v3_decode_bs1"
```

**Parse an existing NCU CSV:**
```bash
python -m profiling.ncu_runner --parse-only data/mla_v3.csv --label "mla_v3"
```

**Full optimization loop (interactive):**
```bash
python -m optimizer.loop \
    --script kernels/mla_reconstruction.py \
    --args "--model deepseek-v3 --ncu-mode" \
    --iters 3
```

## From Python

```python
from profiling.ncu_runner import run_ncu, load_profiles
from profiling.bottleneck import classify, report

csv_path = run_ncu("kernels/mla_reconstruction.py",
                   args="--ncu-mode", output="data/out.csv")
profiles = classify(load_profiles(csv_path))
print(report(profiles, label="mla_decode_bs1"))
```
