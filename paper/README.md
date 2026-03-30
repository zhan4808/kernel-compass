# Paper draft (LaTeX)

Build (requires a LaTeX distribution with `pdflatex` and `bibtex`):

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or use `latexmk -pdf main.tex`.

## What to collect on a GPU (before final figures)

Run on **H100** (or document A100 separately) with **Nsight Compute** available (`ncu` on PATH). Without root, the repo falls back to **CUPTI-based estimates** — fine for development, but the paper should report **hardware NCU** numbers where possible.

| Artifact | Command | Purpose |
|----------|---------|---------|
| Validation timing | `python tests/test_validation.py` | Latency / TFLOPS table (Level 1) |
| Validation counters | `python tests/test_validation.py --ncu` | SM%, DRAM%, L2% vs expected bands; classifier check |
| Grid demo | `python -m optimizer.loop --grid 128 1 128 512 --precision bf16` | L2-resident: expect no FP8/INT4 candidates enumerated |
| Grid demo | `python -m optimizer.loop --grid 128 1 128 4096 --precision bf16` | HBM-bound: FP8 then INT4 profiled; acceptance per `verify()` |
| Compare mode | `python -m optimizer.loop --compare` | Grid vs LLM/heuristic: tries to first accept, wasted runs |
| MLA sweep (optional) | `python -m kernels.mla_reconstruction --model deepseek-v3 --ncu-mode` | Realistic batch-size sweep CSV |

**Export for the paper:** save NCU CSVs under `data/` (gitignored) and copy summarized tables into `main.tex` or a `figures/` CSV. Note GPU SKU, driver, CUDA, PyTorch, and Triton versions in an appendix.

**Not yet in the draft as measured claims:** native FP8 **W8A8** (cuBLASLt / Transformer Engine); only weight-only W8A16 is implemented.
