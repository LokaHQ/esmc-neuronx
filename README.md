# ESMC-300M on AWS Trainium2 with torch.compile

This repository contains Loka's ESMC-specific Trainium2 benchmark package for AWS Trainium2.

It is intentionally focused: it does not vendor the full ESM library. Instead, it contains
the small Neuron adapter (`esm_neuron/`), benchmark scripts, measured results, and the HTML
blog artifact that can be published with GitHub Pages.

## Why ESMC

[ESMC-300M](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) is EvolutionaryScale's
protein language model from the ESM Cambrian (ESM-C) family. It ships with open weights and
three sizes (300M, 600M, 3B), and is trained on hundreds of millions of protein sequences from
the ESM Atlas and UniProt.

Architecturally, ESMC is an encoder-only Transformer — it produces per-residue representations
rather than generating new sequences. That makes it a natural fit for the most common
production use case in bio and HCLS: **scoring every possible single-amino-acid substitution
in a protein and predicting which ones are pathogenic**. The zero-shot log-probability delta
is an established scoring approach, and ESMC is strong at it.

For infrastructure teams, the most practical detail is that ESMC loads as a standard
PyTorch `nn.Module` with no unusual operator requirements, and runs immediately via
`torch.compile(backend="neuron")` after disabling Flash Attention. That combination is what
made a fast Trainium2 path possible without a model rewrite.

## Key Result

ESMC-300M compiled and ran on a single `trn2.3xlarge` with 4 NeuronCores using
`torch.compile(backend="neuron")`. The evaluation task is D2Deep: zero-shot pathogenicity
scoring of 125,190 human protein variants.

| Hardware | Config | Throughput | ROC-AUC | Hourly cost | Cost / M samples |
|---|---|---:|---:|---:|---:|
| Trainium2 trn2.3xlarge | 4×NeuronCore, batch=1 | **220.3 s/s** | 0.8525 | $2.235 | **$2.82** |
| Trainium2 trn2.3xlarge | 1×NeuronCore, batch=16 | 101.8 s/s | 0.8525 | $2.235 | $6.10 |
| H100 p5.4xlarge | 1 GPU, batch=16 | 414.0 s/s | 0.8524 | $10.18 | $6.83 |
| RTX 5060 Ti | 1 GPU, batch=16 | 77.7 s/s | 0.8524 | — | — |

**Trainium2 is 2.4× more cost-efficient than H100** at the currently measured configurations
(220 s/s at $2.235/hr vs 414 s/s at $10.18/hr). Once the batch>1 compiler regression is resolved,
the projected Trainium2 4×batch=16 (~407 s/s) would match H100 throughput at **4.6× lower cost**.

At batch=1 per unit, a single Trainium2 NeuronCore (~60 s/s) is **1.67× faster than one H100 GPU**
(35.9 s/s) — the NeuronCore's smaller MatMul engine is better suited to single-sample encoder inference.

## The ESMC Neuron Fix

This is the main practical compatibility lesson.

ESMC uses Flash Attention by default, which is not supported on Trainium2. The fix is
one argument at load time:

```python
from esm.pretrained import ESMC_300M_202412

model = ESMC_300M_202412("cpu", use_flash_attn=False)
model = model.to(torch.bfloat16).to("neuron").eval()
```

Without `use_flash_attn=False`, the model will fail to compile on Neuron. This is the
equivalent of Carbon's tokenizer fix: a small one-line change that is easy to miss and
blocks every downstream benchmark.

The `esm_neuron/` package in this repository handles this automatically via
`NeuronESMC("esmc_300m")`.

## Dataset

The benchmark uses **D2Deep** — 125,190 single-amino-acid substitution variants from human proteins
with binary pathogenicity labels (109,700 benign / 15,490 pathogenic).

The compiled CSV is available from the Loka S3 benchmark bucket:

```bash
aws s3 cp \
  s3://torch-neuronx-loka-datasets-846430536449-sa-east-1/d2deep_data/compiled_data/compiled_data.csv \
  data/compiled_data.csv
```

## Model Weights

ESMC-300M weights download automatically from HuggingFace on first run via the `esm` library.
The model (`EvolutionaryScale/esmc-300m-2024-12`) is gated — you need to accept the licence at
[huggingface.co/EvolutionaryScale/esmc-300m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12)
and authenticate once:

```bash
huggingface-cli login   # paste your HF token when prompted
```

Weights are cached to `~/.cache/huggingface/` and reused on subsequent runs (~1.2 GB).

## Quick Start on `trn2.3xlarge`

Use an AWS Neuron DLAMI or Neuron DLC with the Native PyTorch Neuron Engine.

```bash
git clone <repository-url>
cd esmc-neuronx

bash scripts/bootstrap.sh
source .venv-esmc-neuron/bin/activate
```

Split the D2Deep dataset and run the 4-worker benchmark:

```bash
python -c "
import pandas as pd, pathlib
df = pd.read_csv('data/compiled_data.csv')
out = pathlib.Path('data/d2deep_splits4'); out.mkdir(parents=True, exist_ok=True)
n = len(df) // 4
for i in range(4):
    df.iloc[i*n:(i+1)*n if i < 3 else len(df)].to_csv(out/f'part{i}.csv', index=False)
"

SHARD_DIR=data/d2deep_splits4 bash scripts/run_d2deep_neuron.sh
```

Results are written to `results/`.

## Single-Core Smoke Test

```bash
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_NUM_CORES=1 \
python experiments/esmc_d2deep_benchmark.py \
    --device neuron \
    --d2deep-csv data/compiled_data.csv \
    --batch-size 1 \
    --compile-strategy regional \
    --limit 500 \
    --summary-json results/smoke.json
```

## Compile Strategies

**Regional compile (recommended for batch=1):**
Compiles each transformer block individually. Avoids full-model graph complexity.
~277 s first-time compile per core, ~20 s cached load.

```bash
python experiments/esmc_d2deep_benchmark.py \
    --device neuron --batch-size 1 --compile-strategy regional \
    --d2deep-csv data/compiled_data.csv --summary-json results/regional.json
```

**Full compile (for batch=16, when supported):**
Compiles the entire model graph. Required for batch>1 once the compiler
regression is resolved.

```bash
python experiments/esmc_d2deep_benchmark.py \
    --device neuron --batch-size 16 --compile-strategy full \
    --d2deep-csv data/compiled_data.csv --summary-json results/full_bs16.json
```

See [docs/index.html](docs/index.html) for the publishable benchmark writeup.

## GitHub Pages Blog

The blog is checked in as:

```text
docs/index.html
```

The intended Pages source is the `docs/` directory on the `main` branch. The repository
includes `docs/.nojekyll` so GitHub Pages serves the static HTML and assets directly.

## Technical Post

The publishable technical article lives in `docs/index.html`. It covers the benchmark
setup, the Flash Attention fix, D2Deep results, per-unit and full-instance comparisons,
and cost analysis.

## What This Repo Includes

- `esm_neuron/` — ESMC/ESM3 Neuron adapter: loads with Flash Attention disabled,
  wraps with `torch.compile(backend="neuron")`. This is the "patch" equivalent
  for ESMC — a small adapter rather than a model rewrite.
- `scripts/` — setup, benchmark, and S3 upload helpers.
- `experiments/` — self-contained Python benchmark script.
- `results/` — curated JSON benchmark results from Trainium2 and H100 runs.
- `manifests/` — plain-text run manifests and result summaries.
- `docs/index.html` — the GitHub Pages blog post.

## Repository Layout

```text
.
├── esm_neuron/       # Neuron adapter package
├── docs/             # GitHub Pages static site
├── experiments/      # Python benchmark scripts
├── manifests/        # Plain-text experiment manifests
├── results/          # Curated benchmark JSON outputs
└── scripts/          # Bootstrap, benchmark, upload helpers
```

## Known Caveats

- **Batch>1 compile failure**: `vector::reserve` compiler error in the Neuron compiler
  prevents batch_size>1 for ESMC-300M seq_len=512 on the current beta stack. Workaround
  is batch=1 per NeuronCore with 4 parallel workers. This is a compiler regression,
  not an architectural limitation.
- **Missing 4×batch=16 measurement**: the projected Trainium2 full-instance result at
  4×batch=16 (~407 samples/s) has not been measured due to the batch>1 regression.
  It is expected to approximately match H100 at batch=16 (414 samples/s).
- **Warmup cost**: regional compile takes ~277 s first-time per NeuronCore.
  Subsequent runs load cached NEFFs in ~20 s.

## License

The `esm_neuron/` adapter and benchmark scripts in this repository are provided under
this repository's license. The ESM library (`esm` package) is subject to EvolutionaryScale's
license terms.
