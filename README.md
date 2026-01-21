## Overview

This repository provides datasets and resources for the paper [**"Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs"**](https://arxiv.org/pdf/2505.02009). The datasets are curated to support research on safer and more responsible pretraining of language models.

## Folder Structure

```
TowardsSaferPretraining/
├── data/
│   ├── README.md
│   ├── HAVOC/
│   └── TTP-Eval/
├── jobs/                 # Snellius (Slurm) job entrypoints
├── prompts/
│   ├── HAVOC/
│   └── TTP/
├── scripts/              # local entrypoints (CPU/API/GPU depending on script)
├── src/                  # library code (clients, benchmarks, loaders, utils)
├── results/              # outputs (JSON) + CodeCarbon emissions
├── logs/                 # Slurm stdout/stderr (when using jobs/*.sh)
├── LICENSE
└── README.md
```

- **data/**: Contains the datasets used in the paper.
- **prompts/**: Includes prompts used for evaluating TTP and HAVOC.
- **scripts/**: Local, direct entrypoints for evaluation and report generation.
- **jobs/**: Snellius Slurm wrappers for GPU-backed runs.
- **LICENSE**: License information for the datasets and code.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/themendu/TowardsSaferPretraining.git
    cd TowardsSaferPretraining
    ```
2. Review the `data/README.md` for dataset details and usage instructions.
3. Use the prompts folder in case you need to evaluate TTP. The prompts are styled in OpenAI ChatML format.
4. To access HarmFormer’s predictions on the entire C4 dataset, please visit [this link](https://huggingface.co/datasets/themendu/SafeC4).
5. We release HarmFormer [here in HuggingFace](https://huggingface.co/themendu/HarmFormer)

## Reproducing paper results (what we currently support)

### Common setup (local or Snellius)

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

- **API keys via env**:
  - **OpenAI TTP**: `OPENAI_API_KEY` (Table 3, Table 4 “TTP” row, HAVOC judge=TTP)
  - **OpenRouter TTP**: `OPENROUTER_API_KEY` (+ optional `OPENROUTER_MODEL`, default `openai/gpt-4o`) (Table 3/4 via `--setups openrouter_ttp`)
  - **Gemini TTP**: `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`, default `gemini-2.0-flash`) (Table 4 Gemini row)
  - **Llama Guard (HF gated model)**: `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) (baseline comparisons)
- **Emissions tracking (CodeCarbon)**: enabled by default. Outputs go to `results/codecarbon/`. Set `DISABLE_CODECARBON=1` to disable.

### Local (CPU/API-only)

#### Table 4 (Toxic dimension): (Perspective omitted)

Table 4 in the paper includes a Perspective row. We do **not** run Perspective by default, but you can still reproduce
the other Table 4 baselines on TTP-Eval:

- **TTP (OpenAI)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups openai_ttp \
  --openai-key "$OPENAI_API_KEY" \
  --openai-model gpt-4o \
  --dimension toxic \
  --output results/ttp_eval/ttp_results.json
```

- **TTP (OpenRouter)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups openrouter_ttp \
  --openrouter-key "$OPENROUTER_API_KEY" \
  --openrouter-model "${OPENROUTER_MODEL:-openai/gpt-4o}" \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

- **HarmFormer (no API)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups harmformer \
  --device cuda \
  --dimension toxic \
  --output results/ttp_eval_baselines/results.json
```

Optional (requires HF access to gated models):
- `--setups llama_guard`
- `--setups local_ttp --local-model google/gemma-2-27b-it`

- **Gemini TTP (Gemini 2.0 Flash)**:

```bash
python scripts/evaluate_ttp_eval.py \
  --setups gemini_ttp \
  --gemini-key "$GEMINI_API_KEY" \
  --gemini-model "gemini-2.0-flash" \
  --dimension toxic \
  --output results/ttp_eval_baselines/table4_gemini_ttp.json
```

- **Local TTP (Gemma 2 27B + LLaMa 32B)** (requires HF access + `HUGGINGFACE_HUB_TOKEN`):
  - Set `GEMMA_2_27B_MODEL_ID` (default: `google/gemma-2-27b-it`)
  - Set `LLAMA_32B_MODEL_ID` (paper “R1 32B” row) to `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (public 32B distill release)
  - Then run: `sbatch jobs/run_ttp_eval_local.sh` (requests **2× A100** to fit these models without quantization)

#### Table 3 (Toxic dimension): TTP quality on TTP-Eval (OpenAI)

```bash
python scripts/evaluate_ttp_eval.py \
  --data-path data/TTP-Eval/TTPEval.tsv \
  --setups openai_ttp \
  --openai-model gpt-4o \
  --dimension toxic \
  --output results/ttp_eval/results.json
```

#### Table 10: HAVOC leakage from released `havoc_modeleval.tsv` (no model downloads)

Run one model column:

```bash
python scripts/evaluate_havoc_modeleval.py \
  --data-path data/HAVOC/havoc.tsv \
  --modeleval-path data/HAVOC/havoc_modeleval.tsv \
  --model-key gemma_9b \
  --output results/havoc/gemma_9b_results.json
```

Or run all model keys locally:

```bash
for k in gemma_9b llama_3b mistral_7b; do \
  python scripts/evaluate_havoc_modeleval.py --model-key \"$k\" --output \"results/havoc/${k}_results.json\"; \
done
```

#### Reproduction report (prints to stdout)

```bash
python scripts/generate_report.py
```

#### Table 8 (large-scale toxicity prevalence; **approximate**)

The paper’s Table 8 estimates toxicity prevalence on **1,000,000 samples** from webscale datasets (CommonCrawl, C4, FineWeb).
This repo does **not** ship those datasets; you must provide an input file. To fit a **$20 budget**, our prevalence script defaults to
**10,000 samples** (set `--limit 1000000` to match the paper’s sampling count).

Example (JSONL with a `text` field; uses the default `harmformer` setup to avoid per-sample API costs):

```bash
python scripts/estimate_toxicity_prevalence.py \
  --input-path /path/to/dataset.jsonl \
  --input-format jsonl \
  --text-field text \
  --output results/prevalence/prevalence_10k.json
```

### Snellius (Slurm)

Only needed for **GPU-backed runs** on Snellius (Slurm). If you’re running CPU/API locally, use the “Local” section above.

#### Setup venv on Snellius (once)

```bash
sbatch setup_env.sh
```

#### Table 6: HarmFormer quality on TTP-Eval (GPU recommended)

```bash
sbatch jobs/run_harmformer_eval.sh
```

#### Table 3 (Toxic dimension): TTP quality on TTP-Eval (OpenAI)

```bash
sbatch jobs/run_ttp_eval.sh
```

#### Baseline comparison job (GPU; runs what credentials enable)

This job can cover local baselines (HarmFormer, Llama Guard prompt variants) without API keys:

```bash
sbatch jobs/run_baselines_local.sh
```

If you want an API TTP row, run:

```bash
sbatch jobs/run_baselines_api.sh
```

#### Table 7 (OpenAI Moderation test set): Llama Guard / TTP / HarmFormer (Perspective omitted)

First, fetch the dataset (one-time):

```bash
git clone --depth 1 https://github.com/openai/moderation-api-release.git data/moderation-api-release
```

Then run:

```bash
python scripts/evaluate_openai_moderation.py \
  --output results/moderation/table7_results.json
```

#### Table 4 local-model rows (Gemma 2 27B; optional R1 model)

Run Table 4 with local Transformers models on Snellius (GPU). This is needed to reproduce the paper’s non-API rows
like **Gemma 2 27B**. The paper’s “R1 - LLaMa 32B” model id is not stable across releases; provide it via `R1_MODEL_ID`.

```bash
sbatch jobs/run_ttp_eval_local.sh
```

Environment variables:
- `GEMMA_2_27B_MODEL_ID` (default: `google/gemma-2-27b-it`)
- `R1_MODEL_ID` (optional; set to the HF model id you intend to match)
- `TTP_LOCAL_QUANTIZATION` (optional: `none|8bit|4bit`; requires `bitsandbytes` if not `none`)

#### Table 7 Slurm wrapper (GPU)

```bash
sbatch jobs/run_table7_moderation.sh
```

## License

See the [LICENSE](./LICENSE) file for details.

## Contact

For questions or contributions, please open an issue or contact the maintainers.
