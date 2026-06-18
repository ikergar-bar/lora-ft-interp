# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

`lora-sentiment-interpretability-spec.md` is the authoritative spec for this project and **must be read in full before acting**. It defines milestones (M0–M7), each with verifiable acceptance criteria, plus an explicit out-of-scope section. Do not introduce dependencies or design changes that contradict it without annotating the deviation and its justification. This CLAUDE.md only summarizes the non-obvious constraints; the spec governs.

## Current state

The repo currently contains **only the spec** — no code, configs, or dependency files exist yet. The structure, commands, and setup below describe what the spec prescribes, not what is already present. When building, follow the spec's milestone order and stop at any acceptance criterion that fails rather than accumulating debt.

## What this project is

Fine-tune GPT-2 small (124M) with LoRA for SST-2 sentiment classification, then run an interpretability analysis of *what changed internally*. The central object of study is the **LoRA delta ΔW = BA** (low-rank, analyzable via SVD, per-layer norms, activation projection). Research question: does LoRA **create**, **sharpen**, or **relocate** a linear sentiment direction vs. the base model, and does the delta's location match where sentiment is computed causally? Baseline comparison: Tigges et al. (2023).

## Hard constraints (do not violate without annotating in the spec)

- **4 GB VRAM is a hard limit.** The whole design is sized for it: small model, base frozen (only the adapter trains), short `max_seq_len` (64–128). Memory valves if needed: lower batch + `gradient_accumulation_steps`, enable `gradient_checkpointing`, lower `max_seq_len`, or fall back to CPU for analysis/caching.
- **No QLoRA / 4-bit quantization of the base.** This is a design decision, not a preference: a numerically clean base is required to compare against the fine-tuned model. Quantization would contaminate the interpretability analysis.
- **Generative framing, not a classification head.** Prompt as `"Review: {text} Sentiment:"` and predict the `positive`/`negative` token. This enables logit attribution to label tokens, activation patching, and residual-stream direction analysis. A classification head breaks this and is out of scope.
- **Precision:** prefer `bf16` on Ampere+; a 4 GB card is likely Turing/Pascal → use `fp16`. Detect capability at runtime and log the choice.
- **Pin dependency versions.** `transformers` ↔ `transformer-lens` compatibility is fragile. Find a working set, verify `HookedTransformer.from_pretrained("gpt2")` loads without error, pin versions, and do not upgrade mid-project.
- **Activation caching:** in TransformerLens, `run_with_cache` caches everything by default — limit with `names_filter`, use small batches (8–32), and move the cache to CPU when it accumulates.
- **Merge parity is the most common silent failure (M2).** Always verify the merged model in TransformerLens produces the same logits as the merged HF model (within a documented tolerance) before drawing any interpretability conclusions. Without parity, the analysis is noise.

## Planned architecture

Pipeline staged across `src/` modules, in milestone order:

- `data.py` — load/format SST-2 into the generative framing.
- `train_lora.py` — PEFT LoRA fine-tuning loop (base frozen). Config in `configs/train.yaml`.
- `eval.py` — generative accuracy, base vs. fine-tuned.
- `merge.py` — `merge_and_unload()`, port the merged model to TransformerLens, run the **parity check**.
- `delta_analysis.py` — per-layer ΔW norms + SVD (effective rank).
- `directions.py` — linear sentiment direction via probing, base vs. fine-tuned (create/sharpen/relocate).
- `representations.py` — per-layer probing + UMAP of hidden states.
- `patching.py` — (stretch, M5) activation patching, cross-referenced with delta localization.

Training/analysis lives in **scripts under `src/`**; exploratory work and final figures in `notebooks/`, but **all figures must be regenerable from code**. Artifacts (adapters, merged models, figures) go in `artifacts/`; metrics/tables/plots in `results/`.

## Stack

GPT-2 small (or Pythia-160M) · SST-2 (GLUE) · HuggingFace `transformers` + `peft` · TransformerLens (hooks, `run_with_cache`, activation patching) · scikit-learn + umap-learn for probing/dim-reduction. Tracking via W&B or local CSV (optional).

## Setup & environment check

The agent must pass these before any training (adjust the torch wheel to the system's CUDA version):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url <CUDA wheel for the card>
pip install transformers peft datasets accelerate transformer-lens scikit-learn umap-learn matplotlib

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2'); print('TL OK')"
```

Note: this is a Windows environment (PowerShell); use `.venv\Scripts\Activate.ps1` to activate locally.

## Working conventions

- Work in small, verifiable steps: implement → run the milestone's acceptance criterion → continue. If a criterion fails, **stop and report** — do not push forward.
- M0–M4 + M7 are the core deliverable; **M5 (activation patching) is optional by design**.
- Always log: config used, global seed (fixed for determinism), peak VRAM (training and caching), metrics.
- No new dependency without pinning it and justifying why.
- Prefer small correct results over large unverified pipelines.
