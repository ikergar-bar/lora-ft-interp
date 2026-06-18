"""Generative evaluation of sentiment accuracy (spec §4 decision 1, hito M0/M1).

We score each prompt by comparing the next-token logits of the " positive" and
" negative" label tokens; argmax of the two is the prediction. Works for the base
model and (via --adapter) the LoRA fine-tuned model.

Usage:
    python -m src.eval                       # base GPT-2 on SST-2 validation
    python -m src.eval --adapter artifacts/lora_adapter   # fine-tuned (M1+)
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_eval_set
from src.utils import (
    DEFAULT_CONFIG,
    get_device,
    load_config,
    log_result,
    resolve_dtype,
    set_seed,
)


def _resolve_label_token_id(tokenizer, label: str) -> int:
    """Resolve a label string (e.g. " positive") to a single token id.

    The generative scoring compares the logits of one token per class. If the
    label tokenizes to multiple tokens we fall back to the first one and warn,
    since comparing the first token still discriminates the two classes here.
    """
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) != 1:
        print(
            f"[warn] label {label!r} -> {len(ids)} tokens {ids}; "
            f"using first token id {ids[0]} for scoring."
        )
    return ids[0]


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def evaluate(config: dict, adapter: str | None = None) -> dict:
    set_seed(config["training"]["seed"])
    device = get_device()
    dtype = resolve_dtype(config["model"].get("precision"), device)

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token; reuse eos. Left padding keeps the last real token
    # at position -1 so logits[:, -1, :] is the next-token distribution.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()

    pos_id = _resolve_label_token_id(tokenizer, config["data"]["pos_token"])
    neg_id = _resolve_label_token_id(tokenizer, config["data"]["neg_token"])

    split = config["data"].get("eval_split", "validation")
    examples = build_eval_set(split, config["data"]["prompt_template"])
    batch_size = config["training"]["batch_size"]
    max_len = config["data"]["max_seq_len"]

    correct = 0
    for batch in _batched(examples, batch_size):
        prompts = [ex.prompt for ex in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        logits = model(**enc).logits[:, -1, :]  # next-token logits
        # Predict positive (label 1) when its logit beats negative's.
        preds = (logits[:, pos_id] > logits[:, neg_id]).long()
        gold = torch.tensor([ex.gold_label for ex in batch], device=device)
        correct += int((preds == gold).sum())

    accuracy = correct / len(examples)
    return {
        "milestone": "M0" if adapter is None else "eval",
        "model": model_name,
        "adapter": adapter,
        "split": split,
        "seed": config["training"]["seed"],
        "device": device.type,
        "dtype": str(dtype),
        "pos_token_id": pos_id,
        "neg_token_id": neg_id,
        "n_examples": len(examples),
        "accuracy": round(accuracy, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative sentiment eval.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (M1+).")
    args = parser.parse_args()

    config = load_config(args.config)
    result = evaluate(config, adapter=args.adapter)
    name = "eval_base" if args.adapter is None else "eval_finetuned"
    log_result(name, result)


if __name__ == "__main__":
    main()
