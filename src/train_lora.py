"""LoRA fine-tuning for GPT-2 on SST-2 (spec §5, hito M1).

Trains a LoRA adapter (base frozen) using the generative framing:
    "Review: {text} Sentiment: positive|negative"
Only the label token position contributes to the loss (all prompt positions
are masked with -100). Base weights are never updated.

Usage:
    python -m src.train_lora                          # full run (GPU)
    python -m src.train_lora --max-train-samples 64  # smoke-test (CPU)
    python -m src.train_lora --smoke                 # alias for --max-train-samples 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.data import build_train_examples
from src.utils import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    get_device,
    load_config,
    log_result,
    resolve_dtype,
    set_seed,
)


def _tokenize_examples(
    examples: list[tuple[str, str]], tokenizer, max_seq_len: int
) -> list[dict]:
    """Tokenize (prompt, label_token) pairs with causal label masking.

    Prompt tokens are masked with -100; only the label token contributes
    to the cross-entropy loss. If the full sequence exceeds max_seq_len,
    the prompt is truncated from the right so the label token is always kept.
    """
    records = []
    for prompt_text, label_token in examples:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        label_ids = tokenizer.encode(label_token, add_special_tokens=False)

        max_prompt_len = max_seq_len - len(label_ids)
        if max_prompt_len <= 0:
            max_prompt_len = 1
        prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + label_ids
        labels = [-100] * len(prompt_ids) + label_ids
        records.append({"input_ids": input_ids, "labels": labels})
    return records


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def train(config: dict, max_train_samples: Optional[int] = None) -> dict:
    seed = config["training"]["seed"]
    set_seed(seed)
    device = get_device()
    dtype = resolve_dtype(config["model"].get("precision"), device)

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_examples = build_train_examples(
        split="train",
        template=config["data"]["prompt_template"],
        pos_token=config["data"]["pos_token"],
        neg_token=config["data"]["neg_token"],
    )
    if max_train_samples is not None:
        raw_examples = raw_examples[:max_train_samples]

    max_seq_len = config["data"]["max_seq_len"]
    tokenized = _tokenize_examples(raw_examples, tokenizer, max_seq_len)
    dataset = _ListDataset(tokenized)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_cfg = config["training"]
    precision = config["model"].get("precision", "fp16")
    use_fp16 = device.type == "cuda" and precision == "fp16"
    use_bf16 = device.type == "cuda" and precision == "bf16"
    use_gradient_checkpointing = training_cfg.get("gradient_checkpointing", False)

    if use_gradient_checkpointing:
        # Required for PEFT + gradient_checkpointing: ensures inputs require grad.
        model.enable_input_require_grads()

    output_dir = str(REPO_ROOT / "artifacts" / "trainer_tmp")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_epochs"]),
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        logging_steps=config["logging"]["log_every_n_steps"],
        seed=seed,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to="none",
        save_strategy="no",
        dataloader_pin_memory=(device.type == "cuda"),
        dataloader_num_workers=0,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    train_output = trainer.train()

    peak_vram_mb: Optional[float] = None
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        print(f"[train] Peak VRAM: {peak_vram_mb} MB")

    adapter_dir = str(REPO_ROOT / config["output"]["adapter_dir"])
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"[train] Adapter saved to {adapter_dir}")

    return {
        "milestone": "M1",
        "model": model_name,
        "seed": seed,
        "device": device.type,
        "dtype": str(dtype),
        "fp16_training": use_fp16,
        "n_train_examples": len(dataset),
        "num_epochs": training_cfg["num_epochs"],
        "peak_vram_mb": peak_vram_mb,
        "train_loss": round(train_output.training_loss, 4),
        "adapter_dir": adapter_dir,
        "smoke_test": max_train_samples is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning (M1).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training set size (smoke-test on CPU).",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Alias for --max-train-samples 64."
    )
    args = parser.parse_args()

    max_samples = args.max_train_samples
    if args.smoke and max_samples is None:
        max_samples = 64

    config = load_config(args.config)
    result = train(config, max_train_samples=max_samples)
    log_result("train_lora", result)


if __name__ == "__main__":
    main()
