"""SST-2 loading and formatting into the generative framing (spec §4 decision 1).

Prompt framing: "Review: {text} Sentiment:" -> predict the " positive"/" negative"
token. We never use a classification head (out of scope, spec §11).
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from src.utils import REPO_ROOT

# SST-2 GLUE label convention: 0 = negative, 1 = positive.
LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
DATA_CACHE_DIR = REPO_ROOT / "data"


@dataclass
class EvalExample:
    prompt: str
    gold_label: int  # 0 = negative, 1 = positive


def load_sst2(split: str = "validation"):
    """Load an SST-2 split from GLUE, cached under data/.

    Note: the ``test`` split has hidden labels (all -1), so accuracy must be
    measured on ``validation`` (872 labelled examples).
    """
    # datasets>=5.0 requires the full namespace; "glue" alone fails URI parsing.
    return load_dataset("nyu-mll/glue", "sst2", split=split, cache_dir=str(DATA_CACHE_DIR))


def format_prompt(text: str, template: str) -> str:
    """Apply the generative prompt template to a raw review."""
    return template.format(text=text.strip())


def build_eval_set(split: str, template: str) -> list[EvalExample]:
    """Build (prompt, gold_label) pairs for generative evaluation."""
    dataset = load_sst2(split)
    return [
        EvalExample(prompt=format_prompt(row["sentence"], template), gold_label=int(row["label"]))
        for row in dataset
    ]
