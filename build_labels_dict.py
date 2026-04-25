"""
Build a dictionary mapping feature indices to:
    {id: {"explanation": {"label": <=7 words, "description": full text}}}

Uses the SAME local model that produced the delphi explanations
(unsloth/Meta-Llama-3.1-8B-Instruct) via vLLM for fast batched inference.

Usage:
    python build_labels_dict.py \
        --explanations_dir results/gemma3_4b_it_layer15/explanations \
        --model unsloth/Meta-Llama-3.1-8B-Instruct

Requires:
    pip install vllm transformers
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# PROMPT — Llama-3.1-Instruct chat format (system + user + few-shot)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are labeling features from a Sparse Autoencoder. Given a LONG "
    "EXPLANATION describing what a single feature activates on, produce a "
    "SHORT LABEL.\n\n"
    "Rules:\n"
    "- MUST be 7 words or fewer\n"
    "- Concise noun phrase capturing the core concept\n"
    "- No quotes, no trailing punctuation, no preamble, no reasoning\n"
    "- Output ONLY the label text — nothing else"
)

USER_PROMPT_TEMPLATE = """Examples:
Explanation: "Activates on cooking verbs in French recipes, especially preparation steps."
Short label: French cooking verbs in recipes

Explanation: "Fires on closing punctuation of declarative sentences in news articles."
Short label: Sentence-ending punctuation in news

Explanation: "Activates on variable names referring to loop counters in Python code."
Short label: Python loop counter variables

Now label this one:
Explanation: \"\"\"{explanation}\"\"\"
Short label (≤7 words):"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def build_chat_prompts(explanations: list[str], tokenizer) -> list[str]:
    """Apply Llama-3.1-Instruct chat template to a list of explanations."""
    prompts = []
    for exp in explanations:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(explanation=exp.strip())},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def detect_layer_name(explanations_dir: str) -> str:
    """
    Scan the explanations directory and extract the layer identifier
    (e.g. 'layers.15') from the filenames.
    """
    explanations_path = Path(explanations_dir)
    layer_pattern = re.compile(r"(layers\.\d+)")

    found: set[str] = set()
    for txt_file in explanations_path.glob("*.txt"):
        m = layer_pattern.search(txt_file.name)
        if m:
            found.add(m.group(1))

    if not found:
        raise ValueError(
            f"Could not detect a 'layers.N' segment in any filename under {explanations_dir}"
        )
    if len(found) > 1:
        raise ValueError(
            f"Explanations directory contains multiple layers: {sorted(found)}. "
            "Run this script once per layer."
        )
    return found.pop()


def detect_layer_number(explanations_dir: str) -> int:
    """Extract just the layer number (e.g. 15) from filenames."""
    layer_name = detect_layer_name(explanations_dir)
    return int(layer_name.split(".")[-1])


def clean_label(raw: str, max_words: int = 7) -> str:
    """Sanitize a raw model output into a clean short label."""
    label = raw.strip()
    for line in label.splitlines():
        if line.strip():
            label = line.strip()
            break
    label = re.sub(r"^\s*short label\s*[:\-]\s*", "", label, flags=re.IGNORECASE)
    label = label.strip().strip('"').strip("'").rstrip(".!?,;:")
    words = label.split()
    if len(words) > max_words:
        label = " ".join(words[:max_words])
    return label


# ---------------------------------------------------------------------------
# SUMMARIZATION FUNCTION (batched via vLLM)
# ---------------------------------------------------------------------------

def summarize_labels_batch(
    explanations: list[str],
    llm: LLM,
    tokenizer,
    max_words: int = 7,
) -> list[str]:
    """Summarize a batch of long explanations into short labels."""
    prompts = build_chat_prompts(explanations, tokenizer)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        stop=["\n\n", "<|eot_id|>"],
    )

    outputs = llm.generate(prompts, sampling_params)
    return [clean_label(o.outputs[0].text, max_words=max_words) for o in outputs]


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def build_labels_dict(
    explanations_dir: str,
    llm: LLM,
    tokenizer,
) -> dict[int, dict]:
    """
    Read all explanation .txt files and build:
        {feature_index: {"explanation": {"label": str, "description": str}}}
    """
    explanations_path = Path(explanations_dir)

    items: list[tuple[int, str]] = []
    for txt_file in explanations_path.glob("*.txt"):
        match = re.search(r"_latent(\d+)\.txt$", txt_file.name)
        if not match:
            continue
        idx = int(match.group(1))
        long_exp = txt_file.read_text(encoding="utf-8").strip().strip('"')
        items.append((idx, long_exp))

    if not items:
        raise FileNotFoundError(f"No explanation .txt files in {explanations_dir}")

    items.sort(key=lambda x: x[0])

    indices = [i for i, _ in items]
    explanations = [e for _, e in items]

    print(f"Summarizing {len(explanations)} features with vLLM...")
    short_labels = summarize_labels_batch(explanations, llm, tokenizer)

    labels = {
        idx: {"explanation": {"label": short, "description": long_exp}}
        for idx, short, long_exp in zip(indices, short_labels, explanations)
    }
    return dict(sorted(labels.items()))


def main():
    parser = argparse.ArgumentParser(
        description="Build {id: {explanation: {label, description}}} dict"
    )
    parser.add_argument(
        "--explanations_dir", type=str,
        default="results/gemma3_4b_it_layer15/explanations",
        help="Path to the delphi explanations directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="feature_labels",
        help="Output directory (created if it doesn't exist)",
    )
    parser.add_argument(
        "--model", type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="Same local model used for explanation generation",
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # Auto-detect layer number
    layer_num = detect_layer_number(args.explanations_dir)
    print(f"Auto-detected layer: {layer_num}")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Output file named by layer
    output_file = output_path / f"feature_labels_layer_{layer_num}.json"

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    labels = build_labels_dict(
        explanations_dir=args.explanations_dir,
        llm=llm,
        tokenizer=tokenizer,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\nTotal features labeled: {len(labels)}")
    print(f"Saved to: {output_file}")

    print("\nExamples:")
    for k, v in list(labels.items())[:5]:
        print(f"  [{k}] {v['explanation']['label']}")
        print(f"       └─ {v['explanation']['description'][:90]}...")


if __name__ == "__main__":
    main()
    