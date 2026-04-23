"""Native PubMedQA evaluation for EAGLE.

This script stays inside the EAGLE evaluation package and uses only native
EAGLE decode paths:

- `naivegenerate()` for the repo's target-only baseline
- `eagenerate()` for native EAGLE speculative decoding

It is intended to answer one concrete question cleanly:
"On PubMedQA, what are the accuracy and token/sec of native EAGLE vs the
native EAGLE baseline path, without the outer SpecRAG harness?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from accelerate.utils import set_seed
from datasets import get_dataset_split_names, load_dataset
from transformers import AutoConfig

SCRIPT_DIR = Path(__file__).resolve().parent
EAGLE_ROOT = SCRIPT_DIR.parent.parent
if str(EAGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EAGLE_ROOT))

try:
    from ..model.ea_model import EaModel
except ImportError:
    from eagle.model.ea_model import EaModel


set_seed(0)

ANSWER_STYLE_LABEL_ONLY = "label_only"
ANSWER_STYLE_LABEL_RATIONALE = "label_rationale"


def _resolve_dataset_split(requested_split: str, available_splits: Sequence[str]) -> str:
    available = list(available_splits)
    if requested_split in available:
        return requested_split
    for fallback in ("train", "validation", "test"):
        if fallback in available:
            return fallback
    raise ValueError(
        f'Unknown split "{requested_split}". Available splits: {available}.'
    )


def _coerce_pubmedqa_passages(sample: Dict[str, Any]) -> List[str]:
    contexts = sample.get("contexts")
    if isinstance(contexts, dict):
        labels = contexts.get("labels", []) or []
        texts = contexts.get("contexts", []) or contexts.get("text", []) or []
        passages: List[str] = []
        for idx, text in enumerate(texts):
            text_str = str(text).strip()
            if not text_str:
                continue
            label = labels[idx] if idx < len(labels) else None
            passages.append(f"{label}: {text_str}" if label else text_str)
        if passages:
            return passages

    if isinstance(contexts, list):
        passages = [str(item).strip() for item in contexts if str(item).strip()]
        if passages:
            return passages

    for key in ("context", "passages", "documents"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            passages = [str(item).strip() for item in value if str(item).strip()]
            if passages:
                return passages

    return []


def normalize_pubmedqa_label(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if not lowered:
        return None

    explicit_patterns = (
        r"\b(?:final answer|the answer is|answer|label|decision)\s*[:\-]\s*(yes|no|maybe)\b",
        r"^\s*(yes|no|maybe)\s*(?:[\s\.,;:!\?-]|$)",
    )
    for pattern in explicit_patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1)

    first_line = lowered.splitlines()[0].strip()
    match = re.match(r"^(yes|no|maybe)\b", first_line)
    if match:
        return match.group(1)

    fallback = re.search(r"\b(yes|no|maybe)\b", lowered)
    if fallback:
        return fallback.group(1)
    return None


def _build_prompt(
    question: str,
    passages: Sequence[str],
    answer_style: str,
    rationale_min_words: int,
) -> str:
    docs = "\n".join(
        f"[Doc {idx + 1}] {passage}" for idx, passage in enumerate(passages[:5])
    )
    if docs:
        docs = f"\n\nRetrieved Documents:\n{docs}"

    if answer_style == ANSWER_STYLE_LABEL_ONLY:
        instructions = "Reply with exactly one word: yes, no, or maybe."
    elif answer_style == ANSWER_STYLE_LABEL_RATIONALE:
        instructions = (
            "Reply using exactly this format:\n"
            "Answer: yes, no, or maybe\n"
            f"Rationale: 1-2 short sentences grounded in the retrieved documents, with at least {rationale_min_words} words total.\n"
            "Do not omit the rationale."
        )
    else:
        raise ValueError(f"Unsupported answer style: {answer_style}")

    return (
        "You are answering a medical question.\n"
        f"{instructions}\n\n"
        f"Question: {question}{docs}\n\n"
        "Answer:"
    )


def _build_chat_input_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer([rendered], add_special_tokens=False, return_tensors="pt")
    else:
        encoded = tokenizer([prompt], return_tensors="pt")
    return encoded.input_ids.to(device)


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tokens_per_second(token_count: int, elapsed_sec: float) -> float:
    if token_count <= 0 or elapsed_sec <= 0:
        return 0.0
    return float(token_count) / float(elapsed_sec)


def _required_cache_length(model: EaModel, prompt_length: int, max_new_tokens: int) -> int:
    total_tokens = int(getattr(getattr(model, "ea_layer", None), "total_tokens", 0))
    return int(prompt_length + max_new_tokens + max(total_tokens + 16, 64))


def _strip_special_tokens(text: str, tokenizer) -> str:
    cleaned = text
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                if not special_tok:
                    continue
                cleaned = cleaned.replace(special_tok, "")
        else:
            if not special_token:
                continue
            cleaned = cleaned.replace(special_token, "")
    return cleaned.strip()


@dataclass
class ExampleResult:
    prediction: str
    normalized_prediction: Optional[str]
    generated_tokens: int
    wall_time: float
    decode_steps: int
    accuracy: float


def _decode_one(
    *,
    model: EaModel,
    tokenizer,
    input_ids: torch.Tensor,
    mode: str,
    max_new_tokens: int,
    device: torch.device,
) -> ExampleResult:
    max_length = _required_cache_length(model, int(input_ids.size(1)), max_new_tokens)

    _synchronize(device)
    start_time = time.perf_counter()
    if mode == "baseline":
        output_ids, _new_token, idx = model.naivegenerate(
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
        )
    elif mode == "eagle":
        output_ids, _new_token, idx = model.eagenerate(
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            log=True,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    _synchronize(device)
    wall_time = time.perf_counter() - start_time

    generated_ids = output_ids[0, input_ids.size(1):]
    generated_tokens = int(generated_ids.numel())
    decode_steps = max(0, int(idx) + 1) if generated_tokens > 0 else 0
    decoded = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
    prediction = _strip_special_tokens(decoded, tokenizer)
    normalized_prediction = normalize_pubmedqa_label(prediction)
    return ExampleResult(
        prediction=prediction,
        normalized_prediction=normalized_prediction,
        generated_tokens=generated_tokens,
        wall_time=wall_time,
        decode_steps=decode_steps,
        accuracy=0.0,
    )


def _load_pubmedqa(split: str):
    available = get_dataset_split_names("pubmed_qa", "pqa_labeled")
    resolved_split = _resolve_dataset_split(split, available)
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=resolved_split)
    return dataset, resolved_split


def _iter_samples(dataset: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        if limit > 0 and idx >= limit:
            break
        label = sample.get("final_decision") or sample.get("label") or sample.get("answer") or ""
        items.append(
            {
                "question_id": sample.get("pubid", idx),
                "question": str(sample.get("question", "")).strip(),
                "passages": _coerce_pubmedqa_passages(sample),
                "label": normalize_pubmedqa_label(str(label)) or str(label).strip().lower(),
            }
        )
    return items


def _print_mode_summary(
    *,
    mode: str,
    split: str,
    answer_style: str,
    warmup_examples: int,
    accuracy: float,
    num_examples: int,
    avg_generated_tokens: float,
    speed: float,
    acceptance: float,
    avg_decode_steps: float,
    total_wall_time: float,
) -> None:
    print(f"=== PubMedQA Summary: {mode} ===")
    print(f"Warmup examples: {warmup_examples}")
    print(f"Split: {split}")
    print(f"Answer style: {answer_style}")
    print(f"Accuracy: {accuracy:.3f} ({num_examples} examples)")
    print(f"Avg generated tokens: {avg_generated_tokens:.2f}")
    print(f"Speed: {speed:.2f} tok/s")
    print(f"Acceptance: {acceptance:.2f}")
    print(f"Avg decode steps: {avg_decode_steps:.2f}")
    print(f"Total wall time: {total_wall_time:.2f}s")


def _dtype_for_model(base_model_path: str) -> torch.dtype:
    arch = AutoConfig.from_pretrained(base_model_path).architectures[0]
    if arch == "Qwen2ForCausalLM" and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


@torch.inference_mode()
def run_pubmedqa_eval(
    *,
    base_model_path: str,
    ea_model_path: str,
    mode: str,
    split: str,
    num_samples: int,
    max_new_tokens: int,
    warmup_examples: int,
    total_token: int,
    depth: int,
    top_k: int,
    use_eagle3: bool,
    answer_style: str,
    rationale_min_words: int,
    output_dir: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    if answer_style not in (ANSWER_STYLE_LABEL_ONLY, ANSWER_STYLE_LABEL_RATIONALE):
        raise ValueError(f"Unsupported answer style: {answer_style}")
    if rationale_min_words < 1:
        raise ValueError("rationale_min_words must be >= 1.")

    dataset, resolved_split = _load_pubmedqa(split)
    samples = _iter_samples(dataset, num_samples)
    if not samples:
        raise ValueError("Resolved PubMedQA dataset is empty.")

    load_kwargs: Dict[str, Any] = {
        "use_eagle3": use_eagle3,
        "base_model_path": base_model_path,
        "ea_model_path": ea_model_path,
        "total_token": total_token,
        "depth": depth,
        "top_k": top_k,
        "torch_dtype": _dtype_for_model(base_model_path),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    print("Loading native EAGLE model...")
    model = EaModel.from_pretrained(**load_kwargs)
    model.eval()
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _module_device(model.base_model)

    modes = ["baseline", "eagle"] if mode == "compare" else [mode]

    warmup_count = min(max(0, warmup_examples), len(samples))
    for active_mode in modes:
        if warmup_count <= 0:
            continue
        if verbose:
            print(f"Warming up {active_mode} on {warmup_count} example(s)...")
        for item in samples[:warmup_count]:
            prompt = _build_prompt(
                item["question"],
                item["passages"],
                answer_style,
                rationale_min_words,
            )
            input_ids = _build_chat_input_ids(tokenizer, prompt, device)
            _ = _decode_one(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                mode=active_mode,
                max_new_tokens=max_new_tokens,
                device=device,
            )

    summaries: Dict[str, Any] = {}
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for active_mode in modes:
        total_accuracy = 0.0
        total_generated_tokens = 0
        total_wall_time = 0.0
        total_decode_steps = 0
        predictions_path = (
            os.path.join(output_dir, f"{active_mode}.jsonl") if output_dir else None
        )
        if predictions_path and os.path.exists(predictions_path):
            os.remove(predictions_path)

        for idx, item in enumerate(samples):
            prompt = _build_prompt(
                item["question"],
                item["passages"],
                answer_style,
                rationale_min_words,
            )
            input_ids = _build_chat_input_ids(tokenizer, prompt, device)
            result = _decode_one(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                mode=active_mode,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            result.accuracy = float(result.normalized_prediction == item["label"])
            total_accuracy += result.accuracy
            total_generated_tokens += result.generated_tokens
            total_wall_time += result.wall_time
            total_decode_steps += result.decode_steps

            if predictions_path:
                with open(predictions_path, "a", encoding="utf-8") as fout:
                    fout.write(
                        json.dumps(
                            {
                                "question_id": item["question_id"],
                                "mode": active_mode,
                                "answer_style": answer_style,
                                "label": item["label"],
                                "prediction": result.prediction,
                                "normalized_prediction": result.normalized_prediction,
                                "accuracy": result.accuracy,
                                "generated_tokens": result.generated_tokens,
                                "wall_time": result.wall_time,
                                "decode_steps": result.decode_steps,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

            if verbose:
                running_speed = _tokens_per_second(total_generated_tokens, total_wall_time)
                print(
                    f"[{active_mode} {idx + 1}/{len(samples)}] "
                    f"acc={result.accuracy:.1f} "
                    f"pred={result.normalized_prediction or 'n/a'} "
                    f"speed={running_speed:.2f} tok/s "
                    f"t={result.wall_time:.2f}s"
                )

        avg_accuracy = total_accuracy / len(samples)
        avg_generated_tokens = total_generated_tokens / len(samples)
        avg_decode_steps = total_decode_steps / len(samples)
        speed = _tokens_per_second(total_generated_tokens, total_wall_time)
        acceptance = (
            float(total_generated_tokens) / float(total_decode_steps)
            if total_decode_steps > 0
            else 0.0
        )
        summaries[active_mode] = {
            "split": resolved_split,
            "answer_style": answer_style,
            "warmup_examples": warmup_count,
            "accuracy": avg_accuracy,
            "num_examples": len(samples),
            "avg_generated_tokens": avg_generated_tokens,
            "speed": speed,
            "acceptance": acceptance,
            "avg_decode_steps": avg_decode_steps,
            "total_wall_time": total_wall_time,
            "predictions_path": predictions_path,
        }

        _print_mode_summary(
            mode=active_mode,
            split=resolved_split,
            answer_style=answer_style,
            warmup_examples=warmup_count,
            accuracy=avg_accuracy,
            num_examples=len(samples),
            avg_generated_tokens=avg_generated_tokens,
            speed=speed,
            acceptance=acceptance,
            avg_decode_steps=avg_decode_steps,
            total_wall_time=total_wall_time,
        )

    if mode == "compare":
        baseline_speed = summaries["baseline"]["speed"]
        eagle_speed = summaries["eagle"]["speed"]
        speedup = eagle_speed / baseline_speed if baseline_speed > 0 else 1.0
        print("=== Native Compare Summary ===")
        print(f"Split: {resolved_split}")
        print(f"Answer style: {answer_style}")
        print(
            f"Baseline accuracy/speed: "
            f"{summaries['baseline']['accuracy']:.3f} / {baseline_speed:.2f} tok/s"
        )
        print(
            f"EAGLE accuracy/speed: "
            f"{summaries['eagle']['accuracy']:.3f} / {eagle_speed:.2f} tok/s"
        )
        print(f"Speedup: {speedup:.2f}x")
        summaries["compare"] = {"speedup": speedup}

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native EAGLE PubMedQA evaluation")
    parser.add_argument(
        "--mode",
        choices=("compare", "baseline", "eagle"),
        default="compare",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="thoughtworks/Qwen2.5-7B-Instruct-Eagle3",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-examples", type=int, default=2)
    parser.add_argument("--total-token", type=int, default=32)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--answer-style",
        choices=(ANSWER_STYLE_LABEL_ONLY, ANSWER_STYLE_LABEL_RATIONALE),
        default=ANSWER_STYLE_LABEL_ONLY,
    )
    parser.add_argument("--rationale-min-words", type=int, default=20)
    parser.add_argument("--use-eagle3", action="store_true")
    parser.add_argument("--disable-eagle3", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    use_eagle3 = True
    if args.disable_eagle3:
        use_eagle3 = False
    elif args.use_eagle3:
        use_eagle3 = True

    run_pubmedqa_eval(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        mode=args.mode,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_examples=args.warmup_examples,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        use_eagle3=use_eagle3,
        answer_style=args.answer_style,
        rationale_min_words=args.rationale_min_words,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
