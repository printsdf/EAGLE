"""Pure-EAGLE greedy vs eagenerate lossless audit on MedQA.

This script intentionally avoids SpecRAG / MemoryDecoder code paths.
It compares:

1. target-only greedy decoding through ``EaModel.base_model`` with temperature=0
2. native ``EaModel.eagenerate()`` with temperature=0

Both runs use the same chat-template prompt and the same MedQA sample subset.
When a mismatch happens, the script also reports the EAGLE verification-step
record that covers the first mismatching generated token.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import reset_tree_mode
except ImportError:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import reset_tree_mode


set_seed(0)

MEDQA_DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
MEDQA_SPLIT_FALLBACKS = ("test", "validation", "train")


def _resolve_dataset_split(requested_split: str, available_splits: Sequence[str]) -> str:
    available = list(available_splits)
    if requested_split in available:
        return requested_split
    for fallback in MEDQA_SPLIT_FALLBACKS:
        if fallback in available:
            return fallback
    raise ValueError(
        f'Unknown split "{requested_split}". Available splits: {available}.'
    )


def normalize_medqa_label(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if not lowered:
        return None

    candidate_sections: List[str] = []
    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    if lines:
        candidate_sections.append(lines[0])
        if len(lines) > 1 and lines[0].startswith(("label", "answer", "option", "prediction")):
            candidate_sections.append(lines[1])
    candidate_sections.append(lowered[:160])

    explicit_patterns = (
        r"\b(?:final answer|correct answer|answer|label|option|choice|prediction)\s*[:\-]?\s*\(?([abcd])\)?\b",
        r"^\s*\(?([abcd])\)?\s*(?:[\s\.,;:!\?\-]|$)",
    )

    for section in candidate_sections:
        normalized = re.sub(r"[_\-]", " ", section)
        for pattern in explicit_patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).upper()

    fallback = re.search(r"\b([abcd])\b", lowered)
    if fallback:
        return fallback.group(1).upper()
    return None


def _medqa_to_item(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = str(sample.get("question") or sample.get("query") or "").strip()
    options = sample.get("options") or {}
    if not isinstance(options, dict):
        options = {}

    normalized_label = normalize_medqa_label(
        str(
            sample.get("answer_idx")
            or sample.get("label")
            or sample.get("answer")
            or ""
        )
    ) or ""

    option_lines: List[str] = []
    for key in ("A", "B", "C", "D"):
        value = options.get(key) or options.get(key.lower())
        if value is None:
            continue
        option_lines.append(f"{key}. {str(value).strip()}")

    prompt = (
        "Medical multiple-choice question. Select the single best answer.\n"
        "Reply in exactly two lines using this format:\n"
        "Label: A|B|C|D\n"
        "Rationale: 1-2 short sentences explaining the choice.\n\n"
        f"Question: {question}"
    )
    if option_lines:
        prompt += "\nOptions:\n" + "\n".join(option_lines)

    return {
        "question_id": sample.get("id", index),
        "prompt": prompt,
        "label": normalized_label,
    }


def _load_medqa_samples(split: str, num_samples: int, sample_offset: int) -> Tuple[List[Dict[str, Any]], str]:
    available = get_dataset_split_names(MEDQA_DATASET_NAME)
    resolved_split = _resolve_dataset_split(split, available)
    dataset = load_dataset(MEDQA_DATASET_NAME, split=resolved_split)
    if sample_offset < 0:
        raise ValueError("sample_offset must be >= 0")

    start = min(sample_offset, len(dataset))
    if num_samples == 0:
        stop = len(dataset)
    else:
        stop = min(start + max(0, int(num_samples)), len(dataset))
    items = [_medqa_to_item(dataset[i], i) for i in range(start, stop)]
    return items, resolved_split


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
    if token_count <= 0 or elapsed_sec <= 0.0:
        return 0.0
    return float(token_count) / float(elapsed_sec)


def _dtype_for_model(base_model_path: str) -> torch.dtype:
    arch = AutoConfig.from_pretrained(base_model_path).architectures[0]
    if arch == "Qwen2ForCausalLM" and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _required_cache_length(model: EaModel, prompt_length: int, max_new_tokens: int) -> int:
    total_tokens = int(getattr(getattr(model, "ea_layer", None), "total_tokens", 0))
    return int(prompt_length + max_new_tokens + max(total_tokens + 16, 64))


def _strip_special_tokens(text: str, tokenizer) -> str:
    cleaned = text
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                if special_tok:
                    cleaned = cleaned.replace(special_tok, "")
        else:
            if special_token:
                cleaned = cleaned.replace(special_token, "")
    return cleaned.strip()


def _decode_generated_text(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    generated_ids = output_ids[0, prompt_len:]
    decoded = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
    return _strip_special_tokens(decoded, tokenizer)


def _token_to_text(tokenizer, token_id: Optional[int]) -> Optional[str]:
    if token_id is None or token_id < 0:
        return None
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def _first_generated_token_mismatch(
    greedy_ids: torch.Tensor,
    eagle_ids: torch.Tensor,
) -> Optional[int]:
    greedy_len = int(greedy_ids.numel())
    eagle_len = int(eagle_ids.numel())
    shared = min(greedy_len, eagle_len)
    for idx in range(shared):
        if int(greedy_ids[idx].item()) != int(eagle_ids[idx].item()):
            return idx
    if greedy_len != eagle_len:
        return shared
    return None


@torch.no_grad()
def _strict_target_greedy(
    model: EaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    max_length = _required_cache_length(model, int(input_ids.size(1)), max_new_tokens)
    prompt_len = int(input_ids.size(1))
    eos_id = model.tokenizer.eos_token_id
    total_tokens = int(getattr(getattr(model, "ea_layer", None), "total_tokens", 0))
    loop_limit = max_length - total_tokens - 10

    past_key_values, _, _ = initialize_past_key_values(model.base_model, max_length=max_length)
    reset_tree_mode(model)

    _synchronize(device)
    prefill_start = time.perf_counter()
    outputs = model.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
    _synchronize(device)
    prefill_wall_time = time.perf_counter() - prefill_start

    sequence_ids = input_ids
    token_trace: List[Dict[str, Any]] = []
    decode_wall_time = 0.0
    token_count = 0

    for step_idx in range(loop_limit):
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        token_id = int(next_token.item())
        token_trace.append(
            {
                "step_index": step_idx,
                "token_id": token_id,
            }
        )

        _synchronize(device)
        step_start = time.perf_counter()
        outputs = model.base_model(
            next_token,
            use_cache=True,
            past_key_values=past_key_values,
        )
        _synchronize(device)
        decode_wall_time += time.perf_counter() - step_start

        sequence_ids = torch.cat([sequence_ids, next_token], dim=-1)
        token_count += 1

        if eos_id is not None and eos_id in sequence_ids[0, prompt_len:].tolist():
            break
        if token_count > max_new_tokens:
            break
        if int(sequence_ids.shape[1]) > loop_limit:
            break

    reset_tree_mode(model)
    stats = {
        "generated_tokens": max(0, int(sequence_ids.size(1) - prompt_len)),
        "decode_steps": token_count,
        "prefill_wall_time": prefill_wall_time,
        "decode_wall_time": decode_wall_time,
        "wall_time": prefill_wall_time + decode_wall_time,
    }
    return sequence_ids, stats, token_trace


@torch.no_grad()
def _native_eagenerate(
    model: EaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    prompt_len = int(input_ids.size(1))
    max_length = _required_cache_length(model, prompt_len, max_new_tokens)
    audit_state: Dict[str, Any] = {}

    _synchronize(device)
    start_time = time.perf_counter()
    output_ids, _new_token, idx = model.eagenerate(
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        log=True,
        audit_state=audit_state,
    )
    _synchronize(device)
    wall_time = time.perf_counter() - start_time

    generated_tokens = max(0, int(output_ids.size(1) - prompt_len))
    decode_steps = max(0, int(idx) + 1) if generated_tokens > 0 else 0
    stats = {
        "generated_tokens": generated_tokens,
        "decode_steps": decode_steps,
        "prefill_wall_time": 0.0,
        "decode_wall_time": wall_time,
        "wall_time": wall_time,
    }
    return output_ids, stats, list(audit_state.get("records", []))


def _locate_eagle_step_for_token(
    eagle_records: Sequence[Dict[str, Any]],
    generated_token_index: int,
) -> Optional[Dict[str, Any]]:
    offset = 0
    for record in eagle_records:
        accepted_ids = [int(x) for x in record.get("accepted_token_ids", [])]
        accepted_count = len(accepted_ids)
        if generated_token_index < offset + accepted_count:
            return {
                "step_index": int(record.get("step_index", -1)),
                "token_span_start": offset,
                "token_span_end": offset + accepted_count - 1,
                "within_step_offset": generated_token_index - offset,
                "record": record,
            }
        offset += accepted_count
    return None


def _format_step_tokens(tokenizer, token_ids: Sequence[int]) -> str:
    return " | ".join(
        f"{token_id}:{_token_to_text(tokenizer, token_id)}"
        for token_id in token_ids
    )


@torch.inference_mode()
def run_medqa_lossless_audit(
    *,
    base_model_path: str,
    ea_model_path: str,
    split: str,
    num_samples: int,
    sample_offset: int,
    max_new_tokens: int,
    warmup_examples: int,
    total_token: int,
    depth: int,
    top_k: int,
    use_eagle3: bool,
    max_reports: int,
    output_dir: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    items, resolved_split = _load_medqa_samples(split, num_samples, sample_offset)
    if not items:
        raise ValueError("Resolved MedQA sample set is empty.")

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

    print("Loading EAGLE model...")
    model = EaModel.from_pretrained(**load_kwargs)
    model.eval()
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _module_device(model.base_model)

    warmup_count = min(max(0, warmup_examples), len(items))
    if warmup_count > 0 and verbose:
        print(f"Warming up on {warmup_count} example(s)...")
    for item in items[:warmup_count]:
        prompt_ids = _build_chat_input_ids(tokenizer, item["prompt"], device)
        _ = _strict_target_greedy(model, prompt_ids, max_new_tokens, device)
        _ = _native_eagenerate(model, prompt_ids, max_new_tokens, device)
    if warmup_count > 0 and verbose:
        print("Warmup done.")

    exact_match_examples = 0
    length_match_examples = 0
    same_label_examples = 0
    greedy_acc = 0.0
    eagle_acc = 0.0
    total_greedy_tokens = 0
    total_eagle_tokens = 0
    total_greedy_time = 0.0
    total_eagle_time = 0.0
    mismatch_reports: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        prompt_ids = _build_chat_input_ids(tokenizer, item["prompt"], device)
        prompt_len = int(prompt_ids.size(1))

        greedy_output, greedy_stats, greedy_trace = _strict_target_greedy(
            model,
            prompt_ids,
            max_new_tokens,
            device,
        )
        eagle_output, eagle_stats, eagle_records = _native_eagenerate(
            model,
            prompt_ids,
            max_new_tokens,
            device,
        )

        greedy_generated = greedy_output[0, prompt_len:]
        eagle_generated = eagle_output[0, prompt_len:]
        greedy_len = int(greedy_generated.numel())
        eagle_len = int(eagle_generated.numel())
        total_greedy_tokens += greedy_len
        total_eagle_tokens += eagle_len
        total_greedy_time += float(greedy_stats["wall_time"])
        total_eagle_time += float(eagle_stats["wall_time"])

        if torch.equal(greedy_generated, eagle_generated):
            exact_match_examples += 1
        if greedy_len == eagle_len:
            length_match_examples += 1

        greedy_text = _decode_generated_text(tokenizer, greedy_output, prompt_len)
        eagle_text = _decode_generated_text(tokenizer, eagle_output, prompt_len)
        greedy_label = normalize_medqa_label(greedy_text)
        eagle_label = normalize_medqa_label(eagle_text)
        if greedy_label == eagle_label:
            same_label_examples += 1
        greedy_acc += float(greedy_label == item["label"])
        eagle_acc += float(eagle_label == item["label"])

        mismatch_pos = _first_generated_token_mismatch(greedy_generated, eagle_generated)
        if mismatch_pos is not None and len(mismatch_reports) < max_reports:
            greedy_token_id = int(greedy_generated[mismatch_pos].item()) if mismatch_pos < greedy_len else None
            eagle_token_id = int(eagle_generated[mismatch_pos].item()) if mismatch_pos < eagle_len else None
            eagle_step_info = _locate_eagle_step_for_token(eagle_records, mismatch_pos)
            report: Dict[str, Any] = {
                "example_index": idx + 1,
                "question_id": item["question_id"],
                "reference_label": item["label"],
                "first_mismatch_token_index": mismatch_pos,
                "greedy_token_id": greedy_token_id,
                "greedy_token_text": _token_to_text(tokenizer, greedy_token_id),
                "eagle_token_id": eagle_token_id,
                "eagle_token_text": _token_to_text(tokenizer, eagle_token_id),
                "greedy_tokens": greedy_len,
                "eagle_tokens": eagle_len,
                "greedy_label": greedy_label,
                "eagle_label": eagle_label,
                "greedy_text": greedy_text,
                "eagle_text": eagle_text,
                "prompt": item["prompt"],
                "greedy_trace_prefix": greedy_trace[max(0, mismatch_pos - 2): mismatch_pos + 3],
            }
            if eagle_step_info is not None:
                step_record = eagle_step_info["record"]
                accepted_token_ids = [int(x) for x in step_record.get("accepted_token_ids", [])]
                candidate_token_ids = [int(x) for x in step_record.get("candidate_token_ids", [])]
                report["eagle_step_audit"] = {
                    "step_index": eagle_step_info["step_index"],
                    "token_span_start": eagle_step_info["token_span_start"],
                    "token_span_end": eagle_step_info["token_span_end"],
                    "within_step_offset": eagle_step_info["within_step_offset"],
                    "accepted_token_ids": accepted_token_ids,
                    "accepted_token_text": _format_step_tokens(tokenizer, accepted_token_ids),
                    "candidate_token_ids": candidate_token_ids,
                    "candidate_token_text": _format_step_tokens(tokenizer, candidate_token_ids),
                    "target_top1_token_id": step_record.get("target_top1_token_id"),
                    "target_top1_token_text": _token_to_text(tokenizer, step_record.get("target_top1_token_id")),
                    "first_rejected_token_id": step_record.get("first_rejected_token_id"),
                    "first_rejected_token_text": _token_to_text(tokenizer, step_record.get("first_rejected_token_id")),
                    "accept_length": step_record.get("accept_length"),
                    "accepted_token_count": step_record.get("accepted_token_count"),
                    "strict_reject": step_record.get("strict_reject"),
                    "candidate_rank": step_record.get("candidate_rank"),
                    "verification_mode": step_record.get("verification_mode"),
                    "target_topk_token_ids": step_record.get("target_topk_token_ids"),
                    "target_topk_probs": step_record.get("target_topk_probs"),
                }
            mismatch_reports.append(report)

        if verbose:
            status = "exact" if mismatch_pos is None else "mismatch"
            print(
                f"[lossless {idx + 1}/{len(items)}] {status} "
                f"greedy={greedy_len}tok eagle={eagle_len}tok "
                f"labels={greedy_label}/{eagle_label}"
            )

    n = float(len(items))
    summary: Dict[str, Any] = {
        "warmup_examples": warmup_count,
        "split": resolved_split,
        "num_examples": len(items),
        "exact_match_examples": exact_match_examples,
        "exact_match_rate": exact_match_examples / n,
        "length_match_examples": length_match_examples,
        "length_match_rate": length_match_examples / n,
        "same_label_examples": same_label_examples,
        "same_label_rate": same_label_examples / n,
        "greedy_accuracy": greedy_acc / n,
        "eagle_accuracy": eagle_acc / n,
        "avg_greedy_tokens": float(total_greedy_tokens) / n,
        "avg_eagle_tokens": float(total_eagle_tokens) / n,
        "greedy_speed": _tokens_per_second(total_greedy_tokens, total_greedy_time),
        "eagle_speed": _tokens_per_second(total_eagle_tokens, total_eagle_time),
        "mismatch_reports": mismatch_reports,
        "pass": exact_match_examples == len(items),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "summary.json")
        mismatches_path = os.path.join(output_dir, "mismatch_reports.jsonl")
        with open(summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, ensure_ascii=True, indent=2)
        with open(mismatches_path, "w", encoding="utf-8") as fout:
            for report in mismatch_reports:
                fout.write(json.dumps(report, ensure_ascii=True) + "\n")
        summary["summary_path"] = summary_path
        summary["mismatches_path"] = mismatches_path

    print("=== MedQA Lossless Audit Summary ===")
    print(f"Warmup examples: {summary['warmup_examples']}")
    print(f"Split: {summary['split']}")
    print(f"Exact token match: {summary['exact_match_rate']:.3f} ({summary['exact_match_examples']}/{summary['num_examples']})")
    print(f"Length match: {summary['length_match_rate']:.3f} ({summary['length_match_examples']}/{summary['num_examples']})")
    print(f"Same normalized label: {summary['same_label_rate']:.3f} ({summary['same_label_examples']}/{summary['num_examples']})")
    print(f"Greedy/EAGLE accuracy: {summary['greedy_accuracy']:.3f} / {summary['eagle_accuracy']:.3f}")
    print(f"Greedy/EAGLE generated tokens: {summary['avg_greedy_tokens']:.2f} / {summary['avg_eagle_tokens']:.2f}")
    print(f"Greedy/EAGLE speed: {summary['greedy_speed']:.2f} / {summary['eagle_speed']:.2f} tok/s")
    for report in mismatch_reports:
        print(
            f"Mismatch example {report['example_index']}: "
            f"first_token={report['first_mismatch_token_index']} "
            f"greedy_id={report['greedy_token_id']} eagle_id={report['eagle_token_id']}"
        )
        print(f"Reference label: {report['reference_label']}")
        print(f"Greedy label/text: {report['greedy_label']} | {report['greedy_text']}")
        print(f"EAGLE label/text: {report['eagle_label']} | {report['eagle_text']}")
        eagle_step_audit = report.get("eagle_step_audit")
        if eagle_step_audit:
            print(
                f"EAGLE step audit: step={eagle_step_audit['step_index']} "
                f"span=[{eagle_step_audit['token_span_start']},{eagle_step_audit['token_span_end']}] "
                f"within_step={eagle_step_audit['within_step_offset']} "
                f"accept_count={eagle_step_audit['accepted_token_count']}"
            )
            print(
                "EAGLE accepted tokens: "
                f"{eagle_step_audit['accepted_token_text']}"
            )
            print(
                "EAGLE target top1 / first rejected: "
                f"{eagle_step_audit['target_top1_token_id']}:{eagle_step_audit['target_top1_token_text']} / "
                f"{eagle_step_audit['first_rejected_token_id']}:{eagle_step_audit['first_rejected_token_text']}"
            )
    if output_dir:
        print(f"Summary JSON: {summary['summary_path']}")
        print(f"Mismatch JSONL: {summary['mismatches_path']}")
    print(f"Pass: {summary['pass']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure-EAGLE MedQA lossless audit")
    parser.add_argument("--base-model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ea-model-path", type=str, default="thoughtworks/Qwen2.5-7B-Instruct-Eagle3")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-examples", type=int, default=2)
    parser.add_argument("--total-token", type=int, default=32)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--use-eagle3", action="store_true")
    parser.add_argument("--disable-eagle3", action="store_true")
    parser.add_argument("--max-reports", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    use_eagle3 = True
    if args.disable_eagle3:
        use_eagle3 = False
    elif args.use_eagle3:
        use_eagle3 = True

    result = run_medqa_lossless_audit(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        split=args.split,
        num_samples=args.num_samples,
        sample_offset=args.sample_offset,
        max_new_tokens=args.max_new_tokens,
        warmup_examples=args.warmup_examples,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        use_eagle3=use_eagle3,
        max_reports=args.max_reports,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    sys.exit(0 if result["pass"] else 1)
