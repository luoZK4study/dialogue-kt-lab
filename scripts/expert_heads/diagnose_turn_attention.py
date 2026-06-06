"""
Expert Heads Diagnosis for Dialogue-KT (Qwen3-1.7B)

Checks whether deep-layer attention heads in Qwen3-1.7B can stably
identify evidence-relevant historical dialogue turns.

Based on: Wu et al., "Expert Heads: Robust Evidence Identification
for Large Language Models", ICLR 2026.
"""
import argparse, json, os, sys
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dialogue_kt.models.lm import get_model
from dialogue_kt.data_loading import load_annotated_data, get_default_fold
from dialogue_kt.kt_data_loading import apply_annotations
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, get_true_false_tokens
from dialogue_kt.utils import device, initialize_seeds


def get_turn_boundaries_simple(input_ids, tokenizer, n_turns):
    """Find turn boundaries by searching for 'Turn N:' text patterns."""
    boundaries = []
    for turn_num in range(n_turns + 1):
        search_text = f"Turn {turn_num}"
        for pos in range(len(input_ids)):
            chunk = tokenizer.decode(input_ids[max(0, pos-2):pos+len(search_text)+5], skip_special_tokens=True)
            if search_text in chunk and f"Turn {turn_num}" in chunk:
                if not boundaries or pos - boundaries[-1] > 5:
                    boundaries.append(pos)
                    break
    return sorted(set(boundaries))


def compute_turn_attention(attentions, input_ids, turn_boundaries, layer_indices, head_indices):
    """Compute per-head attention to each turn from last few positions."""
    n_turns = len(turn_boundaries)
    results = {}
    seq_len = len(input_ids)

    for layer_idx in layer_indices:
        if layer_idx >= len(attentions):
            continue
        layer_attn = attentions[layer_idx]
        if len(layer_attn.shape) == 4:
            layer_attn = layer_attn[0]  # (heads, seq, seq)
        n_heads = layer_attn.shape[0]

        for head_idx in head_indices:
            if head_idx >= n_heads:
                continue
            head_attn = layer_attn[head_idx]  # (seq, seq)

            # Average attention from last 5 positions as query
            query_start = max(0, seq_len - 6)
            query_attn = head_attn[query_start:seq_len].mean(dim=0)

            turn_scores = np.zeros(n_turns)
            for i in range(n_turns):
                start = turn_boundaries[i]
                end = turn_boundaries[i + 1] if i + 1 < n_turns else seq_len
                if start < seq_len:
                    end = min(end, seq_len)
                    turn_scores[i] = query_attn[start:end].sum().item()

            total = turn_scores.sum()
            if total > 0:
                turn_scores = turn_scores / total

            key = f"L{layer_idx}_H{head_idx}"
            results[key] = turn_scores

    return results


def compute_evidence_quality(turn_attentions, labels, n_evidence=3):
    """Compute how well each head ranks recent 'evidence' turns higher."""
    metrics = {}
    n_turns = len(labels)
    if n_turns <= n_evidence:
        return metrics

    evidence_set = set(range(n_turns - n_evidence, n_turns))
    non_evidence_set = set(range(n_turns)) - evidence_set

    for key, scores in turn_attentions.items():
        if len(scores) != n_turns:
            continue
        ev_attn = scores[list(evidence_set)].sum()
        nev_attn = scores[list(non_evidence_set)].sum()
        concentration = ev_attn / (nev_attn + 1e-8)
        hits = int(scores.argmax() in evidence_set)
        metrics[key] = {
            "concentration": float(concentration),
            "hits_evidence": hits,
            "max_turn": int(scores.argmax()),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", default="lmkt_qwen3_1.7b")
    parser.add_argument("--n_samples", type=int, default=15)
    parser.add_argument("--n_evidence", type=int, default=3)
    parser.add_argument("--output", default="results/expert_heads/attention_diagnosis.json")
    args = parser.parse_args()

    initialize_seeds(221)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading model...")
    model, tokenizer = get_model(args.base_model, True, model_name=args.checkpoint, quantize=False)
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"Model: {n_layers} layers, {n_heads} heads/layer")

    from types import SimpleNamespace
    ns = SimpleNamespace(
        dataset="mathdial", tag_src="atc", typical_cutoff=1,
        split_by_subject=False, debug=False,
        model_type="lmkt", batch_size=1,
        kt_method="base", bayes_evidence=None, informativeness=None,
        selfelicit=None, prompt_inc_labels=False, pack_kcs=True
    )
    _, val_df, _ = load_annotated_data(ns, get_default_fold(ns))
    val_df = val_df[:args.n_samples]
    print(f"Analyzing {len(val_df)} samples...")

    # Deep layers only (last 30%)
    deep_start = int(n_layers * 0.7)
    layer_indices = list(range(deep_start, n_layers))
    head_indices = list(range(min(n_heads, 16)))
    print(f"Focusing on layers {layer_indices}, heads {head_indices}")

    all_head_metrics = defaultdict(list)
    processed = 0

    for idx, sample in tqdm(val_df.iterrows(), total=len(val_df)):
        dialogue = apply_annotations(sample)
        if not dialogue:
            continue

        for turn in dialogue:
            if turn["correct"] is None or turn["turn"] < 3:
                continue

            user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, ns)
            system_content = kt_system_prompt(ns)

            # Build prompt with 1 KC for speed
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ], tokenize=False)

            kc = turn["kcs"][0]
            kc_template = tokenizer.apply_chat_template([
                {"role": "user", "content": "KC_PLACEHOLDER"},
                {"role": "assistant", "content": "\n"}
            ], tokenize=False)
            placeholder_idx = kc_template.find("KC_PLACEHOLDER")
            header_end_idx = kc_template.rfind("\n", 0, placeholder_idx)
            kc_cont = tokenizer.apply_chat_template([
                {"role": "user", "content": kc},
                {"role": "assistant", "content": "\n"}
            ], tokenize=False)
            kc_cont = " " + kc_cont[header_end_idx + 1:]
            full_prompt = prompt + kc_cont

            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            # Forward with attention
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Find turn boundaries (at most turn["turn"] + 1)
            boundaries = get_turn_boundaries_simple(inputs.input_ids[0], tokenizer, turn["turn"])
            if len(boundaries) < 3:
                del outputs
                continue

            turn_attns = compute_turn_attention(
                outputs.attentions, inputs.input_ids[0], boundaries,
                layer_indices, head_indices
            )

            labels_arr = [t["correct"] for t in dialogue[:turn["turn"]] if t["correct"] is not None]
            metrics = compute_evidence_quality(turn_attns, labels_arr, args.n_evidence)

            for key, m in metrics.items():
                all_head_metrics[key].append(m)

            processed += 1
            del outputs

            if processed >= args.n_samples * 5:
                break

        if processed >= args.n_samples * 5:
            break

    print(f"\nProcessed {processed} turn-attention pairs")

    # Aggregate
    results = {}
    for head_key, measurements in all_head_metrics.items():
        if len(measurements) < 3:
            continue
        conc = np.mean([m["concentration"] for m in measurements])
        hit = np.mean([m["hits_evidence"] for m in measurements])
        results[head_key] = {
            "n": len(measurements),
            "avg_concentration": float(conc),
            "hit_rate": float(hit),
            "expert_candidate": bool(conc > 1.5 and hit > 0.5),
        }

    sorted_heads = sorted(results.items(), key=lambda x: x[1]["avg_concentration"], reverse=True)
    print("\n=== Top 10 Attention Heads ===")
    for key, s in sorted_heads[:10]:
        print(f"  {key}: conc={s['avg_concentration']:.3f}, hit={s['hit_rate']:.3f}, "
              f"n={s['n']}, expert={s['expert_candidate']}")

    n_experts = sum(1 for _, s in results.items() if s["expert_candidate"])
    concentrations = [s["avg_concentration"] for _, s in results.items()]
    print(f"\nConcentration: mean={np.mean(concentrations):.3f}, "
          f"max={np.max(concentrations):.3f}, min={np.min(concentrations):.3f}")
    print(f"Expert candidates: {n_experts}/{len(results)}")

    if n_experts >= 3:
        print("\n✅ Expert heads detected. Proceed to full implementation.")
        verdict = "PASS"
    elif n_experts >= 1:
        print("\n⚠️  Weak signal. Consider lightweight implementation.")
        verdict = "WEAK"
    else:
        print("\n❌ No expert heads. Skip implementation.")
        verdict = "FAIL"

    with open(args.output, "w") as f:
        json.dump({
            "model": args.checkpoint, "n_layers": n_layers, "n_heads": n_heads,
            "n_samples": len(val_df), "n_processed": processed,
            "verdict": verdict, "n_expert_candidates": n_experts,
            "head_results": {k: v for k, v in results.items()},
            "top_heads": [(k, v) for k, v in sorted_heads[:20]],
        }, f, indent=2)
    print(f"Saved to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
