"""Test finding dialogue span via token-level marker search."""
from transformers import AutoTokenizer
from dialogue_kt.prompting import get_dialogue_text, kt_system_prompt, kt_user_prompt
from dialogue_kt.kt_data_loading import apply_annotations
import pandas as pd
import ast

df = pd.read_csv("data/annotated/mathdial_train_atc.csv")
for idx in range(min(5, len(df))):
    sample = df.iloc[idx].copy()
    sample["dialogue"] = ast.literal_eval(sample["dialogue"]) if isinstance(sample["dialogue"], str) else sample["dialogue"]
    sample["meta_data"] = ast.literal_eval(sample["meta_data"]) if isinstance(sample["meta_data"], str) else sample["meta_data"]
    sample["annotation"] = ast.literal_eval(sample["annotation"]) if isinstance(sample["annotation"], str) else sample["annotation"]

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token

    class Args:
        dataset = "mathdial"
        prompt_inc_labels = False
    args = Args()

    dialogue = apply_annotations(sample)
    first_turn = None
    for turn in dialogue:
        if turn["correct"] is not None:
            first_turn = turn
            break
    if not first_turn:
        continue

    system = kt_system_prompt(args)
    user = kt_user_prompt(sample, dialogue, first_turn["turn"], None, args)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
    )
    print(f"\n=== Sample {idx}, turn {first_turn['turn']} ===")
    print(f"  prompt_ids type: {type(prompt_ids)}")
    if isinstance(prompt_ids, list):
        print(f"  prompt_ids len: {len(prompt_ids)}")
        if len(prompt_ids) > 0:
            print(f"  prompt_ids[0] type: {type(prompt_ids[0])}")
            print(f"  prompt_ids[0:3]: {prompt_ids[:3]}")
    elif hasattr(prompt_ids, 'shape'):
        print(f"  prompt_ids shape: {prompt_ids.shape}")
        print(f"  prompt_ids[0][:3]: {prompt_ids[0][:3]}")
    else:
        print(f"  prompt_ids: {str(prompt_ids)[:200]}")
    break  # Just check first sample

    # Find [BEGIN DIALOGUE] marker as tokens
    begin_marker = "[BEGIN DIALOGUE]"
    begin_ids = tokenizer.encode(begin_marker, add_special_tokens=False)
    end_marker = "[END DIALOGUE]"
    end_ids = tokenizer.encode(end_marker, add_special_tokens=False)

    print(f"  '[BEGIN DIALOGUE]' token IDs: {begin_ids}")
    print(f"  '[END DIALOGUE]' token IDs: {end_ids}")

    # Search for BEGIN marker in prompt
    b_start = -1
    for i in range(len(prompt_ids) - len(begin_ids) + 1):
        if prompt_ids[i : i + len(begin_ids)] == begin_ids:
            b_start = i
            break

    # Search for END marker
    e_end = -1
    for i in range(len(prompt_ids) - len(end_ids) + 1):
        if prompt_ids[i : i + len(end_ids)] == end_ids:
            e_end = i + len(end_ids)

    if b_start >= 0 and e_end >= 0 and e_end > b_start:
        print(f"  SUCCESS: dialogue span = [{b_start}, {e_end}) ({e_end - b_start} tokens)")
    else:
        print(f"  FAILED: begin={b_start}, end={e_end}")
        # Try finding in decoded text
        decoded = tokenizer.decode(prompt_ids)
        bp = decoded.find(begin_marker)
        ep = decoded.find(end_marker)
        print(f"  Text positions: begin={bp}, end={ep}")
        # Count tokens before begin
        if bp > 0:
            before_text = decoded[:bp]
            before_tokens = len(tokenizer.encode(before_text, add_special_tokens=False))
            # Also count with special tokens
            before_tokens_sp = len(tokenizer.encode(before_text, add_special_tokens=True))
            print(f"  Tokens before begin (no special): {before_tokens}")
            print(f"  Tokens before begin (with special): {before_tokens_sp}")
            print(f"  Prompt has {len(prompt_ids)} tokens total")
