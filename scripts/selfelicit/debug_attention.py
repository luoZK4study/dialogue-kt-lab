"""Debug SELFELICIT attention extraction."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt
from dialogue_kt.kt_data_loading import apply_annotations
from dialogue_kt.selfelicit import get_evidence_layer_range
import pandas as pd, ast

device = torch.device("cuda:0")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
model.eval()

# Load sample
df = pd.read_csv("data/annotated/mathdial_train_atc.csv")
sample = df.iloc[0].copy()
sample["dialogue"] = ast.literal_eval(sample["dialogue"])
sample["meta_data"] = ast.literal_eval(sample["meta_data"])
sample["annotation"] = ast.literal_eval(sample["annotation"])

class Args:
    dataset = "mathdial"
    prompt_inc_labels = False

dialogue = apply_annotations(sample)
first_turn = None
for t in dialogue:
    if t["correct"] is not None:
        first_turn = t
        break

system = kt_system_prompt(Args())
user = kt_user_prompt(sample, dialogue, first_turn["turn"], None, Args())
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False,
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"Prompt: {len(prompt)} chars, {input_ids.shape[1]} tokens")

# Run forward with attention
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True, attention_mask=torch.ones_like(input_ids))

print(f"Number of attention layers: {len(outputs.attentions)}")
print(f"Layer 0 attention shape: {outputs.attentions[0].shape if outputs.attentions[0] is not None else 'None'}")
print(f"Layer 14 shape: {outputs.attentions[14].shape if outputs.attentions[14] is not None else 'None'}")

# Check: is any attention None?
none_count = sum(1 for a in outputs.attentions if a is None)
print(f"None attention layers: {none_count}/{len(outputs.attentions)}")

# Check: can we extract meaningful attention?
layer_range = get_evidence_layer_range(len(outputs.attentions))
print(f"Evidence layer range: {layer_range}")

# Check what context span the per-token approach finds
ids_list = input_ids[0].tolist()
cum_text = ""
token_char_starts = []
for tid in ids_list:
    token_char_starts.append(len(cum_text))
    cum_text += tokenizer.decode([tid])

b_char = cum_text.find("[BEGIN DIALOGUE]")
e_char = cum_text.find("[END DIALOGUE]")
print(f"Markers at chars: begin={b_char}, end={e_char}")

if b_char >= 0:
    cs = 0
    for ti, ts in enumerate(token_char_starts):
        if ts <= b_char:
            cs = ti
    print(f"context_start token index: {cs}")
    print(f"Token at cs: '{tokenizer.decode([ids_list[cs]])}'")

    # Try extract attention from last token to context
    att = outputs.attentions[14][0].mean(dim=0)  # avg across heads
    print(f"Attention matrix shape: {att.shape}")
    to_context = att[-1, cs:cs+50]
    print(f"Attention to context tokens [cs:cs+50]: min={to_context.min():.6f} max={to_context.max():.6f} mean={to_context.mean():.6f}")
    if to_context.sum() > 0:
        print("SUCCESS: Valid attention to context found!")
    else:
        print("FAIL: Zero attention to context")
