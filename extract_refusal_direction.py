#!/usr/bin/env python3
"""
extract_refusal_direction.py — Compute the refusal direction (Arditi et al., 2024).

d_refusal[l] = mean_x in HARMFUL  h_l(x)  -  mean_x in HARMLESS  h_l(x)

where h_l(x) is the residual-stream activation at layer l. By default we take the
activation at the LAST prompt token (the position the model is about to generate from),
which is the canonical extraction point in Arditi et al. Optionally (--aggregate response)
we instead average over generated response tokens, to match exactly how generate_vec.py
builds tau_A / tau_U (useful if you want d_refusal measured in the identical regime as
your persona vectors).

LAYER ALIGNMENT
---------------
We use output_hidden_states=True and stack the FULL hidden_states tuple
(index 0 = embeddings, index l = output of block l), i.e. the same post-residual
convention generate_vec.py uses. The result is [n_hidden_states, d], matching the layer
count/indexing of your tau vectors, so metrics.py's per-layer cross-trait cosine lines up
1:1 with no re-indexing.

OUTPUT
------
Saved as a drop-in CAA-style vector:
    <save_dir>/refusal_response_avg_diff.pt        # shape [n_layers, d]
so you can immediately run, e.g.:

    python metrics.py --vec_dir <vec_dir>
        --ai_trait <fac>_tauA --user_trait <fac>_tauU --joint_trait <fac>_joint
        --cross_trait_vec_dir <save_dir>
        --cross_trait_left <fac>_joint --cross_trait_right refusal
        --output_dir output/metrics --output_prefix <fac>_vs_refusal

To get the full cosine battery (cos(R,d_ref), cos(R_perp,d_ref), cos(tau_A,d_ref),
cos(tau_U,d_ref)), the simplest path is the small companion snippet in the runbook:
load refusal_response_avg_diff.pt + the tau vectors and compute per-layer cosines
directly (R and R_perp need the Gram-Schmidt split, which metrics.py already exposes
via its decomposition; the runbook gives a 20-line script).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: example metrics.py invocation is in FACTORIAL_AND_REFUSAL_RUNBOOK.md (kept out of
# this docstring to avoid backslash-continuation escape warnings).


def build_prompt(tokenizer, instruction):
    msgs = [{"role": "user", "content": instruction}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def last_token_hidden(model, tokenizer, instruction, device):
    """Stack full hidden_states at the last prompt token -> [n_hidden_states, d]."""
    prompt = build_prompt(tokenizer, instruction)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # tuple len = n_layers+1, each [1, seq, d]
    stacked = torch.stack([h[0, -1, :] for h in hs], dim=0)  # [n_hidden_states, d]
    return stacked.float().cpu()


@torch.no_grad()
def response_avg_hidden(model, tokenizer, instruction, device, max_new_tokens=64):
    """Average full hidden_states over generated response tokens -> [n_hidden_states, d].
    Matches generate_vec.py's response-averaged regime."""
    prompt = build_prompt(tokenizer, instruction)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    full = gen[0]
    resp_ids = full[enc["input_ids"].shape[1]:]
    if resp_ids.numel() == 0:
        return last_token_hidden(model, tokenizer, instruction, device)
    full_in = full.unsqueeze(0)
    out = model(full_in, output_hidden_states=True)
    start = enc["input_ids"].shape[1]
    hs = out.hidden_states
    stacked = torch.stack([h[0, start:, :].mean(dim=0) for h in hs], dim=0)
    return stacked.float().cpu()


def main():
    p = argparse.ArgumentParser(description="Extract the refusal direction via difference-in-means.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--refusal_json", default="persona_steering/data_generation/refusal/refusal.json")
    p.add_argument("--save_dir", required=True, help="Where to write refusal_response_avg_diff.pt")
    p.add_argument("--aggregate", choices=["last_token", "response"], default="last_token")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Only for --aggregate response.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Cap pairs for a quick run (0 = all).")
    args = p.parse_args()

    data = json.loads(Path(args.refusal_json).read_text())
    harmful, harmless = data["harmful"], data["harmless"]
    if args.limit > 0:
        harmful, harmless = harmful[:args.limit], harmless[:args.limit]
    print(f"  {len(harmful)} harmful / {len(harmless)} harmless  | aggregate={args.aggregate}")

    dtype = torch.float16 if "cuda" in args.device else torch.float32
    kw = dict(torch_dtype=dtype, trust_remote_code=True)
    if args.load_in_8bit:
        kw.update(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kw)
    if not args.load_in_8bit:
        model.to(args.device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    fn = last_token_hidden if args.aggregate == "last_token" else \
        (lambda m, t, ins, dev: response_avg_hidden(m, t, ins, dev, args.max_new_tokens))

    def mean_over(instructions, label):
        acc = None
        for i, ins in enumerate(instructions):
            v = fn(model, tok, ins, args.device)
            acc = v if acc is None else acc + v
            if (i + 1) % 16 == 0:
                print(f"    {label}: {i + 1}/{len(instructions)}")
        return acc / len(instructions)

    print("  Harmful pass..."); mu_harm = mean_over(harmful, "harmful")
    print("  Harmless pass..."); mu_harmless = mean_over(harmless, "harmless")

    d_refusal = (mu_harm - mu_harmless)  # [n_hidden_states, d]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "refusal_response_avg_diff.pt"
    torch.save(d_refusal, str(out))
    norms = torch.linalg.vector_norm(d_refusal, dim=-1)
    print(f"  Saved {out}  shape {tuple(d_refusal.shape)}")
    print(f"  Per-layer ||d_refusal||: min={norms.min():.3f} max={norms.max():.3f} "
          f"argmax_layer={int(norms.argmax())}  (peak layer ~ best for ablation)")


if __name__ == "__main__":
    main()
