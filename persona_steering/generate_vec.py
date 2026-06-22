from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# ── head-ablation hooks (mirror ablation_study.py; LAYER = block index) ──────────
def parse_head_spec(spec: str) -> dict:
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        L, h = tok.split(":")
        out.setdefault(int(L), []).append(int(h))
    return out


def make_head_ablation_prehook(head_ids, head_dim, coeff):
    """forward_pre_hook on self_attn.o_proj: null-ablate listed heads in the
    o_proj input.  z'_h = (1 - coeff) * z_h   (coeff=1 zeros the head)."""
    def prehook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        x = x.clone()
        for h in head_ids:
            sl = slice(h * head_dim, (h + 1) * head_dim)
            x[..., sl] = (1.0 - coeff) * x[..., sl]
        return (x,) + tuple(inputs[1:])
    return prehook


def register_head_ablation(model, ablate_heads, coeff):
    """Register o_proj pre-hooks for all specified heads. Returns handles."""
    spec = parse_head_spec(ablate_heads)
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
    layers = model.model.layers
    handles = []
    for L, hds in spec.items():
        if not (0 <= L < len(layers)):
            print(f"[skip head layer {L} OOB]")
            continue
        ph = make_head_ablation_prehook(hds, head_dim, coeff)
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(ph))
    print(f"Head ablation active: {spec}  (coeff={coeff}, null) -> {len(handles)} hook(s)")
    return handles

    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg

import pandas as pd
import os

"""def get_persona_effective(pos_path, neg_path, trait, threshold=50, coherence_threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= coherence_threshold)
        & (persona_neg["coherence"] >= coherence_threshold)
    )"""
def get_persona_effective(pos_path, neg_path, trait, threshold=50, coherence_threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    if threshold <= 0 and coherence_threshold <= 0:
        # Factorial passthrough: keep ALL rows in original order so every cell retains
        # identical question×instruction×sample rows (required for clean cancellation
        # and for per-prompt alignment across pp/pn/np/nn).
        if len(persona_pos) != len(persona_neg):
            raise ValueError(f"pos ({len(persona_pos)}) vs neg ({len(persona_neg)}) "
                             f"row mismatch; factorial alignment broken.")
        mask = pd.Series(True, index=persona_pos.index)
    else:
        mask = (
            (persona_pos[trait] >= threshold)
            & (persona_neg[trait] < 100 - threshold)
            & (persona_pos["coherence"] >= coherence_threshold)
            & (persona_neg["coherence"] >= coherence_threshold)
        )
    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()    
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(
    model_name,
    pos_path,
    neg_path,
    trait,
    save_dir,
    threshold=50,
    coherence_threshold=50,
    max_samples=0,
    load_in_8bit=False,
    save_activations=False,
    ablate_heads="",
    head_coeff=1.0,
):
    model_kwargs = {"device_map": "auto"}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Head ablation: register o_proj pre-hooks so ALL extraction forward passes
    # (pos and neg) see the ablated model. The resulting *_response_avg_diff.pt /
    # activation matrices are the ablated counterparts, drop-in for the unchanged
    # factorial_residue -> metrics -> make_plots chain.
    abl_handles = []
    if ablate_heads:
        abl_handles = register_head_ablation(model, ablate_heads, head_coeff)

    print("######### ", f"{threshold=}", f"{coherence_threshold=}", f"{max_samples=}", f"{load_in_8bit=}")

    persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(
        pos_path, neg_path, trait, threshold, coherence_threshold
    )

    if len(persona_pos_effective_prompts) == 0 or len(persona_neg_effective_prompts) == 0:
        raise ValueError(
            f"No effective rows found for trait={trait}. "
            f"Try lowering --threshold and/or --coherence_threshold."
        )

    if max_samples and max_samples > 0:
        persona_pos_effective_prompts = persona_pos_effective_prompts[:max_samples]
        persona_neg_effective_prompts = persona_neg_effective_prompts[:max_samples]
        persona_pos_effective_responses = persona_pos_effective_responses[:max_samples]
        persona_neg_effective_responses = persona_neg_effective_responses[:max_samples]

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)
    


    persona_effective_prompt_avg_diff = torch.stack([persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_avg["pos"]))], dim=0)
    persona_effective_response_avg_diff = torch.stack([persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float() for l in range(len(persona_effective_response_avg["pos"]))], dim=0)
    persona_effective_prompt_last_diff = torch.stack([persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float() for l in range(len(persona_effective_prompt_last["pos"]))], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")

    # ── Per-prompt activation matrices (for EffRank / per-prompt residue /
    #    bootstrap CIs / NCF permutation null). Shape [n_prompts, n_layers, d].
    #    factorial_residue.py --per_prompt consumes "<trait>_pos_activation_matrix.pt".
    #    The mean over dim=0 of the pos matrix minus the mean of the neg matrix
    #    reproduces *_response_avg_diff.pt exactly, so nothing else changes. ──
    if save_activations:
        n_layers = len(persona_effective_response_avg["pos"])
        pos_mat = torch.stack(
            [persona_effective_response_avg["pos"][l].float() for l in range(n_layers)], dim=1
        )  # [n_prompts, n_layers, d]
        neg_mat = torch.stack(
            [persona_effective_response_avg["neg"][l].float() for l in range(n_layers)], dim=1
        )
        torch.save(pos_mat, f"{save_dir}/{trait}_pos_activation_matrix.pt")
        torch.save(neg_mat, f"{save_dir}/{trait}_neg_activation_matrix.pt")
        # Also persist the prompt-last per-prompt matrices (cheap; useful for
        # last-token refusal-direction work without re-extraction).
        pos_last = torch.stack(
            [persona_effective_prompt_last["pos"][l].float() for l in range(n_layers)], dim=1
        )
        torch.save(pos_last, f"{save_dir}/{trait}_pos_prompt_last_matrix.pt")
        print(f"Per-prompt activation matrices saved: {pos_mat.shape[0]} prompts, "
              f"{n_layers} layers  ->  {trait}_pos/neg_activation_matrix.pt")

    print(f"Persona vectors saved to {save_dir}")
    for h in abl_handles:
        h.remove()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--coherence_threshold", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--save_activations", action="store_true",
                        help="Persist per-prompt activation matrices [n_prompts,n_layers,d] "
                             "for EffRank / per-prompt residue / bootstrap / permutation null.")
    parser.add_argument("--ablate_heads", type=str, default="",
                        help="Head spec 'L:h,L:h' (block indices from localize_heads) to "
                             "null-ablate during extraction. Produces ablated activation "
                             "vectors/matrices, drop-in for factorial_residue/metrics/make_plots.")
    parser.add_argument("--head_coeff", type=float, default=1.0,
                        help="Ablation strength (1.0 = full zeroing). Match the behavioral run.")
    args = parser.parse_args()
    save_persona_vector(
        args.model_name,
        args.pos_path,
        args.neg_path,
        args.trait,
        args.save_dir,
        args.threshold,
        args.coherence_threshold,
        args.max_samples,
        args.load_in_8bit,
        args.save_activations,
        args.ablate_heads,
        args.head_coeff,
    )