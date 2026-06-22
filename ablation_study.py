"""
ablation_study.py — Residue-Direction Intervention Study
=========================================================
Runs multiple intervention modes on the humorous_ai + evil_user joint condition
and scores behavioral change via the judge.

Intervention modes:
  baseline      — No intervention. Standard forward pass under joint instruction.
  ablate_ai     — Project out τ_A (humorous AI direction) at target layers.
                  h' = h - coeff * <h, τ̂_A> * τ̂_A
  amplify_user  — Additive CAA-style steering with τ_U (evil user direction).
                  h' = h + coeff * τ̂_U
  ablate_R      — Project out R_{A,U} (interaction residue direction).
                  h' = h - coeff * <h, R̂> * R̂
  reflect_R     — Reflect activations through hyperplane perpendicular to R.
                  h' = h - 2 * coeff * <h, R̂> * R̂

Usage:
    python ablation_study.py \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
        --trait_json persona_steering/data_generation/trait_data_eval/humorous_ai_user_evil.json \\
        --ai_vec    persona_steering/persona_vectors/Qwen2.5-7B-Instruct/orig_humorous_response_avg_diff.pt \\
        --user_vec  persona_steering/persona_vectors/Qwen2.5-7B-Instruct/user_evil_response_avg_diff.pt \\
        --residue   persona_steering/persona_vectors/Qwen2.5-7B-Instruct/residue_humorous_ai_user_evil_response_avg_diff.pt \\
        --intervention_layers 18-22 \\
        --modes baseline,ablate_ai,amplify_user,ablate_R,reflect_R \\
        --coeff 1.0 \\
        --persona_side pos \\
        --instruction_indices 0,1,2,3,4 \\
        --n_per_question 3 \\
        --max_new_tokens 128 \\
        --output_dir output/ablation_humorous_ai_evil_user \\
        --output_csv generations.csv

    # Sweep over multiple coefficients:
    python ablation_study.py ... --coeff 0.5,1.0,2.0 --modes ablate_R,reflect_R

    # Compute rho_eff before generating and save eval hidden states:
    python ablation_study.py ... --compute_rho_eff --save_eval_hidden_states
"""

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_vec(path: str) -> torch.Tensor:
    """Load a [n_layers, d] steering vector from .pt file."""
    v = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    return v.float()


def parse_layer_spec(spec: str) -> list:
    """Parse '18-22' or '18,20,22' into a sorted list of pub-layer indices."""
    out = []
    for part in spec.split(","):
        t = part.strip()
        if "-" in t:
            a, b = t.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return sorted(set(out))


def parse_coeff_spec(spec: str) -> list:
    """Parse '1.0' or '0.5,1.0,2.0' into a list of floats."""
    return [float(c.strip()) for c in spec.split(",")]


def locate_layers(model):
    """Find the transformer block list in the model."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "encoder.layer"):
        cur = model
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                break
        else:
            if hasattr(cur, "__getitem__") and hasattr(cur, "__len__"):
                return cur
    raise ValueError("Cannot locate transformer block list.")


# ── Intervention hook factory ──────────────────────────────────────────────────

def make_hook(mode: str, direction: torch.Tensor, coeff: float, device: str):
    """
    Returns a forward hook that applies the chosen intervention.

    Args:
        mode:      one of ablate_ai | amplify_user | ablate_R | reflect_R
        direction: [d] unit vector (already normalized) on CPU
        coeff:     intervention strength scalar
        device:    model device string

    The hook modifies the FIRST element of the transformer block output tuple
    (the residual stream hidden states, shape [batch, seq, d]).
    """
    d_vec = direction.to(device)

    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hs):
            return output

        # Ensure direction dtype matches hidden-state dtype to avoid type mismatch
        d_local = d_vec.to(hs.dtype)

        # hs: [batch, seq, d]
        if mode in ("ablate_ai", "ablate_R", "ablate_vec"):
            # directional ablation: h' = h - coeff * <h, d_hat> d_hat   (d_local is UNIT)
            #   ablate_ai  : d_hat = tau_A (e.g. the AI-evil axis)
            #   ablate_R   : d_hat = R     (wholesale residue ablation -- also kills coherence)
            #   ablate_vec : d_hat = any --ablate_vec axis (surgical evil axis / leakage axis)
            proj = (hs @ d_local).unsqueeze(-1) * d_local  # [batch, seq, d]
            hs_new = hs - coeff * proj

        elif mode in ("amplify_user", "add_raw", "restore_along"):
            # additive steering: h' = h + coeff * v
            #   amplify_user : v is the UNIT trait dir (legacy; needs LARGE coeff)
            #   add_raw      : v is a RAW vector (e.g. shared-leak gamma*tau_A) -- magnitude kept
            #   restore_along: v = -<R, tau_hat> tau_hat per layer (undo the interaction's
            #                  suppression of tau; coeff=1 exactly cancels it)
            hs_new = hs + coeff * d_local

        elif mode == "reflect_R":
            # h' = h - 2 * coeff * <h, d_local> * d_local  (reflection, coeff=1 → full reflection)
            proj = (hs @ d_local).unsqueeze(-1) * d_local
            hs_new = hs - 2.0 * coeff * proj

        else:
            return output  # should never reach here

        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new

    return hook_fn


# ── head-ablation machinery ─────────────────────────────────────────────────────

def parse_head_spec(spec: str) -> dict:
    """'19:27,19:22,20:0' -> {19: [27, 22], 20: [0]}.
    LAYER is the block index as reported by localize_heads (model.model.layers[L])."""
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        L, h = tok.split(":")
        out.setdefault(int(L), []).append(int(h))
    return out


def make_head_ablation_prehook(head_ids, head_dim, coeff, ref_vec=None):
    """forward_pre_hook on self_attn.o_proj: ablate the listed query heads IN the
    o_proj input (the concatenated per-head outputs z).
        z'_h = z_h - coeff * (z_h - ref_h)
      ref=None (null)  -> z'_h = (1-coeff) z_h      (coeff=1 zeros the head)
      ref=mean         -> z'_h = z_h - coeff(z_h-mean) (coeff=1 sets head to its mean)
    coeff acts as a dose (0..1 partial, 1 full); >1 over-ablates (extrapolates)."""
    def prehook(module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        x = x.clone()
        for h in head_ids:
            sl = slice(h * head_dim, (h + 1) * head_dim)
            ref = ref_vec[sl].to(x.dtype).to(x.device) if ref_vec is not None else 0.0
            x[..., sl] = x[..., sl] - coeff * (x[..., sl] - ref)
        return (x,) + tuple(inputs[1:])
    return prehook


def precompute_head_means(model, tokenizer, layers, head_layers, questions,
                          instructions, instruction_indices, persona_side,
                          device, max_new_tokens, n_ref=32):
    """Mean o_proj input per layer over reference prompt tokens (prefill only).
    Used for mean-ablation (replace head output with its dataset mean)."""
    acc = {L: None for L in head_layers}
    cnt = {L: 0 for L in head_layers}
    handles = []

    def mk(L):
        def pre(m, inputs):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1]).float().sum(0).cpu()
            acc[L] = flat if acc[L] is None else acc[L] + flat
            cnt[L] += x.shape[0] * x.shape[1]
        return pre

    for L in head_layers:
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)))
    seen = 0
    for q in questions:
        if seen >= n_ref:
            break
        for ins_idx in instruction_indices:
            if seen >= n_ref:
                break
            instruction = instructions[ins_idx][persona_side]
            sys_msg = f"{instruction} Keep your response within {max_new_tokens-13} tokens."
            messages = [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            seen += 1
    for h in handles:
        h.remove()
    return {L: (acc[L] / max(cnt[L], 1)) for L in head_layers}


# ── rho_eff diagnostic ─────────────────────────────────────────────────────────

def compute_rho_eff(
    model,
    tokenizer,
    layers,
    layer_ids: list,
    residue_vec: torch.Tensor,   # [n_layers, d]
    questions: list,
    instructions: list,
    instruction_indices: list,
    device: str,
    n_samples: int = 20,
    persona_side: str = "pos",
) -> dict:
    """
    Compute rho_eff and rho_rel layerwise before running any generation.

    rho_eff^(l) = E[|<h_x, R̂>|]
    rho_rel^(l) = rho_eff / E[||h_x||]

    Interpretation:
      > 0.05  → significant; null ablation = non-identifiability
      > 0.01  → moderate
      ≤ 0.01  → negligible; null result mechanically explained

    Returns a dict: {pub_layer: {"rho_eff": float, "rho_rel": float, "label": str}}
    """
    qs = questions[:n_samples]
    captured = {}

    def make_capture_hook(pub_layer):
        def hook_fn(m, i, output):
            hs = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hs):
                captured[pub_layer] = hs.detach().cpu().float()
        return hook_fn

    handles = []
    for pub_layer in layer_ids:
        mid = pub_layer - 1
        if 0 <= mid < len(layers):
            handles.append(layers[mid].register_forward_hook(make_capture_hook(pub_layer)))

    all_hs = {l: [] for l in layer_ids}

    try:
        for q in qs:
            for ins_idx in instruction_indices[:2]:  # just first 2 instructions for speed
                instruction = instructions[ins_idx][persona_side]
                sys_msg = f"{instruction} Keep your response within {args.max_new_tokens-13} tokens."
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": q},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    gen_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"], # this line was added by gemini
                        temperature=1.0,
                        top_p=1.0,
                        top_k=20,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                captured.clear()
                with torch.no_grad():
                    _ = model(gen_ids)

                for pub_layer in layer_ids:
                    if pub_layer in captured:
                        hs = captured[pub_layer]  # [1, full_seq, d]
                        resp_hs = hs[:, prompt_len:, :]
                        if resp_hs.shape[1] > 0:
                            all_hs[pub_layer].append(resp_hs.mean(dim=1).squeeze(0))
    finally:
        for h in handles:
            h.remove()

    results = {}
    for pub_layer in layer_ids:
        if not all_hs[pub_layer]:
            results[pub_layer] = {"rho_eff": 0.0, "rho_rel": 0.0, "label": "no data"}
            continue

        layer_vec_idx = pub_layer - 1  # residue_vec is 0-indexed by pub_layer-1
        # Clamp to available vector layers
        vec_idx = min(layer_vec_idx, residue_vec.shape[0] - 1)
        r = residue_vec[vec_idx]
        r_norm = float(torch.linalg.vector_norm(r).item())
        if r_norm < 1e-8:
            results[pub_layer] = {"rho_eff": 0.0, "rho_rel": 0.0, "label": "zero residue"}
            continue

        r_hat = r / r_norm
        h_stack = torch.stack(all_hs[pub_layer], dim=0)  # [n, d]
        projs = (h_stack @ r_hat).abs()
        rho_e = float(projs.mean().item())
        rho_r = rho_e / (float(h_stack.norm(dim=-1).mean().item()) + 1e-8)

        if rho_r > 0.05:
            label = "SIGNIFICANT — null = non-identifiability"
        elif rho_r > 0.01:
            label = "moderate"
        else:
            label = "negligible — null mechanically explained"

        results[pub_layer] = {"rho_eff": rho_e, "rho_rel": rho_r, "label": label}

    return results


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_responses(
    model,
    tokenizer,
    layers,
    layer_ids: list,
    mode: str,
    direction: torch.Tensor,   # [d] unit vector, CPU
    coeff: float,
    questions: list,
    instructions: list,
    instruction_indices: list,
    persona_side: str,
    n_per_question: int,
    max_new_tokens: int,
    device: str,
    save_eval_hidden_states: bool = False,
    residue_vec: torch.Tensor = None,
) -> tuple:
    """
    Generate responses under a given intervention mode.

    Returns:
        rows: list of dicts for CSV output
        hidden_states: [n_prompts, n_layers, d] tensor or None
    """
    handles = []
    eval_hs_store = {l: [] for l in layer_ids}
    captured_hs = {}

    if mode != "baseline":
        for pub_layer in layer_ids:
            mid = pub_layer - 1
            if 0 <= mid < len(layers):
                hook = make_hook(mode, direction, coeff, device)
                handles.append(layers[mid].register_forward_hook(hook))

    # If we need to capture eval hidden states, add capture hooks ON TOP
    # (they fire after intervention hooks since registered later)
    if save_eval_hidden_states and residue_vec is not None:
        def make_capture(pub_layer):
            def hook_fn(m, i, output):
                hs = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(hs):
                    captured_hs[pub_layer] = hs.detach().cpu().float()
            return hook_fn
        for pub_layer in layer_ids:
            mid = pub_layer - 1
            if 0 <= mid < len(layers):
                handles.append(layers[mid].register_forward_hook(make_capture(pub_layer)))

    rows = []
    prompt_counter = 0

    try:
        for q_idx, question in enumerate(questions):
            for ins_idx in instruction_indices:
                instruction = instructions[ins_idx][persona_side]
                sys_msg = f"{instruction} Keep your response within {args.max_new_tokens-13} tokens."
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": question},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]

                for _ in range(n_per_question):
                    with torch.no_grad():
                        gen_ids = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            temperature=1.0,
                            top_p=1.0,
                            top_k=20,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    # Capture hidden states: re-run forward pass on generated sequence
                    if save_eval_hidden_states and residue_vec is not None:
                        captured_hs.clear()
                        with torch.no_grad():
                            _ = model(gen_ids)
                        for pub_layer in layer_ids:
                            if pub_layer in captured_hs:
                                hs = captured_hs[pub_layer]
                                resp = hs[:, prompt_len:, :]
                                if resp.shape[1] > 0:
                                    eval_hs_store[pub_layer].append(
                                        resp.mean(dim=1).squeeze(0)
                                    )

                    response_ids = gen_ids[0, prompt_len:]
                    answer = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                    rows.append({
                        "question":        question,
                        "question_id":     q_idx,
                        "instruction_idx": ins_idx,
                        "mode":            mode,
                        "coeff":           coeff,
                        "answer":          answer,
                        "prompt":          prompt,
                    })
                    prompt_counter += 1

    finally:
        for h in handles:
            h.remove()

    # Stack eval hidden states if collected
    hs_tensor = None
    if save_eval_hidden_states and all(eval_hs_store[l] for l in layer_ids):
        layer_stacks = []
        for pub_layer in layer_ids:
            layer_stacks.append(
                torch.stack(eval_hs_store[pub_layer], dim=0)  # [n, d]
            )
        # shape: [n_prompts, n_layers, d]
        hs_tensor = torch.stack(layer_stacks, dim=1)

    return rows, hs_tensor


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Residue-direction intervention study.")

    # Model
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype",      default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])

    # Data
    parser.add_argument("--trait_json",
                        default="data_generation/trait_data_eval/humorous_ai_user_evil.json")
    parser.add_argument("--limit_questions", type=int, default=0,
                        help="Use only the first N questions. 0 = all.")
    parser.add_argument("--instruction_indices", default="0,1,2,3,4")
    parser.add_argument("--persona_side", default="pos", choices=["pos", "neg"])
    parser.add_argument("--n_per_question", type=int, default=3)

    # Steering vectors (any subset can be provided; only modes that need them will run)
    parser.add_argument("--ai_vec",  default="",
                        help="[n_layers, d] humorous AI steering vector (.pt).")
    parser.add_argument("--user_vec", default="",
                        help="[n_layers, d] evil user steering vector (.pt).")
    parser.add_argument("--residue",  default="",
                        help="[n_layers, d] interaction residue vector R (.pt). "
                             "NOTE: use the actual R file (factorial_residue.py now emits "
                             "<prefix>_fac_R_response_avg_diff.pt), NOT the joint.")
    parser.add_argument("--ablate_vec", default="",
                        help="[n_layers, d] arbitrary axis to ablate (mode ablate_vec): "
                             "e.g. the AI-evil axis tau_A for surgical evil ablation, or for "
                             "leakage ablation when generating the evil-USER condition.")
    parser.add_argument("--add_vec", default="",
                        help="[n_layers, d] RAW vector to ADD (mode add_raw), magnitude kept: "
                             "e.g. trait_leakage.py's *_shared_leak_*.pt to amplify leakage.")
    parser.add_argument("--restore_target", default="",
                        help="[n_layers, d] trait axis tau for mode restore_along; with --residue R, "
                             "adds -<R,tau_hat>tau_hat per layer to undo R's suppression of tau.")

    # Head ablation (mode ablate_head)
    parser.add_argument("--ablate_heads", default="",
                        help="Head spec 'L:h,L:h' using block indices as reported by "
                             "localize_heads (model.model.layers[L].self_attn head h). "
                             "e.g. '19:27,19:22,19:2,19:4'. Used by mode ablate_head.")
    parser.add_argument("--head_ablation_type", default="null", choices=["null", "mean"],
                        help="null = zero the head output; mean = replace with its dataset mean.")
    parser.add_argument("--head_mean_samples", type=int, default=32,
                        help="reference prompts for the mean (head_ablation_type=mean).")

    # Intervention
    parser.add_argument("--intervention_layers", default="18-22",
                        help="Pub-layer spec: '18-22' or '18,20,22'.")
    parser.add_argument("--modes", default="baseline,ablate_ai,amplify_user,ablate_R,reflect_R",
                        help="Comma-separated intervention modes.")
    parser.add_argument("--coeff", default="1.0",
                        help="Intervention coefficient(s). Comma-sep for sweep: '0.5,1.0,2.0'.")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")

    # Diagnostics
    parser.add_argument("--compute_rho_eff", action="store_true",
                        help="Compute rho_eff/rho_rel before generation.")
    parser.add_argument("--rho_eff_samples", type=int, default=20,
                        help="Number of questions used for rho_eff estimation.")
    parser.add_argument("--save_eval_hidden_states", action="store_true",
                        help="Save [n_prompts, n_layers, d] hidden states per mode.")

    # Output
    parser.add_argument("--output_dir", default="output/ablation")
    parser.add_argument("--output_csv", default="generations.csv")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",")]
    coeffs = parse_coeff_spec(args.coeff)
    layer_ids = parse_layer_spec(args.intervention_layers)
    instruction_indices = [int(i) for i in args.instruction_indices.split(",")]

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading {args.model_name} ({args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    layers = locate_layers(model)
    n_model_layers = len(layers)
    hidden_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", hidden_size // n_heads)

    # Validate and clamp layer spec
    valid_layer_ids = []
    for l in layer_ids:
        mid = l - 1
        if 0 <= mid < n_model_layers:
            valid_layer_ids.append(l)
        else:
            print(f"  [WARNING] pub_layer={l} → block {mid} out of range [0,{n_model_layers-1}]; skipping.")
    layer_ids = valid_layer_ids
    if not layer_ids:
        raise ValueError("No valid intervention layers after bounds check.")
    print(f"  Intervention layers: {layer_ids} (model has {n_model_layers} blocks)")

    # ── Load steering vectors ───────────────────────────────────────────────────
    # Each vector is [n_vec_layers, d]; we index by pub_layer-1.
    # We pre-compute unit vectors per layer for efficiency.

    def get_unit_vec(vec_tensor: torch.Tensor, pub_layer: int, device: str) -> torch.Tensor:
        """Return normalized direction vector for a given pub_layer."""
        # vec_tensor shape: [n_layers, d], indexed at pub_layer - 1
        idx = min(pub_layer - 1, vec_tensor.shape[0] - 1)
        v = vec_tensor[idx].float().to(device)
        norm = torch.linalg.vector_norm(v)
        return v / (norm + 1e-8)

    def get_raw_vec(vec_tensor: torch.Tensor, pub_layer: int, device: str) -> torch.Tensor:
        """Return the RAW (un-normalized) per-layer vector for a given pub_layer.
        Used by add_raw / restore_along where the vector's MAGNITUDE is meaningful."""
        idx = min(pub_layer - 1, vec_tensor.shape[0] - 1)
        return vec_tensor[idx].float().to(device)

    ai_vec   = load_vec(args.ai_vec)   if args.ai_vec   else None
    user_vec = load_vec(args.user_vec) if args.user_vec else None
    residue  = load_vec(args.residue)  if args.residue  else None
    ablate_vec_t     = load_vec(args.ablate_vec)     if args.ablate_vec     else None
    add_vec_t        = load_vec(args.add_vec)        if args.add_vec        else None
    restore_target_t = load_vec(args.restore_target) if args.restore_target else None

    # Validate that required vectors are available for requested modes
    mode_requirements = {
        "baseline":    [],
        "ablate_ai":   ["ai_vec"],
        "amplify_user":["user_vec"],
        "ablate_R":    ["residue"],
        "reflect_R":   ["residue"],
        "ablate_vec":  ["ablate_vec_t"],
        "add_raw":     ["add_vec_t"],
        "restore_along":["residue", "restore_target_t"],
        "ablate_head": [],  # needs --ablate_heads (string), validated separately
    }
    for mode in modes:
        if mode not in mode_requirements:
            raise ValueError(f"Unknown mode: {mode}. "
                             f"Choose from: {list(mode_requirements.keys())}")
        for req in mode_requirements[mode]:
            if locals()[req] is None:
                raise ValueError(f"Mode '{mode}' requires --{req.replace('_', '_')} "
                                 f"to be provided.")
    if "ablate_head" in modes and not args.ablate_heads:
        raise ValueError("Mode 'ablate_head' requires --ablate_heads 'L:h,L:h'.")

    # ── Load dataset ────────────────────────────────────────────────────────────
    data = json.loads(Path(args.trait_json).read_text())
    questions    = data["questions"]
    instructions = data["instruction"]
    if args.limit_questions > 0:
        questions = questions[:args.limit_questions]
    print(f"  Questions: {len(questions)}, instructions: {instruction_indices}")

    # ── Head-mean precompute (only if ablate_head + mean) ────────────────────────
    head_means = None
    if "ablate_head" in modes and args.head_ablation_type == "mean":
        head_layers = sorted(parse_head_spec(args.ablate_heads).keys())
        print(f"  Precomputing head means for layers {head_layers} "
              f"({args.head_mean_samples} ref prompts)...")
        head_means = precompute_head_means(
            model, tokenizer, layers, head_layers, questions, instructions,
            instruction_indices, args.persona_side, args.device,
            args.max_new_tokens, n_ref=args.head_mean_samples)

    # ── rho_eff diagnostic ──────────────────────────────────────────────────────
    if args.compute_rho_eff and residue is not None:
        print("\n── rho_eff diagnostic ────────────────────────────────────────────")
        rho_results = compute_rho_eff(
            model=model, tokenizer=tokenizer,
            layers=layers, layer_ids=layer_ids,
            residue_vec=residue,
            questions=questions,
            instructions=instructions,
            instruction_indices=instruction_indices,
            device=args.device,
            n_samples=args.rho_eff_samples,
            persona_side=args.persona_side,
        )
        rho_path = out_dir / "rho_eff_diagnostic.json"
        import json as _json
        rho_path.write_text(_json.dumps(rho_results, indent=2))
        print(f"{'Layer':<8} {'rho_eff':>10} {'rho_rel':>10}  Label")
        print("-" * 60)
        for l, r in rho_results.items():
            print(f"L{l:<7} {r['rho_eff']:>10.4f} {r['rho_rel']:>10.4f}  {r['label']}")
        print(f"\nSaved: {rho_path}")

    # ── Main generation loop ────────────────────────────────────────────────────
    all_rows = []

    # Expand over coefficient sweep × modes
    run_specs = []
    for coeff in coeffs:
        for mode in modes:
            if mode == "baseline" and coeff != coeffs[0]:
                continue  # baseline is coefficient-independent; run once
            run_specs.append((mode, coeff))

    print(f"\n── Generation ({len(run_specs)} run(s)) ──────────────────────────────")

    for mode, coeff in run_specs:
        print(f"  Mode: {mode:<15} coeff={coeff:.2f} ...", end=" ", flush=True)

        # Select direction vector for this mode
        if mode == "baseline":
            direction = torch.zeros(hidden_size)  # unused for baseline
        elif mode == "ablate_ai":
            # Average unit vectors across intervention layers (or use per-layer in hook)
            # We pass a placeholder here; the hook uses per-layer logic below.
            direction = torch.zeros(hidden_size)  # handled per-layer in multi-layer version
        elif mode == "amplify_user":
            direction = torch.zeros(hidden_size)
        elif mode in ("ablate_R", "reflect_R"):
            direction = torch.zeros(hidden_size)

        # ── Multi-layer hook registration with per-layer vectors ──────────────
        # We override the per-layer logic here directly for cleanliness.

        handles = []
        eval_hs_store = {l: [] for l in layer_ids}
        captured_hs = {}

        if mode == "ablate_head":
            head_spec = parse_head_spec(args.ablate_heads)
            ref_means = head_means if args.head_ablation_type == "mean" else None
            for L, hds in head_spec.items():
                if not (0 <= L < n_model_layers):
                    print(f"[skip head layer {L} OOB] ", end="")
                    continue
                ref = ref_means[L] if ref_means is not None else None
                ph = make_head_ablation_prehook(hds, head_dim, coeff, ref)
                handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(ph))
        elif mode != "baseline":
            for pub_layer in layer_ids:
                mid = pub_layer - 1
                if not (0 <= mid < n_model_layers):
                    continue

                # Compute per-layer unit direction
                if mode == "ablate_ai":
                    d_hat = get_unit_vec(ai_vec, pub_layer, args.device)
                elif mode == "amplify_user":
                    d_hat = get_unit_vec(user_vec, pub_layer, args.device)
                elif mode in ("ablate_R", "reflect_R"):
                    d_hat = get_unit_vec(residue, pub_layer, args.device)
                elif mode == "ablate_vec":
                    d_hat = get_unit_vec(ablate_vec_t, pub_layer, args.device)
                elif mode == "add_raw":
                    d_hat = get_raw_vec(add_vec_t, pub_layer, args.device)  # RAW: magnitude kept
                elif mode == "restore_along":
                    R_l   = get_raw_vec(residue, pub_layer, args.device)
                    t_hat = get_unit_vec(restore_target_t, pub_layer, args.device)
                    coef  = torch.dot(R_l, t_hat)   # <R, tau_hat>; negative when tau is suppressed
                    d_hat = -coef * t_hat           # add back exactly what R removed (coeff=1)

                hook = make_hook(mode, d_hat, coeff, args.device)
                handles.append(layers[mid].register_forward_hook(hook))

        # Eval hidden-state capture: ONLY in baseline mode. Capturing while the
        # intervention hooks are live (they are removed only in `finally`) gives
        # post-intervention states, which makes rho_eff/rho_rel meaningless. The
        # rho_rel denominator needs the CLEAN residual stream, so we capture baseline only.
        capture_eval = args.save_eval_hidden_states and residue is not None and mode == "baseline"
        if args.save_eval_hidden_states and residue is not None and mode != "baseline":
            print("[skip eval-hs capture: not baseline] ", end="")
        if capture_eval:
            def make_capture_hook(pub_layer):
                def hook_fn(m, i, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    if torch.is_tensor(hs):
                        captured_hs[pub_layer] = hs.detach().cpu().float()
                return hook_fn
            for pub_layer in layer_ids:
                mid = pub_layer - 1
                if 0 <= mid < n_model_layers:
                    handles.append(layers[mid].register_forward_hook(make_capture_hook(pub_layer)))

        # Generation
        rows = []
        try:
            for q_idx, question in tqdm(enumerate(questions), desc=f"Generating for {mode}", total=len(questions)):
                for ins_idx in instruction_indices:
                    instruction = instructions[ins_idx][args.persona_side]
                    sys_msg = f"{instruction} Keep your response within {args.max_new_tokens-13} tokens."
                    messages = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user",   "content": question},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    )
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    prompt_len = inputs["input_ids"].shape[1]

                    for _ in range(args.n_per_question):
                        with torch.no_grad():
                            gen_ids = model.generate(
                                inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                temperature=1.0,
                                top_p=1.0,
                                top_k=20,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                            )

                        # Capture hidden states via second forward pass
                        if capture_eval:
                            captured_hs.clear()
                            with torch.no_grad():
                                _ = model(gen_ids)
                            for pub_layer in layer_ids:
                                if pub_layer in captured_hs:
                                    hs = captured_hs[pub_layer]
                                    resp = hs[:, prompt_len:, :]
                                    if resp.shape[1] > 0:
                                        eval_hs_store[pub_layer].append(
                                            resp.mean(dim=1).squeeze(0)
                                        )

                        response_ids = gen_ids[0, prompt_len:]
                        answer = tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        ).strip()

                        rows.append({
                            "question":        question,
                            "question_id":     q_idx,
                            "instruction_idx": ins_idx,
                            "mode":            mode,
                            "coeff":           coeff,
                            "ablated_heads":   args.ablate_heads if mode == "ablate_head" else "",
                            "head_ablation_type": args.head_ablation_type if mode == "ablate_head" else "",
                            "answer":          answer,
                            "prompt":          prompt,
                            "system_message":  sys_msg,
                        })
        finally:
            for h in handles:
                h.remove()

        print(f"{len(rows)} responses")
        all_rows.extend(rows)

        # Save eval hidden states for this mode
        if capture_eval:
            if all(eval_hs_store[l] for l in layer_ids):
                layer_stacks = [
                    torch.stack(eval_hs_store[l], dim=0) for l in layer_ids
                ]
                hs_tensor = torch.stack(layer_stacks, dim=1)  # [n, n_layers, d]
                hs_path = out_dir / f"eval_hidden_states_{mode}_coeff{coeff:.1f}.pt"
                torch.save(hs_tensor, hs_path)
                print(f"    Saved hidden states: {hs_path}  shape={tuple(hs_tensor.shape)}")

    # ── Write CSV ───────────────────────────────────────────────────────────────
    csv_path = out_dir / args.output_csv
    fieldnames = [
        "question", "question_id", "instruction_idx",
        "mode", "coeff", "ablated_heads", "head_ablation_type",
        "answer", "prompt", "system_message",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nAll done. {len(all_rows)} total rows → {csv_path}")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()