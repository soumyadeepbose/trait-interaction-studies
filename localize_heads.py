#!/usr/bin/env python3
"""
localize_heads.py — decompose attention layers into per-head contributions to the
interaction's evil-projection, to find the ROLE-SELECTIVE suppression head(s).

Follows from localize_combiner.py, which localized the interaction suppression to
attn_19 (slot-selective) + mlp_18/19/20 (slot-agnostic). Here we split the target
attention layers head-by-head and run BOTH settings in one pass, so the role
selectivity (suppresses AI-evil but not user-evil) is read off directly.

Per-head decomposition is exact: o_proj is linear, so
    attn_out = Σ_h  W_o[:, h·hd:(h+1)·hd] @ z_h
where z = the concatenated per-head outputs (the input to o_proj). Each head's
contribution is projected onto the shared unit evil axis v̂ = τ̂_evil[L_read] and
second-differenced across the 4 factorial cells. The per-head projections sum to the
layer-level attribution from localize_combiner (printed as a sanity check).

Outputs:
  - per_head_attribution.csv : layer, head, interaction_B, interaction_A,
    role_selectivity (= interaction_A − interaction_B; large positive ⇒ suppresses
    AI-evil far more than user-evil), ai_evil_write_B, user_evil_write_A
  - per_head_attribution.png : scatter of interaction_B vs interaction_A per head;
    role-selective heads sit far below the diagonal.

Optional --attn_patterns K: for the top-K role-selective heads, report the last-token
attention mass over the {system, user, assistant-header} token spans (does the role
head attend to the system prompt that declares the AI persona?).

Usage:
    python localize_heads.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --extract_dir persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct \
      --prefix_b fac_evil_ai_humor_user --prefix_a fac_humor_evil_user \
      --model_suffix instruct \
      --evil_axis persona_steering/persona_vectors/Qwen2.5-7B-Instruct/fac_evil_ai_humor_user_fac_tauA_response_avg_diff.pt \
      --layers 19,20 --L_read 21 --max_prompts 64 \
      --out_dir output/localize_heads/Qwen2.5-7B-Instruct \
      [--attn_patterns 4]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def unit(v, eps=1e-8):
    return v / (v.norm() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--extract_dir", required=True)
    ap.add_argument("--prefix_b", default="fac_evil_ai_humor_user", help="evil-in-AI setting")
    ap.add_argument("--prefix_a", default="fac_humor_evil_user", help="evil-in-USER setting")
    ap.add_argument("--model_suffix", default="instruct")
    ap.add_argument("--evil_axis", required=True, help="shared evil axis .pt [n_layers,d]")
    ap.add_argument("--layers", default="19,20", help="attention layers to decompose")
    ap.add_argument("--L_read", type=int, default=21)
    ap.add_argument("--max_prompts", type=int, default=64)
    ap.add_argument("--attn_patterns", type=int, default=0, help="top-K role heads for attn-pattern readout")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"n_heads={n_heads}, head_dim={head_dim}")

    tau = torch.load(args.evil_axis, map_location="cpu", weights_only=False).float()
    v_hat = unit(tau[args.L_read]).to(device).float()

    # precompute per-head readout vectors u_{L,h} = W_o[:, h_slice]^T @ v̂   (shape [head_dim])
    # so ⟨head_h output, v̂⟩ = u_{L,h} · z_h   (z_h = o_proj input slice for head h)
    u = {}
    for L in target_layers:
        Wo = model.model.layers[L].self_attn.o_proj.weight.detach().float()  # [hidden, n_heads*head_dim]
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            u[(L, h)] = (Wo[:, sl].t() @ v_hat).cpu()  # [head_dim]

    # forward_pre_hook on o_proj captures its input (concatenated head outputs) at last token
    store = {}
    def mk(L):
        def pre(_m, args_in):
            x = args_in[0]
            store[L] = x[0, -1, :].detach().float().cpu()  # [n_heads*head_dim]
        return pre

    def head_projections(prompt):
        handles = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L))
                   for L in target_layers]
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            model(ids, use_cache=False)
        for h in handles:
            h.remove()
        proj = {}
        for L in target_layers:
            z = store[L]
            for h in range(n_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                proj[(L, h)] = float(torch.dot(u[(L, h)], z[sl]))
        return proj

    def load_cells(prefix):
        cells = {}
        for c in ["pp", "pn", "np", "nn"]:
            p = Path(args.extract_dir) / f"{prefix}_{c}_pos_{args.model_suffix}.csv"
            cells[c] = pd.read_csv(p)["prompt"].tolist()
        return cells

    def run_setting(prefix):
        cells = load_cells(prefix)
        n = min(min(len(cells[c]) for c in cells), args.max_prompts)
        inter = {(L, h): [] for L in target_layers for h in range(n_heads)}
        ai_ef = {(L, h): [] for L in target_layers for h in range(n_heads)}
        us_ef = {(L, h): [] for L in target_layers for h in range(n_heads)}
        for i in range(n):
            pr = {c: head_projections(cells[c][i]) for c in ["pp", "pn", "np", "nn"]}
            for key in inter:
                pp, pn, np_, nn = pr["pp"][key], pr["pn"][key], pr["np"][key], pr["nn"][key]
                inter[key].append(pp - np_ - pn + nn)
                ai_ef[key].append(0.5 * ((pp - np_) + (pn - nn)))
                us_ef[key].append(0.5 * ((pp - pn) + (np_ - nn)))
            if (i + 1) % 16 == 0:
                print(f"  [{prefix}] {i+1}/{n}")
        return n, inter, ai_ef, us_ef

    print("Running Setting B (evil-AI) ...")
    nB, interB, aiB, _ = run_setting(args.prefix_b)
    print("Running Setting A (evil-user) ...")
    nA, interA, _, usA = run_setting(args.prefix_a)

    rng = np.random.default_rng(0)
    def ci(d):
        a = np.array(d); m = a.mean()
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)])
        return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    rows = []
    for L in target_layers:
        for h in range(n_heads):
            k = (L, h)
            mB, loB, hiB = ci(interB[k])
            mA, loA, hiA = ci(interA[k])
            rows.append({
                "layer": L, "head": h,
                "interaction_B": mB, "interaction_B_lo": loB, "interaction_B_hi": hiB,
                "interaction_A": mA, "interaction_A_lo": loA, "interaction_A_hi": hiA,
                "role_selectivity": mA - mB,           # large positive ⇒ suppresses AI-evil >> user-evil
                "ai_evil_write_B": np.mean(aiB[k]),
                "user_evil_write_A": np.mean(usA[k]),
            })
    df = pd.DataFrame(rows).sort_values("role_selectivity", ascending=False)
    df.to_csv(out / "per_head_attribution.csv", index=False)

    # sanity: per-head sum per layer == layer attribution (compare to localize_combiner)
    for L in target_layers:
        sB = df[df.layer == L]["interaction_B"].sum()
        sA = df[df.layer == L]["interaction_A"].sum()
        print(f"  layer {L}: Σ_heads interaction  B={sB:+.3f}  A={sA:+.3f}  "
              f"(should match attn_{L} in localize_combiner)")

    # plot: interaction_B vs interaction_A per head; role-selective heads sit below diagonal
    fig, ax = plt.subplots(figsize=(7.2, 6.4)); ax.grid(alpha=0.25)
    lim = float(np.abs(df[["interaction_B", "interaction_A"]].values).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5, label="slot-agnostic (B = A)")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    for L, mk_ in zip(target_layers, ["o", "s", "^", "D"]):
        d = df[df.layer == L]
        ax.scatter(d["interaction_B"], d["interaction_A"], marker=mk_, s=40,
                   alpha=0.8, label=f"layer {L}")
        # label the most role-selective heads (Fixed code using bracket notation)
        for _, r in d.sort_values("role_selectivity", ascending=False).head(2).iterrows():
            ax.annotate(f"L{int(r['layer'])}.h{int(r['head'])}", (r.interaction_B, r.interaction_A),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(r"interaction (evil in AI slot)  $\langle R_h, \hat v_{evil}\rangle$")
    ax.set_ylabel(r"interaction (evil in USER slot)")
    ax.set_title("Per-head interaction: role-selective heads suppress AI-evil but not user-evil")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "per_head_attribution.png", dpi=150); plt.close(fig)
    print(f"wrote per_head_attribution.csv + .png to {out}/")

    print("\nTop role-selective heads (suppress AI-evil >> user-evil):")
    for _, r in df.head(6).iterrows():
        print(f"  L{int(r['layer'])}.h{int(r['head'])}: interaction_B={r.interaction_B:+.3f}  "
              f"interaction_A={r.interaction_A:+.3f}  selectivity={r.role_selectivity:+.3f}")

    # ── optional: attention-pattern readout for top role-selective heads ─────────
    if args.attn_patterns > 0:
        top = df.head(args.attn_patterns)[["layer", "head"]].values.tolist()
        print(f"\nAttention patterns for top-{args.attn_patterns} role heads "
              f"(last-token mass over role spans, Setting B pp, first 16 prompts):")
        cellsB = load_cells(args.prefix_b)
        imstart = tok.convert_tokens_to_ids("<|im_start|>")
        span_acc = {(L, h): {"system": [], "user": [], "assistant": []} for L, h in top}
        for i in range(min(16, len(cellsB["pp"]))):
            ids = tok(cellsB["pp"][i], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            seq = ids[0].tolist()
            marks = [j for j, t in enumerate(seq) if t == imstart]
            if len(marks) < 3:
                continue
            spans = {"system": range(marks[0], marks[1]),
                     "user": range(marks[1], marks[2]),
                     "assistant": range(marks[2], len(seq))}
            with torch.no_grad():
                o = model(ids, output_attentions=True, use_cache=False)
            for (L, h) in top:
                attn = o.attentions[L][0, h, -1, :].float().cpu().numpy()  # last-token over keys
                for name, rng_ in spans.items():
                    span_acc[(L, h)][name].append(float(attn[list(rng_)].sum()))
        prows = []
        for (L, h) in top:
            d = span_acc[(L, h)]
            row = {"layer": L, "head": h}
            for name in ["system", "user", "assistant"]:
                row[f"attn_{name}"] = float(np.mean(d[name])) if d[name] else float("nan")
            prows.append(row)
            print(f"  L{L}.h{h}: system={row['attn_system']:.2f}  user={row['attn_user']:.2f}  "
                  f"assistant={row['attn_assistant']:.2f}")
        pd.DataFrame(prows).to_csv(out / "role_head_attention_spans.csv", index=False)
        print(f"wrote role_head_attention_spans.csv")


if __name__ == "__main__":
    main()