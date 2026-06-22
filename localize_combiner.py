#!/usr/bin/env python3
"""
localize_combiner.py — causal localization of the persona-interaction "combiner".

WHY THIS DESIGN: the per-prompt residue R is high-rank/near-isotropic, so "find the
heads that compute R" (function-vector style) is ill-posed. Instead we localize the
LOW-RANK component that actually carries the claim: R's projection onto the evil axis.

CORE (exact, no patching). The residual stream is additive:
    resid[L_read] = embed + Σ_{j<L_read} (attn_out[j] + mlp_out[j])
Project every component's output onto the unit evil axis v̂ = τ̂_evil[L_read]. Because
projection is linear, the factorial second difference distributes over components:
    ⟨R, v̂⟩ = Σ_c ( ⟨c_pp, v̂⟩ − ⟨c_np, v̂⟩ − ⟨c_pn, v̂⟩ + ⟨c_nn, v̂⟩ )
So each component gets an EXACT attribution of the interaction's evil projection. The
sum over components reproduces ⟨R, v̂⟩ (printed as a sanity check).

Outputs per component (embed, attn_j, mlp_j):
  - interaction_attr : ⟨R_c, v̂⟩          (which layers compute the interaction suppression)
  - ai_effect_attr   : AI-slot main effect projection (½[(pp−np)+(pn−nn)])
  - user_effect_attr : user-slot main effect projection (½[(pp−pn)+(np−nn)])
The ai vs user main-effect profiles, projected onto the SAME shared evil axis, expose
the slot-asymmetry / γ-unification substrate (run once per setting; compare).

OPTIONAL --patch K: total-effect (direct+indirect) confirmation. For the top-K
interaction components, overwrite that component's last-token output in the pp run with
its nn-run value and re-measure ⟨resid[L_read], v̂⟩. Δ = total causal effect.

Cell convention (matches factorial_residue.py): first letter = AI slot, second = user.
  pp=AI+/user+  pn=AI+/user−  np=AI−/user+  nn=AI−/user−

Usage:
    python localize_combiner.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --extract_dir persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct \
      --prefix fac_evil_ai_humor_user --model_suffix instruct \
      --evil_axis persona_steering/persona_vectors/Qwen2.5-7B-Instruct/fac_evil_ai_humor_user_fac_tauA_response_avg_diff.pt \
      --L_read 21 --max_prompts 64 \
      --out_dir output/localize/Qwen2.5-7B-Instruct_evilai
    # run again with --prefix fac_humor_evil_user (keep the SAME --evil_axis) to compare slots
    # add --patch 6 for total-effect confirmation of the top-6 components
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


# ── per-prompt forward that captures component last-token outputs ───────────────

def component_projections(model, tok, prompt, v_hat, L_read, device):
    """Return dict component->scalar ⟨last-token output, v̂⟩ for one prompt.

    Components: 'embed', ('attn', j), ('mlp', j) for j in [0, L_read).
    Also returns the direct readout ⟨resid[L_read]_lasttok, v̂⟩ for the sanity check.
    """
    store = {}
    handles = []

    def mk(key):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            store[key] = t[0, -1, :].detach().float()  # last token, batch=1
        return hook

    layers = model.model.layers
    for j in range(L_read):
        handles.append(layers[j].self_attn.register_forward_hook(mk(("attn", j))))
        handles.append(layers[j].mlp.register_forward_hook(mk(("mlp", j))))

    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()

    hs = out.hidden_states  # tuple len n_layers+1; hs[0]=embed, hs[L]=resid after L blocks
    embed_last = hs[0][0, -1, :].float()
    resid_read = hs[L_read][0, -1, :].float()

    vh = v_hat.float()
    proj = {"embed": float(torch.dot(embed_last, vh))}
    for j in range(L_read):
        proj[("attn", j)] = float(torch.dot(store[("attn", j)].to(vh.device), vh))
        proj[("mlp", j)] = float(torch.dot(store[("mlp", j)].to(vh.device), vh))
    direct = float(torch.dot(resid_read, vh))
    return proj, direct


# ── optional: total-effect patching of one component (pp run, nn value) ─────────

def patch_component_effect(model, tok, prompt_pp, prompt_nn, comp, v_hat, L_read, device):
    """Overwrite comp's last-token output in the pp run with its nn-run value;
    return (baseline_proj, patched_proj) of ⟨resid[L_read]_lasttok, v̂⟩."""
    kind, j = comp
    mod = model.model.layers[j].self_attn if kind == "attn" else model.model.layers[j].mlp

    # 1. capture nn-run value of this component at its last token
    cache = {}
    def grab(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        cache["v"] = t[0, -1, :].detach().clone()
    h = mod.register_forward_hook(grab)
    ids_nn = tok(prompt_nn, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        model(ids_nn, use_cache=False)
    h.remove()

    vh = v_hat.float()
    ids_pp = tok(prompt_pp, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 2. baseline pp readout
    with torch.no_grad():
        base = model(ids_pp, output_hidden_states=True, use_cache=False)
    base_proj = float(torch.dot(base.hidden_states[L_read][0, -1, :].float(), vh))

    # 3. patched pp run: overwrite comp last-token output with nn value
    def patch(_m, _i, out):
        if isinstance(out, tuple):
            t = out[0]; t[0, -1, :] = cache["v"].to(t.dtype); return (t,) + tuple(out[1:])
        out[0, -1, :] = cache["v"].to(out.dtype); return out
    hp = mod.register_forward_hook(patch)
    with torch.no_grad():
        pat = model(ids_pp, output_hidden_states=True, use_cache=False)
    hp.remove()
    pat_proj = float(torch.dot(pat.hidden_states[L_read][0, -1, :].float(), vh))
    return base_proj, pat_proj


def comp_label(c):
    return c if isinstance(c, str) else f"{c[0]}_{c[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--extract_dir", required=True)
    ap.add_argument("--prefix", required=True, help="fac_evil_ai_humor_user or fac_humor_evil_user")
    ap.add_argument("--model_suffix", default="instruct", help="instruct | base (CSV filename suffix)")
    ap.add_argument("--evil_axis", required=True, help=".pt of the SHARED evil axis (Setting B tauA); [n_layers,d]")
    ap.add_argument("--L_read", type=int, default=21)
    ap.add_argument("--max_prompts", type=int, default=64)
    ap.add_argument("--patch", type=int, default=0, help="if >0, total-effect patch the top-K components")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    tau = torch.load(args.evil_axis, map_location="cpu", weights_only=False).float()
    v_hat = unit(tau[args.L_read]).to(device)
    print(f"Readout layer L={args.L_read}; evil axis ||τ||={float(tau[args.L_read].norm()):.2f}")

    # load matched prompts from the 4 cell CSVs
    cells = {}
    for c in ["pp", "pn", "np", "nn"]:
        p = Path(args.extract_dir) / f"{args.prefix}_{c}_pos_{args.model_suffix}.csv"
        cells[c] = pd.read_csv(p)["prompt"].tolist()
    n = min(len(cells[c]) for c in cells)
    n = min(n, args.max_prompts)
    print(f"Using {n} matched prompts across 4 cells.")

    # accumulate per-prompt component projections
    comps = ["embed"] + [("attn", j) for j in range(args.L_read)] + [("mlp", j) for j in range(args.L_read)]
    inter = {c: [] for c in comps}   # interaction attribution per prompt
    ai_ef = {c: [] for c in comps}   # AI main-effect attribution
    us_ef = {c: [] for c in comps}   # user main-effect attribution
    direct_inter = []                # ⟨R, v̂⟩ direct (sanity)

    for i in range(n):
        proj = {}
        dct = {}
        for c in ["pp", "pn", "np", "nn"]:
            proj[c], dct[c] = component_projections(model, tok, cells[c][i], v_hat, args.L_read, device)
        for comp in comps:
            pp, pn, np_, nn = proj["pp"][comp], proj["pn"][comp], proj["np"][comp], proj["nn"][comp]
            inter[comp].append(pp - np_ - pn + nn)
            ai_ef[comp].append(0.5 * ((pp - np_) + (pn - nn)))
            us_ef[comp].append(0.5 * ((pp - pn) + (np_ - nn)))
        direct_inter.append(dct["pp"] - dct["np"] - dct["pn"] + dct["nn"])
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{n} prompts")

    rng = np.random.default_rng(0)
    def stat(d):
        a = np.array(d)
        m = a.mean()
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)])
        return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    rows = []
    for comp in comps:
        mi, li, hi = stat(inter[comp])
        ma, _, _ = stat(ai_ef[comp])
        mu, _, _ = stat(us_ef[comp])
        rows.append({"component": comp_label(comp),
                     "interaction_attr": mi, "interaction_lo": li, "interaction_hi": hi,
                     "ai_effect_attr": ma, "user_effect_attr": mu})
    df = pd.DataFrame(rows)
    df.to_csv(out / "component_attribution.csv", index=False)

    # sanity check: Σ component interaction_attr ≈ mean direct ⟨R, v̂⟩
    summed = df["interaction_attr"].sum()
    direct = float(np.mean(direct_inter))
    print(f"\nSANITY: Σ_c ⟨R_c,v̂⟩ = {summed:.3f}   vs   direct ⟨R,v̂⟩ = {direct:.3f}   "
          f"(rel err {abs(summed-direct)/(abs(direct)+1e-8):.4f})")

    # ── plots (single axes each) ────────────────────────────────────────────
    top = df.reindex(df["interaction_attr"].abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(8.5, 5)); ax.grid(alpha=0.25, axis="x")
    ax.barh(top["component"][::-1], top["interaction_attr"][::-1],
            xerr=[ (top["interaction_attr"]-top["interaction_lo"])[::-1],
                   (top["interaction_hi"]-top["interaction_attr"])[::-1] ],
            color="#8e44ad")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(r"interaction attribution  $\langle R_c,\hat v_{evil}\rangle$")
    ax.set_title(f"Top components computing the interaction evil-projection (L_read={args.L_read})")
    fig.tight_layout(); fig.savefig(out / "interaction_attribution.png", dpi=150); plt.close(fig)

    # slot-asymmetry: ai vs user main-effect attribution per layer (attn+mlp summed)
    layer_ai = [df[df.component == f"attn_{j}"]["ai_effect_attr"].values[0] +
                df[df.component == f"mlp_{j}"]["ai_effect_attr"].values[0] for j in range(args.L_read)]
    layer_us = [df[df.component == f"attn_{j}"]["user_effect_attr"].values[0] +
                df[df.component == f"mlp_{j}"]["user_effect_attr"].values[0] for j in range(args.L_read)]
    fig, ax = plt.subplots(figsize=(8.5, 4.7)); ax.grid(alpha=0.25)
    ax.axhline(0, color="k", lw=1)
    ax.plot(range(args.L_read), layer_ai, "o-", color="#c0392b", ms=3, label="AI-slot evil writes")
    ax.plot(range(args.L_read), layer_us, "s-", color="#2e86c1", ms=3, label="user-slot evil writes")
    ax.set_xlabel("layer"); ax.set_ylabel(r"projection onto $\hat v_{evil}$")
    ax.set_title("Slot-asymmetry: per-layer main-effect writes onto the shared evil axis")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "slot_asymmetry.png", dpi=150); plt.close(fig)
    print(f"wrote component_attribution.csv + 2 plots to {out}/")

    # ── optional total-effect patching of top-K ─────────────────────────────
    if args.patch > 0:
        topk = [c for c in comps if c != "embed"]
        topk = sorted(topk, key=lambda c: -abs(np.mean(inter[c])))[:args.patch]
        print(f"\nTotal-effect patching top-{args.patch} components (pp←nn last token), "
              f"first {min(n,24)} prompts:")
        prows = []
        for comp in topk:
            deltas = []
            for i in range(min(n, 24)):
                b, p = patch_component_effect(model, tok, cells["pp"][i], cells["nn"][i],
                                              comp, v_hat, args.L_read, device)
                deltas.append(p - b)
            prows.append({"component": comp_label(comp),
                          "direct_attr": float(np.mean(inter[comp])),
                          "total_effect_patch": float(np.mean(deltas))})
            print(f"  {comp_label(comp):>8}: direct={np.mean(inter[comp]):+.3f}  "
                  f"total(patch)={np.mean(deltas):+.3f}")
        pd.DataFrame(prows).to_csv(out / "patch_total_effect.csv", index=False)
        print(f"wrote patch_total_effect.csv")


if __name__ == "__main__":
    main()