#!/usr/bin/env python3
"""
interaction_tensor.py — Operationalize Second-Order Persona Algebra.

This is the ONLY new file Tier 2 needs. Everything here is offline linear algebra on
vectors you already produce with the existing pipeline:

  * factorial cell vectors  <prefix>_{pp,pn,np,nn}_response_avg_diff.pt
      (from build_factorial_dataset.py -> eval_persona -> generate_vec, per trait pair)
  * the refusal direction   refusal_response_avg_diff.pt  (extract_refusal_direction.py)
  * ablation_study.py CSVs  (steered generations + judge scores) for the behavioral law

Runtime steering is ALREADY handled by your ablation_study.py (amplify_user mode adds
coeff*vector with any vector you pass, sweeps coeff, scores with the judge, and can save
hidden states). So no new model-runtime file is required.

Subcommands
-----------
  assemble        Build the interaction tensor T [k_ai, k_user, n_layers, d] from the
                  per-pair factorial cells. (T[i,j] = pp - np - pn + nn for pair (i,j).)
  decompose       Per-layer mode-3 (HOSVD) decomposition of T: effective # of interaction
                  modes + the activation-space mode directions g_r, saved in steering-vector
                  shape [n_layers, d] so you can feed them straight to ablation_study.py.
  asymmetry       Slot asymmetry ||T - T^swap|| (needs square, same-ordered axes; use on
                  the AI+AI tensor), with per-trait localization.
  refusal_map     S[i,j] = <T[i,j], d_refusal_hat> per layer: the "safety is an interaction"
                  map. Predicts large positive entries on evil rows in instruct, ~0 in base.
  delta           Delta_T = T_instruct - T_base + its decomposition + refusal projection:
                  the second-order structure alignment training installs.
  predict         Held-out composite prediction (the headline bilinearity test): predict
                  R for a composite persona from single-trait T entries, compare to measured.
  behavioral_law  Consume an ablation_study.py coeff-sweep CSV: compute realized alpha(c)
                  and correlate with the judge score; sufficiency overlay across recipes.

All vectors are [n_layers, d]; T is [k_ai, k_user, n_layers, d]. Per-layer is the default
granularity throughout (matches metrics.py).

NOTE ON DEPENDENCIES: torch + numpy + pandas only. The decomposition uses an SVD of the
mode-3 unfolding (HOSVD), which is basis-free in activation space and needs no tensorly.
For a full CP/PARAFAC factorization (joint factors on all three axes) install tensorly and
see the `cp_optional` note in decompose().
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ───────────────────────── shared utilities ─────────────────────────

def load_vec(path):
    v = torch.load(str(path), map_location="cpu", weights_only=False)
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    return v.float()


def unit(v, eps=1e-8):
    return v / (v.norm() + eps)


def cos_per_layer(a, b, eps=1e-8):
    # a, b: [n_layers, d] -> [n_layers]
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return num / den


def effective_rank(singular_values, eps=1e-12):
    """exp(Shannon entropy of normalized squared spectrum) — matches the EffRank metric."""
    s2 = singular_values ** 2
    p = s2 / (s2.sum() + eps)
    p = p[p > eps]
    H = -(p * p.log()).sum()
    return float(H.exp())


def cell_residue(vec_dir, prefix):
    """R = pp - np - pn + nn for one trait pair, from its 4 factorial cell vectors."""
    cells = {}
    for c in ("pp", "pn", "np", "nn"):
        p = Path(vec_dir) / f"{prefix}_{c}_response_avg_diff.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing cell vector: {p}")
        cells[c] = load_vec(p)
    return cells["pp"] - cells["np"] - cells["pn"] + cells["nn"]


# ───────────────────────── assemble ─────────────────────────

def cmd_assemble(args):
    ai = [t.strip() for t in args.ai_traits.split(",")]
    user = [t.strip() for t in args.user_traits.split(",")]
    rows = []
    for i, a in enumerate(ai):
        col = []
        for j, u in enumerate(user):
            prefix = args.pair_prefix.format(ai=a, user=u)
            R = cell_residue(args.vec_dir, prefix)  # [n_layers, d]
            col.append(R)
        rows.append(torch.stack(col, dim=0))       # [k_user, n_layers, d]
    T = torch.stack(rows, dim=0)                   # [k_ai, k_user, n_layers, d]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(T, str(out))
    manifest = {"ai_traits": ai, "user_traits": user,
                "shape": list(T.shape), "pair_prefix": args.pair_prefix}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  assembled T {tuple(T.shape)} -> {out}")
    # quick health check: per-pair residue norm at the deepest layer
    L = T.shape[2] - 1
    print("  ||T[i,j]|| at last layer:")
    for i, a in enumerate(ai):
        line = "    " + f"{a:>14}: " + " ".join(
            f"{user[j][:6]}={T[i, j, L].norm():6.2f}" for j in range(len(user)))
        print(line)


# ───────────────────────── decompose ─────────────────────────

def cmd_decompose(args):
    T = load_vec(args.tensor)                       # [k_ai, k_user, n_layers, d]
    k_ai, k_user, n_layers, d = T.shape
    r = args.rank
    modes = torch.zeros(r, n_layers, d)             # mode directions, steering-vector shaped
    spec_rows, load_rows = [], []
    for l in range(n_layers):
        M = T[:, :, l, :].reshape(k_ai * k_user, d)  # mode-3 unfolding
        # SVD: rows = (i,j) pairs, cols = activation dims.
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        er = effective_rank(S)
        energy = (S[:r] ** 2).sum() / ((S ** 2).sum() + 1e-12)
        spec_rows.append(dict(layer=l, eff_rank=er, top_r_energy=float(energy),
                              **{f"sv{q}": float(S[q]) for q in range(min(r, len(S)))}))
        for q in range(min(r, Vh.shape[0])):
            modes[q, l, :] = Vh[q]                   # activation-space direction g_q
            # loading of this mode back onto the (i,j) grid (which pairs excite it)
            load = (U[:, q] * S[q]).reshape(k_ai, k_user)
            load_rows.append(dict(layer=l, mode=q,
                                  **{f"load_{i}_{j}": float(load[i, j])
                                     for i in range(k_ai) for j in range(k_user)}))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spec_rows).to_csv(outdir / "tensor_spectrum.csv", index=False)
    pd.DataFrame(load_rows).to_csv(outdir / "tensor_mode_loadings.csv", index=False)
    torch.save(modes, str(outdir / "interaction_modes.pt"))
    print(f"  per-layer eff_rank range: "
          f"{min(rw['eff_rank'] for rw in spec_rows):.2f}"
          f"–{max(rw['eff_rank'] for rw in spec_rows):.2f}")
    print(f"  saved interaction_modes.pt {tuple(modes.shape)} "
          f"(feed a mode to ablation_study.py --user_vec to behaviorally validate it)")
    print(f"  spectrum -> {outdir/'tensor_spectrum.csv'}, loadings -> {outdir/'tensor_mode_loadings.csv'}")
    print("  cp_optional: for a full CP/PARAFAC (factors on all 3 axes), use tensorly.decomposition.parafac")


# ───────────────────────── asymmetry ─────────────────────────

def cmd_asymmetry(args):
    T = load_vec(args.tensor)                       # [k, k, n_layers, d]
    k_ai, k_user, n_layers, d = T.shape
    if k_ai != k_user:
        print(f"  [warn] T is {k_ai}x{k_user}; slot-swap asymmetry assumes a SQUARE, "
              f"same-ordered tensor (use the AI+AI tensor with identical trait ordering).")
    k = min(k_ai, k_user)
    Tk = T[:k, :k]
    rows = []
    for l in range(n_layers):
        Tl = Tk[:, :, l, :]                          # [k, k, d]
        Tsw = Tl.transpose(0, 1)
        asym = (Tl - Tsw).norm() / (Tl.norm() + 1e-8)
        # per-trait localization: how asymmetric is trait i across its row vs column
        per_trait = {f"asym_trait{i}": float((Tl[i] - Tl[:, i]).norm()) for i in range(k)}
        rows.append(dict(layer=l, asym=float(asym), **per_trait))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    amax = max(rows, key=lambda r: r["asym"])
    print(f"  peak slot-asymmetry {amax['asym']:.3f} at layer {amax['layer']} -> {out}")


# ───────────────────────── refusal_map ─────────────────────────

def cmd_refusal_map(args):
    T = load_vec(args.tensor)                       # [k_ai, k_user, n_layers, d]
    d_ref = load_vec(args.refusal)                  # [n_layers, d]
    man = json.loads(Path(args.tensor + ".manifest.json").read_text()) \
        if Path(args.tensor + ".manifest.json").exists() else None
    k_ai, k_user, n_layers, d = T.shape
    L = args.layer if args.layer >= 0 else n_layers - 1
    u = unit(d_ref[L])                               # [d]
    S = torch.einsum("ijd,d->ij", T[:, :, L, :], u)  # [k_ai, k_user]
    ai = man["ai_traits"] if man else [f"ai{i}" for i in range(k_ai)]
    user = man["user_traits"] if man else [f"u{j}" for j in range(k_user)]
    df = pd.DataFrame(S.numpy(), index=ai, columns=user)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"  refusal routing S[i,j] = <T[i,j], d_refusal> at layer {L}:")
    print(df.round(3).to_string())
    print(f"  -> {out}")
    print("  prediction: evil-row entries large(+) in INSTRUCT, ~0 in BASE.")


# ───────────────────────── delta (instruct - base) ─────────────────────────

def cmd_delta(args):
    Ti = load_vec(args.tensor_instruct)
    Tb = load_vec(args.tensor_base)
    if Ti.shape != Tb.shape:
        raise ValueError(f"shape mismatch: instruct {tuple(Ti.shape)} vs base {tuple(Tb.shape)}")
    dT = Ti - Tb
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dT, str(out))
    k_ai, k_user, n_layers, d = dT.shape
    # decomposition energy of delta vs base (is the *change* low rank?)
    rows = []
    for l in range(n_layers):
        Md = dT[:, :, l, :].reshape(k_ai * k_user, d)
        Sd = torch.linalg.svdvals(Md)
        row = dict(layer=l, delta_eff_rank=effective_rank(Sd),
                   delta_norm=float(Md.norm()))
        if args.refusal:
            d_ref = load_vec(args.refusal)
            u = unit(d_ref[l])
            Sij = torch.einsum("ijd,d->ij", dT[:, :, l, :], u)
            row["delta_refusal_meanabs"] = float(Sij.abs().mean())
        rows.append(row)
    csv = Path(str(out).replace(".pt", "") + "_summary.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"  saved Delta_T {tuple(dT.shape)} -> {out}")
    print(f"  summary (eff_rank / norm / refusal-proj per layer) -> {csv}")
    print("  claim: Delta_T is low-rank and refusal-aligned on evil pairs (what RLHF installs).")


# ───────────────────────── predict (composite) ─────────────────────────

def cmd_predict(args):
    """Predict R for a composite AI persona vs a fixed user trait, compare to measured.

    R_pred[l] = sum_i w_i * T[i, j, l, :]   (bilinear model, fixed user trait j)
    """
    T = load_vec(args.tensor)                       # [k_ai, k_user, n_layers, d]
    man = json.loads(Path(args.tensor + ".manifest.json").read_text())
    ai_traits = man["ai_traits"]; user_traits = man["user_traits"]
    j = user_traits.index(args.user_trait)
    # weights spec: "humour=1,evil=1"
    w = torch.zeros(len(ai_traits))
    for tok in args.weights.split(","):
        name, val = tok.split("=")
        w[ai_traits.index(name.strip())] = float(val)
    R_pred = torch.einsum("ild,i->ld", T[:, j, :, :], w)   # [n_layers, d]
    R_meas = load_vec(args.measured)                        # [n_layers, d]
    n = min(R_pred.shape[0], R_meas.shape[0])
    cos = cos_per_layer(R_pred[:n], R_meas[:n])
    rel_err = (R_pred[:n] - R_meas[:n]).norm(dim=-1) / (R_meas[:n].norm(dim=-1) + 1e-8)
    df = pd.DataFrame(dict(layer=range(n),
                           cos_pred_meas=cos.numpy(),
                           rel_err=rel_err.numpy(),
                           norm_pred=R_pred[:n].norm(dim=-1).numpy(),
                           norm_meas=R_meas[:n].norm(dim=-1).numpy()))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  composite '{args.weights}' vs user '{args.user_trait}':")
    print(f"    mean cos(pred,meas)={float(cos.mean()):.3f}  "
          f"mean rel_err={float(rel_err.mean()):.3f}  (high cos + low err => bilinearity holds)")
    print(f"  -> {out}")


# ───────────────────────── behavioral_law ─────────────────────────

def cmd_behavioral_law(args):
    """Correlate realized alpha(c) with judge behavior from an ablation_study.py CSV.

    Under additive steering h += c * R_hat (amplify_user mode with --user_vec = R), the
    effective residue scales with c, so realized alpha(A|U) is affine in c:
        alpha(c) = 1 + c * (alpha0 - 1),   alpha0 = 1 + <R, tau_A>/||tau_A||^2  (at --layer)
    We read (coeff, judge_score) from the CSV, compute alpha(c), and report correlation.
    If a 'recipe' column is present (different steering vectors targeting matched alpha),
    we overlay them to test sufficiency (do behavior-vs-alpha curves collapse?).
    """
    df = pd.read_csv(args.csv)
    R = load_vec(args.residue); tau_A = load_vec(args.ai_vec)
    L = args.layer if args.layer >= 0 else R.shape[0] - 1
    alpha0 = 1.0 + float(torch.dot(R[L], tau_A[L]) / (tau_A[L].norm() ** 2 + 1e-8))
    score_col = args.score_col
    coeff_col = args.coeff_col
    g = df.groupby(coeff_col)[score_col].mean().reset_index()
    g["alpha"] = 1.0 + g[coeff_col] * (alpha0 - 1.0)
    corr = float(np.corrcoef(g["alpha"], g[score_col])[0, 1]) if len(g) > 2 else float("nan")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False)
    print(f"  alpha0 (layer {L}) = {alpha0:.3f}")
    print(f"  corr(alpha(c), {score_col}) = {corr:.3f}  over {len(g)} coeff points")
    print(g.round(3).to_string(index=False))
    if "recipe" in df.columns:
        print("\n  sufficiency overlay (behavior vs alpha by recipe):")
        for rec, sub in df.groupby("recipe"):
            gg = sub.groupby(coeff_col)[score_col].mean()
            print(f"    {rec}: " + " ".join(f"c={c}:{v:.1f}" for c, v in gg.items()))
        print("  sufficiency holds if curves collapse onto one alpha->behavior mapping.")
    print(f"  -> {out}")


# ───────────────────────── CLI ─────────────────────────

def main():
    p = argparse.ArgumentParser(description="Second-order persona algebra: interaction tensor engine.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble", help="Build T from per-pair factorial cells.")
    a.add_argument("--vec_dir", required=True)
    a.add_argument("--ai_traits", required=True, help="Comma list, e.g. humour,evil,formal")
    a.add_argument("--user_traits", required=True, help="Comma list, e.g. evil,anxious,polite")
    a.add_argument("--pair_prefix", default="fac_{ai}_{user}",
                   help="Cell filename prefix template; cells are <prefix>_{pp,pn,np,nn}.")
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_assemble)

    d = sub.add_parser("decompose", help="Per-layer HOSVD of T: modes + effective rank.")
    d.add_argument("--tensor", required=True)
    d.add_argument("--rank", type=int, default=4)
    d.add_argument("--outdir", required=True)
    d.set_defaults(func=cmd_decompose)

    s = sub.add_parser("asymmetry", help="Slot-swap asymmetry (use the square AI+AI tensor).")
    s.add_argument("--tensor", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_asymmetry)

    rm = sub.add_parser("refusal_map", help="S[i,j] = <T[i,j], d_refusal>.")
    rm.add_argument("--tensor", required=True)
    rm.add_argument("--refusal", required=True)
    rm.add_argument("--layer", type=int, default=-1)
    rm.add_argument("--out", required=True)
    rm.set_defaults(func=cmd_refusal_map)

    dl = sub.add_parser("delta", help="Delta_T = T_instruct - T_base + summary.")
    dl.add_argument("--tensor_instruct", required=True)
    dl.add_argument("--tensor_base", required=True)
    dl.add_argument("--refusal", default="")
    dl.add_argument("--out", required=True)
    dl.set_defaults(func=cmd_delta)

    pr = sub.add_parser("predict", help="Held-out composite prediction (bilinearity test).")
    pr.add_argument("--tensor", required=True)
    pr.add_argument("--user_trait", required=True)
    pr.add_argument("--weights", required=True, help='e.g. "humour=1,evil=1"')
    pr.add_argument("--measured", required=True, help="Measured composite residue .pt")
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_predict)

    bl = sub.add_parser("behavioral_law", help="Correlate realized alpha(c) with judge score.")
    bl.add_argument("--csv", required=True, help="ablation_study.py output CSV.")
    bl.add_argument("--residue", required=True)
    bl.add_argument("--ai_vec", required=True)
    bl.add_argument("--layer", type=int, default=-1)
    bl.add_argument("--coeff_col", default="coeff")
    bl.add_argument("--score_col", default="score_evil")
    bl.add_argument("--out", required=True)
    bl.set_defaults(func=cmd_behavioral_law)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
