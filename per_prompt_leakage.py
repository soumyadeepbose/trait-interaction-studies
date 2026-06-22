#!/usr/bin/env python3
"""
per_prompt_leakage.py

Answers the key question: is the trait main-effect (leakage channel) low-rank
and consistent across prompts, while the interaction residue is high-rank and
idiosyncratic?

Computes from the saved per-prompt pos activation matrices:

  1. EffRank(τ_U^A), EffRank(τ_A^B), EffRank(R^A), EffRank(R^B) per layer
     — the definitive tau-vs-interaction dimensionality comparison.

  2. Bootstrap CIs on γ (leakage gain: τ_U from Setting A onto τ_A from Setting B)
     and cos_leak per layer.

  3. Per-prompt leakage distribution per layer:
       - Per-prompt γ_x = ⟨τ_U_x[l], τ̂_A_mean[l]⟩
         (project each prompt's user-evil direction onto mean AI-evil direction)
       - Mean, std, 5th/95th percentile, sign-consistency fraction
       — answers: is the leakage consistent per prompt or just a mean artefact?

Outputs: CSVs + one-figure-per-chart PNG plots (no subplot grids).

Usage:
    python per_prompt_leakage.py \\
        --vec_dir persona_steering/persona_vectors/Qwen2.5-7B-Instruct \\
        --setting_a_prefix fac_humor_evil_user \\
        --setting_b_prefix fac_evil_ai_humor_user \\
        --out_dir output/leakage_per_prompt/Qwen2.5-7B-Instruct \\
        [--base_vec_dir persona_steering/persona_vectors/Qwen2.5-7B] \\
        [--n_boot 2000] [--layer_lo 1] [--layer_hi 27]

    # the pos activation matrices live alongside the mean-diff vectors:
    # {vec_dir}/{prefix}_{cell}_pos_activation_matrix.pt   shape [N, n_layers, d]
    # {vec_dir}/{prefix}_fac_per_prompt_residue.pt         shape [N, n_layers, d]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── helpers ──────────────────────────────────────────────────────────────────

def load_cell_matrix(vec_dir: str, prefix: str, cell: str) -> torch.Tensor:
    """Load [N, n_layers, d] pos activation matrix for one factorial cell."""
    path = Path(vec_dir) / f"{prefix}_{cell}_pos_activation_matrix.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run generate_vec.py with --save_activations for this cell, or "
            "download the HF dataset artifact."
        )
    return torch.load(path, map_location="cpu", weights_only=False).float()


def load_per_prompt_residue(vec_dir: str, prefix: str) -> torch.Tensor | None:
    """Load [N, n_layers, d] per-prompt interaction residue (optional)."""
    path = Path(vec_dir) / f"{prefix}_fac_per_prompt_residue.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False).float()
    return None


def compute_per_prompt_tau(
    pp: torch.Tensor, pn: torch.Tensor,
    np_: torch.Tensor, nn: torch.Tensor,
    which: str,
) -> torch.Tensor:
    """
    Compute per-prompt marginal from the 4-cell factorial pos matrices.
    All inputs: [N, n_layers, d].

    which = 'A': τ_A (AI-slot trait main effect)
        = 0.5 * [(pp - np) + (pn - nn)]
    which = 'U': τ_U (user-slot trait main effect)
        = 0.5 * [(pp - pn) + (np - nn)]
    """
    if which == "A":
        return 0.5 * ((pp - np_) + (pn - nn))   # [N, n_layers, d]
    elif which == "U":
        return 0.5 * ((pp - pn) + (np_ - nn))
    raise ValueError(f"which must be 'A' or 'U', got {which!r}")


def effective_rank(mat: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Roy & Cover effective rank: exp(H(p)) where p_i ∝ σ_i.
    mat: [N, d] (one layer's per-prompt activations).
    """
    _, s, _ = torch.linalg.svd(mat, full_matrices=False)
    s = s[s > eps]
    p = s / s.sum()
    h = -(p * p.log()).sum()
    return float(h.exp())


def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm() + eps)


# ── core computations ─────────────────────────────────────────────────────────

def compute_effrank_profile(tau: torch.Tensor, lo: int, hi: int) -> dict:
    """tau: [N, n_layers, d] → per-layer effective rank."""
    return {l: effective_rank(tau[:, l, :]) for l in range(lo, hi + 1)}


def compute_leakage_bootstrap(
    tau_U_A: torch.Tensor,  # [N_A, n_layers, d]
    tau_A_B: torch.Tensor,  # [N_B, n_layers, d]
    lo: int, hi: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict]:
    """Bootstrap CIs on cos_leak and gamma_leak.

    Since τ_U^A and τ_A^B come from different runs (different prompt sets),
    we bootstrap each independently: resample rows of each run, compute
    mean vectors, then compute the overlap.
    """
    rng = np.random.default_rng(seed)
    N_A, N_B = tau_U_A.shape[0], tau_A_B.shape[0]
    rows = []

    for l in range(lo, hi + 1):
        source = tau_U_A[:, l, :]  # [N_A, d]
        target = tau_A_B[:, l, :]  # [N_B, d]

        mean_s = source.mean(0)
        mean_t = target.mean(0)
        norm_t = float(mean_t.norm()) + 1e-8
        norm_s = float(mean_s.norm()) + 1e-8
        cos_obs = float(torch.dot(mean_s, mean_t)) / (norm_s * norm_t)
        gamma_obs = float(torch.dot(mean_s, mean_t)) / (norm_t ** 2)

        cos_boot = np.empty(n_boot, dtype=np.float32)
        gam_boot = np.empty(n_boot, dtype=np.float32)
        for b in range(n_boot):
            idx_a = rng.integers(0, N_A, N_A)
            idx_b = rng.integers(0, N_B, N_B)
            ms = source[idx_a].mean(0)
            mt = target[idx_b].mean(0)
            nt = float(mt.norm()) + 1e-8
            ns = float(ms.norm()) + 1e-8
            cos_boot[b] = float(torch.dot(ms, mt)) / (ns * nt)
            gam_boot[b] = float(torch.dot(ms, mt)) / (nt ** 2)

        rows.append({
            "layer": l,
            "cos_leak": cos_obs,
            "cos_lo": float(np.percentile(cos_boot, 2.5)),
            "cos_hi": float(np.percentile(cos_boot, 97.5)),
            "gamma_leak": gamma_obs,
            "gamma_lo": float(np.percentile(gam_boot, 2.5)),
            "gamma_hi": float(np.percentile(gam_boot, 97.5)),
        })
    return rows


def compute_per_prompt_distribution(
    tau_U_A: torch.Tensor,  # [N_A, n_layers, d]
    tau_A_B_mean: torch.Tensor,  # [n_layers, d] — mean AI-evil axis (reference)
    lo: int, hi: int,
) -> list[dict]:
    """Per-prompt γ_x = ⟨τ_U_A[x,l], τ̂_A_B_mean[l]⟩ distribution per layer.

    Tests whether each individual prompt's user-evil direction points along
    the mean AI-evil axis, or whether the mean leakage is a cancellation of
    opposing prompt-level components.
    """
    rows = []
    for l in range(lo, hi + 1):
        t_hat = unit(tau_A_B_mean[l])            # unit mean AI-evil axis
        # per-prompt scalar projection (unnormalised by ||τ_U||, i.e. signed component)
        projs = (tau_U_A[:, l, :] @ t_hat).numpy()   # [N_A]
        sign_frac = float((np.sign(projs) == np.sign(projs.mean())).mean())
        rows.append({
            "layer": l,
            "gamma_mean": float(projs.mean()),
            "gamma_std":  float(projs.std()),
            "gamma_p05":  float(np.percentile(projs, 5)),
            "gamma_p95":  float(np.percentile(projs, 95)),
            "sign_consistency": sign_frac,
            "n_prompts": len(projs),
        })
    return rows


def compute_leakage_baseline(
    tau_U_A: torch.Tensor,  # [N_A, n_layers, d] — instruct
    tau_A_B: torch.Tensor,  # [N_B, n_layers, d] — instruct
    tau_U_A_base: torch.Tensor,  # [N_A, n_layers, d] — base
    tau_A_B_base: torch.Tensor,  # [N_B, n_layers, d] — base
    lo: int, hi: int,
) -> list[dict]:
    """Instruct vs base mean leakage (cos + gamma) per layer, no bootstrap."""
    rows = []
    for l in range(lo, hi + 1):
        def _cg(s, t):
            ms, mt = s[:, l, :].mean(0), t[:, l, :].mean(0)
            nt = float(mt.norm()) + 1e-8
            ns = float(ms.norm()) + 1e-8
            return float(torch.dot(ms, mt)) / (ns * nt), float(torch.dot(ms, mt)) / (nt**2)
        cos_i, gam_i = _cg(tau_U_A, tau_A_B)
        cos_b, gam_b = _cg(tau_U_A_base, tau_A_B_base)
        rows.append({
            "layer": l,
            "cos_instruct": cos_i, "gamma_instruct": gam_i,
            "cos_base": cos_b,     "gamma_base": gam_b,
            "delta_cos": cos_i - cos_b, "delta_gamma": gam_i - gam_b,
        })
    return rows


# ── plotting helpers ───────────────────────────────────────────────────────────

W, H, DPI = 8.0, 4.8, 140
C = dict(tauU="#2e86c1", tauA="#c0392b", R_A="#8e44ad", R_B="#e67e22",
         base="#95a5a6", instruct="#c0392b", null="#7f8c8d", g="#27ae60")


def _fig():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.grid(alpha=0.25)
    return fig, ax


def _save(fig, ax, out, name, xlabel="layer", ylabel="", title=""):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    p = Path(out) / f"{name}.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"  wrote {p}")


def _band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


def plot_effrank_comparison(er_df: pd.DataFrame, out: str, er_R_A=None, er_R_B=None):
    """One figure: EffRank of τ_U^A, τ_A^B, R^A, R^B vs layer."""
    fig, ax = _fig()
    L = er_df["layer"]
    ax.plot(L, er_df["effrank_tauU_A"], "o-", color=C["tauU"], lw=2, ms=3,
            label=r"EffRank$(\tau_U^A)$ — user-evil (Setting A)")
    ax.plot(L, er_df["effrank_tauA_B"], "s-", color=C["tauA"], lw=2, ms=3,
            label=r"EffRank$(\tau_A^B)$ — AI-evil (Setting B)")
    if er_R_A is not None:
        ax.plot(L, er_df["effrank_R_A"], "^--", color=C["R_A"], lw=1.5, ms=3,
                label=r"EffRank$(R^A)$ — interaction Setting A")
    if er_R_B is not None:
        ax.plot(L, er_df["effrank_R_B"], "v--", color=C["R_B"], lw=1.5, ms=3,
                label=r"EffRank$(R^B)$ — interaction Setting B")
    _save(fig, ax, out, "effrank_comparison", ylabel="effective rank",
          title="Trait directions vs interaction residue: dimensionality comparison")


def plot_leakage_bootstrap(boot_df: pd.DataFrame, out: str):
    """One figure: γ (leakage gain) with bootstrap CI band."""
    fig, ax = _fig()
    L = boot_df["layer"]
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.axhline(1, color="k", ls=":", lw=0.7, alpha=0.4)
    ax.plot(L, boot_df["gamma_leak"], "o-", color=C["tauU"], lw=2, ms=3,
            label=r"$\gamma$ (leakage gain)")
    _band(ax, L, boot_df["gamma_lo"], boot_df["gamma_hi"], C["tauU"])
    ax.plot(L, boot_df["cos_leak"], "s--", color=C["g"], lw=1.5, ms=3,
            label=r"$\cos$ (directional)")
    _band(ax, L, boot_df["cos_lo"], boot_df["cos_hi"], C["g"])
    _save(fig, ax, out, "leakage_bootstrap", ylabel="leakage (cos / γ)",
          title=r"Trait leakage $\tau_U^A \to \tau_A^B$ with 95% bootstrap CI")


def plot_per_prompt_distribution(dist_df: pd.DataFrame, out: str):
    """One figure: mean ± std per-prompt leakage projection with sign-consistency."""
    fig, ax = _fig()
    L = dist_df["layer"]
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.plot(L, dist_df["gamma_mean"], "o-", color=C["tauU"], lw=2, ms=3,
            label="mean per-prompt γ")
    _band(ax, L,
          dist_df["gamma_mean"] - dist_df["gamma_std"],
          dist_df["gamma_mean"] + dist_df["gamma_std"],
          C["tauU"])
    ax.plot(L, dist_df["gamma_p05"], "--", color=C["null"], lw=1, label="5th / 95th pct")
    ax.plot(L, dist_df["gamma_p95"], "--", color=C["null"], lw=1)
    _save(fig, ax, out, "per_prompt_leakage_distribution",
          ylabel="per-prompt γ (signed projection)",
          title="Per-prompt leakage: mean ± std across prompts (shaded = ±1σ)")


def plot_sign_consistency(dist_df: pd.DataFrame, out: str):
    """One figure: fraction of prompts with consistent leakage sign per layer."""
    fig, ax = _fig()
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance (0.5)")
    ax.plot(dist_df["layer"], dist_df["sign_consistency"], "o-",
            color=C["tauU"], lw=2, ms=3)
    ax.set_ylim(0.4, 1.02)
    _save(fig, ax, out, "leakage_sign_consistency",
          ylabel="fraction of prompts with consistent sign",
          title="Per-prompt leakage sign-consistency (1.0 = every prompt leans the same way)")


def plot_base_vs_instruct(base_df: pd.DataFrame, out: str):
    """One figure: instruct vs base leakage (γ and cos) with delta."""
    fig, ax = _fig()
    L = base_df["layer"]
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.plot(L, base_df["gamma_instruct"], "o-", color=C["instruct"], lw=2, ms=3,
            label=r"$\gamma$ instruct")
    ax.plot(L, base_df["gamma_base"], "o--", color=C["base"], lw=1.5, ms=3,
            label=r"$\gamma$ base")
    ax.plot(L, base_df["delta_gamma"], "^-", color=C["R_A"], lw=1.5, ms=3,
            label=r"$\Delta\gamma$ (instruct − base)")
    _save(fig, ax, out, "leakage_base_vs_instruct",
          ylabel=r"$\gamma$ (leakage gain)",
          title=r"RLHF effect on trait leakage: $\Delta\gamma$ = instruct $-$ base")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Per-prompt leakage: EffRank(τ) vs EffRank(R), bootstrap CIs on γ."
    )
    ap.add_argument("--vec_dir", required=True,
                    help="Dir with pos activation matrices and per_prompt_residue.pt "
                         "(e.g. persona_steering/persona_vectors/Qwen2.5-7B-Instruct)")
    ap.add_argument("--setting_a_prefix", default="fac_humor_evil_user",
                    help="Prefix for Setting A cells (humor-AI, evil-USER)")
    ap.add_argument("--setting_b_prefix", default="fac_evil_ai_humor_user",
                    help="Prefix for Setting B cells (evil-AI, humor-USER)")
    ap.add_argument("--base_vec_dir", default="",
                    help="Optional: base-model vec_dir for Δγ comparison")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--layer_lo", type=int, default=1)
    ap.add_argument("--layer_hi", type=int, default=27)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.layer_lo, args.layer_hi

    # ── Load Setting A (humor-AI, evil-USER) cell matrices ──────────────────
    print("Loading Setting A (humor-AI, evil-USER) matrices...")
    A = args.setting_a_prefix
    pp_A = load_cell_matrix(args.vec_dir, A, "pp")
    pn_A = load_cell_matrix(args.vec_dir, A, "pn")
    np_A = load_cell_matrix(args.vec_dir, A, "np")
    nn_A = load_cell_matrix(args.vec_dir, A, "nn")
    print(f"  cell shape: {pp_A.shape}  (N × n_layers × d)")

    tau_U_A = compute_per_prompt_tau(pp_A, pn_A, np_A, nn_A, "U")  # per-prompt user-evil
    tau_A_A = compute_per_prompt_tau(pp_A, pn_A, np_A, nn_A, "A")  # per-prompt AI-humor
    del pp_A, pn_A, np_A, nn_A  # free memory
    R_A = load_per_prompt_residue(args.vec_dir, A)

    # ── Load Setting B (evil-AI, humor-USER) cell matrices ──────────────────
    print("Loading Setting B (evil-AI, humor-USER) matrices...")
    B = args.setting_b_prefix
    pp_B = load_cell_matrix(args.vec_dir, B, "pp")
    pn_B = load_cell_matrix(args.vec_dir, B, "pn")
    np_B = load_cell_matrix(args.vec_dir, B, "np")
    nn_B = load_cell_matrix(args.vec_dir, B, "nn")
    tau_A_B = compute_per_prompt_tau(pp_B, pn_B, np_B, nn_B, "A")  # per-prompt AI-evil
    tau_U_B = compute_per_prompt_tau(pp_B, pn_B, np_B, nn_B, "U")  # per-prompt user-humor
    del pp_B, pn_B, np_B, nn_B
    R_B = load_per_prompt_residue(args.vec_dir, B)

    # ── 1. EffRank: τ_U^A, τ_A^B, and optionally R^A, R^B ──────────────────
    print("Computing EffRank profiles...")
    er_rows = []
    for l in range(lo, hi + 1):
        row = {
            "layer": l,
            "effrank_tauU_A": effective_rank(tau_U_A[:, l, :]),
            "effrank_tauA_A": effective_rank(tau_A_A[:, l, :]),
            "effrank_tauA_B": effective_rank(tau_A_B[:, l, :]),
            "effrank_tauU_B": effective_rank(tau_U_B[:, l, :]),
        }
        if R_A is not None:
            row["effrank_R_A"] = effective_rank(R_A[:, l, :])
        if R_B is not None:
            row["effrank_R_B"] = effective_rank(R_B[:, l, :])
        er_rows.append(row)
    er_df = pd.DataFrame(er_rows)
    er_df.to_csv(out / "effrank_comparison.csv", index=False)
    print(f"  EffRank τ_U^A L20: {er_df.loc[er_df['layer']==20, 'effrank_tauU_A'].values[0]:.1f}")
    print(f"  EffRank τ_A^B L20: {er_df.loc[er_df['layer']==20, 'effrank_tauA_B'].values[0]:.1f}")
    if "effrank_R_B" in er_df:
        print(f"  EffRank R^B   L20: {er_df.loc[er_df['layer']==20, 'effrank_R_B'].values[0]:.1f}")
    plot_effrank_comparison(er_df, str(out),
                            er_R_A="effrank_R_A" in er_df.columns,
                            er_R_B="effrank_R_B" in er_df.columns)

    # ── 2. Bootstrap CIs on γ and cos_leak ──────────────────────────────────
    print(f"Bootstrapping leakage (n_boot={args.n_boot})...")
    boot_rows = compute_leakage_bootstrap(tau_U_A, tau_A_B, lo, hi, args.n_boot)
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out / "leakage_bootstrap.csv", index=False)
    print(f"  γ at L20: {boot_df.loc[boot_df['layer']==20, 'gamma_leak'].values[0]:.3f} "
          f"[{boot_df.loc[boot_df['layer']==20, 'gamma_lo'].values[0]:.3f}, "
          f"{boot_df.loc[boot_df['layer']==20, 'gamma_hi'].values[0]:.3f}]")
    plot_leakage_bootstrap(boot_df, str(out))

    # ── 3. Per-prompt leakage distribution ──────────────────────────────────
    print("Computing per-prompt leakage distribution...")
    tau_A_B_mean = tau_A_B.mean(0)  # [n_layers, d]
    dist_rows = compute_per_prompt_distribution(tau_U_A, tau_A_B_mean, lo, hi)
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out / "per_prompt_leakage_distribution.csv", index=False)
    print(f"  sign-consistency at L20: {dist_df.loc[dist_df['layer']==20, 'sign_consistency'].values[0]:.3f}")
    plot_per_prompt_distribution(dist_df, str(out))
    plot_sign_consistency(dist_df, str(out))

    # ── 4. Base vs instruct comparison (optional) ───────────────────────────
    if args.base_vec_dir:
        print("Loading base-model matrices for Δγ...")
        pp_Ab = load_cell_matrix(args.base_vec_dir, A, "pp")
        pn_Ab = load_cell_matrix(args.base_vec_dir, A, "pn")
        np_Ab = load_cell_matrix(args.base_vec_dir, A, "np")
        nn_Ab = load_cell_matrix(args.base_vec_dir, A, "nn")
        tau_U_A_base = compute_per_prompt_tau(pp_Ab, pn_Ab, np_Ab, nn_Ab, "U")
        del pp_Ab, pn_Ab, np_Ab, nn_Ab

        pp_Bb = load_cell_matrix(args.base_vec_dir, B, "pp")
        pn_Bb = load_cell_matrix(args.base_vec_dir, B, "pn")
        np_Bb = load_cell_matrix(args.base_vec_dir, B, "np")
        nn_Bb = load_cell_matrix(args.base_vec_dir, B, "nn")
        tau_A_B_base = compute_per_prompt_tau(pp_Bb, pn_Bb, np_Bb, nn_Bb, "A")
        del pp_Bb, pn_Bb, np_Bb, nn_Bb

        base_rows = compute_leakage_baseline(
            tau_U_A, tau_A_B, tau_U_A_base, tau_A_B_base, lo, hi
        )
        base_df = pd.DataFrame(base_rows)
        base_df.to_csv(out / "leakage_base_vs_instruct.csv", index=False)
        plot_base_vs_instruct(base_df, str(out))
        print(f"  Δγ at L20: {base_df.loc[base_df['layer']==20, 'delta_gamma'].values[0]:.3f}")

    print(f"\nDone. Outputs in {out}/")
    _print_summary(er_df, boot_df, dist_df)


def _print_summary(er_df, boot_df, dist_df):
    print("\n===== KEY NUMBERS (L20) =====")
    l = 20
    er = er_df[er_df["layer"] == l].iloc[0]
    b  = boot_df[boot_df["layer"] == l].iloc[0]
    d  = dist_df[dist_df["layer"] == l].iloc[0]
    print(f"  EffRank τ_U^A : {er['effrank_tauU_A']:.1f}")
    print(f"  EffRank τ_A^B : {er['effrank_tauA_B']:.1f}")
    if "effrank_R_B" in er:
        print(f"  EffRank R^B   : {er['effrank_R_B']:.1f}")
    print(f"  γ (leakage)   : {b['gamma_leak']:.3f}  [{b['gamma_lo']:.3f}, {b['gamma_hi']:.3f}]")
    print(f"  cos (leakage) : {b['cos_leak']:.3f}  [{b['cos_lo']:.3f}, {b['cos_hi']:.3f}]")
    print(f"  sign consist. : {d['sign_consistency']:.3f}  (n={d['n_prompts']})")
    print(f"  per-prompt std: {d['gamma_std']:.3f}")


if __name__ == "__main__":
    main()