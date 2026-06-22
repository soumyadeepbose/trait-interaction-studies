#!/usr/bin/env python3
"""
make_plots.py — full plot set for one factorial setting (MERGED driver).

Two sources, ONE figure per chart (no subplot grids — they render at weird sizes
and skew visual comparison):

  (A) Vector-geometry plots  -> plotting_utils.py (single-axes / single plotly fig each).
  (B) Per-layer metric plots -> defined here, CSV-driven, with bootstrap-CI bands and
      permutation nulls that the plotting_utils versions don't have.

We deliberately DO NOT call plotting_utils.save_dashboard /
save_plot_grid_dashboard / save_projection_norms_plot / save_rho_eff_plot /
save_activation_projection_plot — those are multi-subplot grids. Their content is
covered here as individual single-axes figures instead.

Sibling metric CSVs (bootstrap_ci, permutation_null, effrank) are auto-derived from
--metrics_csv by swapping the "per_layer_metrics" stem, so the OLD invocation still
works unchanged; new CSVs are picked up if present.

Usage (unchanged from before, plus optional extras):
    python make_plots.py --vec_dir <dir> --fac_prefix <prefix>_fac \
        --metrics_csv <...>_per_layer_metrics.csv --model_short <m> \
        --base_trait <name> --out_dir output/plots/<name> \
        [--refusal_csv <...>] [--leakage_csv <...>] \
        [--base_metrics_csv <...>] [--base_leakage_csv <...>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── single-axes figure helpers ──────────────────────────────────────────────────
W, H, DPI = 8.0, 4.8, 140
C = dict(A="#c0392b", U="#2e86c1", R="#8e44ad", null="#7f8c8d", g="#27ae60")


def _fig():
    fig, ax = plt.subplots(figsize=(W, H))   # exactly one axes, always
    ax.grid(alpha=0.25)
    return fig, ax


def _save(fig, ax, out_dir, name, xlabel="layer", ylabel="", title=""):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    p = Path(out_dir) / f"{name}.png"
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"  [ok] {name} -> {p}")


def _read(path):
    return pd.read_csv(path) if path and Path(path).exists() else None


def _sibling(metrics_csv, stem):
    """Derive e.g. *_bootstrap_ci.csv from *_per_layer_metrics.csv."""
    s = str(metrics_csv)
    if "per_layer_metrics" in s:
        return s.replace("per_layer_metrics", stem)
    return ""


def _band(ax, x, lo, hi, color):
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)


# ── metric plots (single axes each) ─────────────────────────────────────────────

def plot_alpha(m, out, boot=None, base=None):
    fig, ax = _fig(); L = m["layer"]
    ax.axhline(1.0, color="k", ls=":", lw=1, label="α = 1 (no interaction)")
    ax.plot(L, m["alpha_A_given_U"], "o-", color=C["A"], lw=2, ms=3, label=r"$\alpha(A\,|\,U)$")
    ax.plot(L, m["alpha_U_given_A"], "s-", color=C["U"], lw=2, ms=3, label=r"$\alpha(U\,|\,A)$")
    if boot is not None:
        _band(ax, boot["layer"], boot["alpha_A_given_U_lo"], boot["alpha_A_given_U_hi"], C["A"])
        _band(ax, boot["layer"], boot["alpha_U_given_A_lo"], boot["alpha_U_given_A_hi"], C["U"])
    if base is not None:
        ax.plot(base["layer"], base["alpha_A_given_U"], "o--", color=C["A"], lw=1.2, ms=2,
                alpha=0.5, label=r"$\alpha(A\,|\,U)$ base")
        ax.plot(base["layer"], base["alpha_U_given_A"], "s--", color=C["U"], lw=1.2, ms=2,
                alpha=0.5, label=r"$\alpha(U\,|\,A)$ base")
    _save(fig, ax, out, "alpha_vs_layer", ylabel="amplification score",
          title="Amplification (interaction): <1 = suppression of that slot's trait")


def plot_cos(m, out, boot=None):
    fig, ax = _fig(); L = m["layer"]
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.plot(L, m["cos_R_vs_tau_A"], "o-", color=C["A"], lw=2, ms=3, label=r"$\cos(R,\tau_A)$")
    ax.plot(L, m["cos_R_vs_tau_U"], "s-", color=C["U"], lw=2, ms=3, label=r"$\cos(R,\tau_U)$")
    if boot is not None:
        _band(ax, boot["layer"], boot["cos_R_vs_tau_A_lo"], boot["cos_R_vs_tau_A_hi"], C["A"])
        _band(ax, boot["layer"], boot["cos_R_vs_tau_U_lo"], boot["cos_R_vs_tau_U_hi"], C["U"])
    _save(fig, ax, out, "cos_R_vs_tau", ylabel="cosine",
          title="Residue alignment with the marginal trait axes")


def plot_ncf(m, out, perm=None, boot=None):
    fig, ax = _fig(); L = m["layer"]
    ax.plot(L, m["ncf"], "o-", color=C["R"], lw=2, ms=3, label="NCF (observed)")
    if boot is not None:
        _band(ax, boot["layer"], boot["ncf_lo"], boot["ncf_hi"], C["R"])
    if perm is not None:
        ax.plot(perm["layer"], perm["ncf_null_mean"], "--", color=C["null"], lw=1.5,
                label="random-direction null")
    ax.set_ylim(0, 1.02)
    _save(fig, ax, out, "ncf_vs_layer", ylabel="novel component fraction",
          title="NCF vs noise floor (below null = genuine in-span mass)")


def plot_norms(m, out):
    fig, ax = _fig(); L = m["layer"]
    ax.plot(L, m["norm_tau_A"], "o-", color=C["A"], lw=2, ms=3, label=r"$\|\tau_A\|$")
    ax.plot(L, m["norm_tau_U"], "s-", color=C["U"], lw=2, ms=3, label=r"$\|\tau_U\|$")
    ax.plot(L, m["norm_tau_AU"], "^-", color=C["R"], lw=2, ms=3, label=r"$\|R\|$")
    _save(fig, ax, out, "norms_vs_layer", ylabel="L2 norm", title="Marginal and interaction magnitudes")


def plot_refusal(r, out):
    fig, ax = _fig(); L = r["layer"]
    ax.axhline(0.0, color="k", ls=":", lw=1)
    for col, lab, c in [("cos_R_refusal", r"$\cos(R,d_{ref})$", C["R"]),
                        ("cos_tauA_refusal", r"$\cos(\tau_A,d_{ref})$", C["A"]),
                        ("cos_tauU_refusal", r"$\cos(\tau_U,d_{ref})$", C["U"])]:
        if col in r.columns:
            ax.plot(L, r[col], "o-", lw=1.8, ms=3, color=c, label=lab)
    ax.set_ylim(-0.3, 0.3)
    _save(fig, ax, out, "refusal_cosines", ylabel="cosine",
          title="Alignment with the (DIM) refusal direction")


def plot_effrank(er, out):
    fig, ax = _fig()
    ax.plot(er["layer"], er["effective_rank"], "o-", color=C["R"], lw=2, ms=3)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="rank 1 (single direction)")
    _save(fig, ax, out, "effrank_vs_layer", ylabel="effective rank",
          title="Per-prompt residue dimensionality (>1 motivates subspace view)")


def plot_leakage(lk, out, base=None):
    fig, ax = _fig(); L = lk["layer"]
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.plot(L, lk["cos_leak"], "o-", color=C["A"], lw=2, ms=3, label="cos (directional)")
    ax.plot(L, lk["gamma_leak"], "s-", color=C["g"], lw=2, ms=3, label=r"$\gamma$ (gain)")
    if base is not None:
        ax.plot(base["layer"], base["cos_leak"], "o--", color=C["A"], lw=1.2, ms=2, alpha=0.5,
                label="cos base")
        ax.plot(base["layer"], base["gamma_leak"], "s--", color=C["g"], lw=1.2, ms=2, alpha=0.5,
                label=r"$\gamma$ base")
    _save(fig, ax, out, "trait_leakage", ylabel="cross-slot alignment / gain",
          title="Trait leakage: other-slot trait projected on own-slot axis")


# ── driver ───────────────────────────────────────────────────────────────────────

def load(vec_dir, name):
    import torch
    return torch.load(f"{vec_dir}/{name}_response_avg_diff.pt",
                      map_location="cpu", weights_only=False).float()


def run(label, fn, *a, **k):
    try:
        out = fn(*a, **k); print(f"  [ok] {label} -> {out}"); return out
    except Exception as e:
        print(f"  [skip] {label}: {type(e).__name__}: {e}"); return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vec_dir", required=True)
    p.add_argument("--fac_prefix", required=True, help="e.g. fac_humor_evil_user_fac")
    p.add_argument("--metrics_csv", required=True)
    p.add_argument("--model_short", required=True)
    p.add_argument("--base_trait", required=True, help="label for filenames")
    p.add_argument("--out_dir", default="output/plots")
    p.add_argument("--refusal_csv", default="")
    p.add_argument("--leakage_csv", default="")
    p.add_argument("--base_metrics_csv", default="")
    p.add_argument("--base_leakage_csv", default="")
    args = p.parse_args()

    ms, bt, od = args.model_short, args.base_trait, args.out_dir
    Path(od).mkdir(parents=True, exist_ok=True)

    # ── (A) Vector-geometry plots (plotting_utils; single-axes / single fig each) ──
    # Skips cleanly if torch/vectors/plotly are unavailable; metric plots still run.
    print("Vector-geometry plots:")
    try:
        import plotting_utils as P
        a_vec = load(args.vec_dir, f"{args.fac_prefix}_tauA")
        u_vec = load(args.vec_dir, f"{args.fac_prefix}_tauU")
        j_vec = load(args.vec_dir, f"{args.fac_prefix}_joint")
        residue = j_vec - a_vec - u_vec
        df_proj = P.build_dual_projection_df(u_vec, a_vec, residue)
        run("dual_projection", P.save_dual_projection_plot, df_proj, ms, bt, od)
        run("similarity_tile", P.save_similarity_tile_plot, u_vec, a_vec, ms, bt, od)
        run("norms_comparison", P.save_norms_comparison_plot, u_vec, a_vec, j_vec, residue, ms, bt, od)
        run("residue_cosine_heatmap", P.save_residue_cosine_heatmap, u_vec, a_vec, j_vec, residue, ms, bt, od)
        run("synergy_divergence", P.save_synergy_divergence_plot, u_vec, a_vec, j_vec, ms, bt, od)
        run("pca_scree", P.save_pca_scree_plot, u_vec, a_vec, j_vec, residue, ms, bt, od)
        run("pca_2d", P.save_pca_2d_plot, u_vec, a_vec, j_vec, residue, ms, bt, od)
        run("pca_3d", P.save_pca_3d_plot, u_vec, a_vec, j_vec, residue, ms, bt, od)
        run("interaction_dynamics", P.save_interaction_dynamics_plot, u_vec, a_vec, j_vec, ms, bt, od)
    except Exception as e:
        print(f"  [skip geometry block] {type(e).__name__}: {e}")
    # NOTE: save_dashboard intentionally omitted (multi-subplot grid). ESVA/GBC from
    # plotting_utils are single-axes; call them if you want, e.g.:
    #   run("esva", P.save_esva_plot, <esva_csv>, ms, bt, od)

    # ── (B) Per-layer metric plots (CSV-driven, single-axes, with CIs/nulls) ──────
    print("Metric plots (single-axes, from CSVs):")
    m = _read(args.metrics_csv)
    if m is not None:
        boot = _read(_sibling(args.metrics_csv, "bootstrap_ci"))
        perm = _read(_sibling(args.metrics_csv, "permutation_null"))
        er   = _read(_sibling(args.metrics_csv, "effrank"))
        base = _read(args.base_metrics_csv)
        plot_alpha(m, od, boot=boot, base=base)
        plot_cos(m, od, boot=boot)
        plot_ncf(m, od, perm=perm, boot=boot)
        plot_norms(m, od)
        if er is not None:
            plot_effrank(er, od)
    else:
        print(f"  [skip] metrics_csv not found: {args.metrics_csv}")

    rc = _read(args.refusal_csv)
    if rc is not None:
        plot_refusal(rc, od)
    lk = _read(args.leakage_csv)
    if lk is not None:
        plot_leakage(lk, od, base=_read(args.base_leakage_csv))

    print(f"\nDone. Plots in {od}/")


if __name__ == "__main__":
    main()