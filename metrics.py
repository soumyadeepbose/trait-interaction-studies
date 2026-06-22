"""
metrics.py — Persona interaction metrics computation.
Version: 3 (April 2026)

Changes vs v2:
  1. Added span-projected amplification scores:
       alpha_A_given_U_span and alpha_U_given_A_span
     These use only the in-span component of tau_AU (i.e. the projection of
     tau_AU onto span{tau_A, tau_U}), which removes the NCF/noise component
     from the denominator-free amplification formula. This gives cleaner scores
     that are not contaminated by the orthogonal residue.

  2. Added --cross_trait_vec_dir and --cross_trait flags for computing
     layerwise cosine similarity between tau_U from one condition and tau_A from
     another. Specifically designed for the cos(tau_U_evil, tau_A_evil) diagnostic:
     if high at late layers, the evil-user instruction is geometrically pushing
     the AI toward the same region as an evil-AI instruction, suggesting the
     labeled evil user is partially inducing evil-AI behavior.

  3. Output files remain CSV (one row per layer). Added new columns:
       alpha_A_given_U_span, alpha_U_given_A_span  (in per_layer_metrics.csv)
     New optional output:
       <prefix>_cross_trait_cosine.csv

  4. rho_rel thresholds updated in summary (>0.05 significant, >0.01 moderate,
     <=0.01 negligible / ablation mechanically trivial).

Usage:
    python metrics.py \\
        --vec_dir persona_vectors/Qwen2.5-7B-Instruct \\
        --ai_trait orig_humorous \\
        --user_trait user_evil \\
        --joint_trait humorous_ai_user_evil \\
        --output_dir output/metrics \\
        --output_prefix Qwen25_7B_humorous_evil

    # Cross-trait cosine (e.g. cos(tau_U_evil, tau_A_evil)):
    python metrics.py \\
        --vec_dir persona_vectors/Qwen2.5-7B-Instruct \\
        --ai_trait orig_humorous --user_trait user_evil \\
        --joint_trait humorous_ai_user_evil \\
        --cross_trait_vec_dir persona_vectors/Qwen2.5-7B-Instruct \\
        --cross_trait_left user_evil \\
        --cross_trait_right orig_evil \\
        --output_dir output/metrics \\
        --output_prefix Qwen25_7B_humorous_evil

Optional inputs:
    --pos_activation_matrix   [n_prompts, n_layers, d] pos-side activations (ESVA)
    --neg_activation_matrix   [n_prompts, n_layers, d] neg-side activations (ESVA)
    --per_prompt_residue      [n_prompts, n_layers, d] (EffRank, GBC, projection)
    --scored_csv              CSV with 'harmfulness' column (GBC)
    --eval_hidden_states      [n_prompts, n_layers, d] eval-time hidden states (rho_eff)

All outputs are CSV files (one row per layer) plus a human-readable summary.txt.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_vec(path):
    v = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    return v.float()


def load_opt(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [optional] {p.name} not found — skipping.")
        return None
    t = torch.load(p, map_location="cpu", weights_only=False)
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    return t.float()


def safe_cos(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def eff_rank(matrix, eps=1e-12):
    c = matrix - matrix.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    s2 = s ** 2
    tot = s2.sum()
    if tot < eps:
        return 1.0
    lam = s2[s2 > eps] / tot
    return float(np.exp(-np.sum(lam * np.log(lam + 1e-30))))


def gram_schmidt_2(v1, v2, eps=1e-8):
    """Returns orthonormal basis (e1, e2) for span{v1, v2}."""
    e1 = v1 / (torch.linalg.vector_norm(v1) + eps)
    v2o = v2 - torch.dot(v2, e1) * e1
    n2 = torch.linalg.vector_norm(v2o)
    e2 = v2o / (n2 + eps) if n2 > eps else torch.zeros_like(e1)
    return e1, e2


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_per_layer(tau_A, tau_U, tau_AU):
    """
    Compute primary per-layer metrics. Returns a list of dicts (one per layer).

    Includes the standard amplification scores (alpha_A_given_U, alpha_U_given_A)
    computed from full tau_AU, PLUS span-projected versions that remove the NCF
    component:

      alpha_*_span: same formula but using tau_AU_span = proj_{span{tA,tU}}(tau_AU)
                    instead of tau_AU. This removes the orthogonal (NCF) component
                    from the amplification signal and gives cleaner geometry.

    eta_signed is NOT included — it is algebraically a weighted average of the
    cosine profiles and adds no new information.
    """
    n = tau_A.shape[0]
    eps = 1e-8
    rows = []

    for l in range(n):
        tA, tU, tAU = tau_A[l], tau_U[l], tau_AU[l]
        nA  = float(torch.linalg.vector_norm(tA).item())
        nU  = float(torch.linalg.vector_norm(tU).item())
        nAU = float(torch.linalg.vector_norm(tAU).item())

        cos_A = safe_cos(tAU, tA)
        cos_U = safe_cos(tAU, tU)
        theta_A = math.degrees(math.acos(max(-1.0, min(1.0, cos_A)))) if nAU > eps else 90.0
        theta_U = math.degrees(math.acos(max(-1.0, min(1.0, cos_U)))) if nAU > eps else 90.0

        # Standard amplification scores (use full tau_AU including NCF component)
        alpha_A = 1.0 + float(torch.dot(tAU, tA).item()) / (nA ** 2 + eps)
        alpha_U = 1.0 + float(torch.dot(tAU, tU).item()) / (nU ** 2 + eps)

        # Residue decomposition into span{tA, tU} and orthogonal complement
        e1, e2 = gram_schmidt_2(tA, tU)
        proj_A_coeff = torch.dot(tAU, e1)
        proj_U_coeff = torch.dot(tAU, e2)
        tau_AU_span = proj_A_coeff * e1 + proj_U_coeff * e2   # in-span component
        tau_AU_perp = tAU - tau_AU_span                        # NCF component

        n_proj  = float(torch.linalg.vector_norm(tau_AU_span).item())
        n_perp  = float(torch.linalg.vector_norm(tau_AU_perp).item())

        # Individual parallel components along tA and tU (non-orthonormal projection)
        comp_A = (torch.dot(tAU, tA) / (nA ** 2 + eps)) * tA
        comp_U = (torch.dot(tAU, tU) / (nU ** 2 + eps)) * tU

        # ── Span-projected amplification scores ──────────────────────────────
        # These use tau_AU_span instead of tau_AU. Since tau_AU_span is already
        # the component of tau_AU in span{tA, tU}, these scores are uncontaminated
        # by the orthogonal residue (NCF component).
        #
        # Formula unchanged: alpha_*_span = 1 + <tau_AU_span, tau_*> / ||tau_*||^2
        # The NCF orthogonality means <tau_AU_perp, tA> ≈ 0 and <tau_AU_perp, tU> ≈ 0,
        # so in practice alpha_*_span ≈ alpha_* unless tA and tU are non-orthogonal
        # (which they are in general, making the span projection non-trivial).
        alpha_A_span = 1.0 + float(torch.dot(tau_AU_span, tA).item()) / (nA ** 2 + eps)
        alpha_U_span = 1.0 + float(torch.dot(tau_AU_span, tU).item()) / (nU ** 2 + eps)

        nd_A = float(torch.linalg.vector_norm(tA + tAU).item())
        nd_U = float(torch.linalg.vector_norm(tU + tAU).item())

        rows.append({
            "layer":               l,
            "norm_tau_A":          nA,
            "norm_tau_U":          nU,
            "norm_tau_AU":         nAU,
            "eta":                 nAU / (nA + nU + eps),
            "cos_R_vs_tau_A":      cos_A,
            "cos_R_vs_tau_U":      cos_U,
            "theta_A_deg":         theta_A,
            "theta_U_deg":         theta_U,
            # Standard amplification (full tau_AU)
            "alpha_A_given_U":     alpha_A,
            "alpha_U_given_A":     alpha_U,
            # Span-projected amplification (NCF removed from tau_AU)
            "alpha_A_given_U_span": alpha_A_span,
            "alpha_U_given_A_span": alpha_U_span,
            # Decomposition norms
            "norm_R_parallel_A":   float(torch.linalg.vector_norm(comp_A).item()),
            "norm_R_parallel_U":   float(torch.linalg.vector_norm(comp_U).item()),
            "norm_R_in_span":      n_proj,
            "norm_R_perp":         n_perp,
            # NCF = ||R_perp||^2 / ||tau_AU||^2
            "ncf":                 (n_perp ** 2) / (nAU ** 2 + eps),
            # CAI and RNA baseline
            "cai":                 (nd_A - nd_U) / (nd_A + nd_U + eps),
            "rna":                 (nA  - nU)  / (nA  + nU  + eps),
        })
    return rows


def compute_cross_trait_cosine(vec_left, vec_right):
    """
    Compute layerwise cosine similarity between two steering vectors.
    Used for e.g. cos(tau_U_evil, tau_A_evil) to test whether the evil user
    instruction is pushing the AI toward the evil-AI geometric region.

    Args:
        vec_left:  [n_layers, d] tensor (e.g. tau_U from user_evil extraction)
        vec_right: [n_layers, d] tensor (e.g. tau_A from orig_evil extraction)

    Returns:
        list of dicts with 'layer' and 'cross_cosine'.
    """
    assert vec_left.shape == vec_right.shape, \
        f"Shape mismatch: {vec_left.shape} vs {vec_right.shape}"
    n = vec_left.shape[0]
    rows = []
    for l in range(n):
        rows.append({
            "layer": l,
            "cross_cosine": safe_cos(vec_left[l], vec_right[l]),
        })
    return rows


def compute_effrank(ppr):
    """EffRank of per-prompt residue per layer. ppr: [n_prompts, n_layers, d]"""
    _, n_layers, _ = ppr.shape
    rows = []
    for l in range(n_layers):
        try:
            er = eff_rank(ppr[:, l, :].numpy())
        except Exception:
            er = float("nan")
        rows.append({"layer": l, "effective_rank": er})
    return rows


def compute_esva(H_pos, H_neg):
    """ESVA per layer. H_pos/H_neg: [n_prompts, n_layers, d]"""
    _, n_layers, _ = H_pos.shape
    rows = []
    for l in range(n_layers):
        try:
            er_p = eff_rank(H_pos[:, l, :].numpy())
            er_n = eff_rank(H_neg[:, l, :].numpy())
            esva = (er_p - er_n) / (er_p + er_n + 1e-8)
        except Exception:
            esva = float("nan")
        rows.append({"layer": l, "esva": esva, "effrank_pos": er_p, "effrank_neg": er_n})
    return rows


def compute_gbc(ppr, scores):
    """GBC per layer. ppr: [n_prompts, n_layers, d], scores: [n_prompts]"""
    _, n_layers, _ = ppr.shape
    b = scores.astype(np.float32)
    rows = []
    for l in range(n_layers):
        norms = ppr[:, l, :].norm(dim=-1).numpy()
        if np.std(norms) < 1e-8 or np.std(b) < 1e-8:
            rows.append({"layer": l, "gbc_harmfulness": 0.0})
            continue
        r = float(np.corrcoef(norms, b)[0, 1])
        rows.append({"layer": l, "gbc_harmfulness": r if not math.isnan(r) else 0.0})
    return rows


def compute_rho_eff(eval_hs, residue):
    """
    rho_eff and rho_rel per layer.

    eval_hs: [n_prompts, n_layers, d]  — MUST be from real inference-time
             activations captured under the joint persona instruction, averaging
             over response tokens only. NOT from a stub forward pass.
    residue: [n_layers, d]

    rho_eff^(l) = E[|<h_x, r_hat>|]
    rho_rel^(l) = rho_eff / E[||h_x||]

    Interpretation:
      rho_rel > 0.05  → direction is active; null ablation = non-identifiability
      rho_rel > 0.01  → moderate signal
      rho_rel ≤ 0.01  → negligible; null ablation result mechanically explained
    """
    n_prompts, n_layers, _ = eval_hs.shape
    rows = []
    for l in range(n_layers):
        r  = residue[l]
        rn = float(torch.linalg.vector_norm(r).item())
        if rn < 1e-8:
            rows.append({"layer": l, "rho_eff": 0.0, "rho_rel": 0.0})
            continue
        r_hat    = r / rn
        h        = eval_hs[:, l, :]
        projs    = (h @ r_hat).abs()
        rho_e    = float(projs.mean().item())
        mean_n   = float(h.norm(dim=-1).mean().item())
        rows.append({
            "layer":   l,
            "rho_eff": rho_e,
            "rho_rel": rho_e / (mean_n + 1e-8),
        })
    return rows


def compute_projection_on_residue(ppr, residue):
    """
    Mean absolute projection of per-prompt residues onto mean residue direction.

    NOTE: This is computed from per-prompt residues (extracted at generation time),
    which is a different distribution from eval-time activations. This gives a
    proxy for how self-consistent the residue direction is within the extraction
    dataset, NOT a measure of how active the direction is at eval time.
    For true rho_rel, use compute_rho_eff with eval_hidden_states captured under
    the joint persona instruction with real responses.
    """
    _, n_layers, _ = ppr.shape
    rows = []
    for l in range(n_layers):
        r  = residue[l]
        rn = float(torch.linalg.vector_norm(r).item())
        if rn < 1e-8:
            rows.append({"layer": l, "mean_abs_proj": 0.0, "mean_norm_proj": 0.0})
            continue
        r_hat = r / rn
        h     = ppr[:, l, :]
        projs = (h @ r_hat).abs()
        hn    = h.norm(dim=-1)
        rows.append({
            "layer":          l,
            "mean_abs_proj":  float(projs.mean().item()),
            "mean_norm_proj": float((projs / (hn + 1e-8)).mean().item()),
        })
    return rows


# ── Bootstrap CIs and permutation null (per-prompt residue required) ────────────

def _alpha_cos_ncf(R, tA, tU, eps=1e-8):
    """Per-layer (alpha_A, alpha_U, cosA, cosU, ncf) for a residue R [n_layers, d]."""
    out = []
    for l in range(R.shape[0]):
        r, a, u = R[l], tA[l], tU[l]
        nr, na, nu = float(r.norm()), float(a.norm()), float(u.norm())
        ra, ru = float(torch.dot(r, a)), float(torch.dot(r, u))
        e1, e2 = gram_schmidt_2(a, u)
        r_span = torch.dot(r, e1) * e1 + torch.dot(r, e2) * e2
        ncf = float((r - r_span).norm() ** 2) / (nr ** 2 + eps)
        out.append((1 + ra / (na ** 2 + eps), 1 + ru / (nu ** 2 + eps),
                    ra / (nr * na + eps), ru / (nr * nu + eps), ncf))
    return np.array(out)  # [n_layers, 5]


def compute_bootstrap_ci(ppr, tau_A, tau_U, n_boot=2000, seed=0):
    """Bootstrap over prompts -> per-layer 95% CIs for alpha/cos/NCF.
    ppr: [n_prompts, n_layers, d]; tau_A, tau_U: [n_layers, d]."""
    rng = np.random.default_rng(seed)
    n_pp, n_l, _ = ppr.shape
    point = _alpha_cos_ncf(ppr.mean(0), tau_A, tau_U)          # [n_l, 5]
    boots = np.empty((n_boot, n_l, 5), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_pp, n_pp)
        boots[b] = _alpha_cos_ncf(ppr[idx].mean(0), tau_A, tau_U)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["alpha_A_given_U", "alpha_U_given_A", "cos_R_vs_tau_A", "cos_R_vs_tau_U", "ncf"]
    rows = []
    for l in range(n_l):
        row = {"layer": l}
        for j, nm in enumerate(names):
            row[f"{nm}"]      = float(point[l, j])
            row[f"{nm}_lo"]   = float(lo[l, j])
            row[f"{nm}_hi"]   = float(hi[l, j])
        rows.append(row)
    return rows


def compute_permutation_null(ppr, tau_A, tau_U, n_perm=2000, seed=0):
    """Two citeable nulls per layer:
      (1) sign-flip randomization on per-prompt projections <R_x, tau_hat>:
          p-value that the MEAN projection (hence cos/alpha) differs from 0.
      (2) random-direction NCF null: NCF of a random vector matched to ||R||.
          NCF is ~ (d-2)/d under no structure; we report the null mean and a
          one-sided p that the OBSERVED NCF is BELOW chance (= genuine in-span mass)."""
    rng = np.random.default_rng(seed)
    n_pp, n_l, d = ppr.shape
    R = ppr.mean(0)
    rows = []
    for l in range(n_l):
        a = tau_A[l]; u = tau_U[l]
        a_hat = a / (a.norm() + 1e-8); u_hat = u / (u.norm() + 1e-8)
        # (1) sign-flip null on the per-prompt projection onto tau_A and tau_U
        pA = (ppr[:, l, :] @ a_hat).numpy(); pU = (ppr[:, l, :] @ u_hat).numpy()
        obsA, obsU = pA.mean(), pU.mean()
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n_pp))
        nullA = (signs * pA).mean(1); nullU = (signs * pU).mean(1)
        pvalA = float((np.abs(nullA) >= abs(obsA)).mean())
        pvalU = float((np.abs(nullU) >= abs(obsU)).mean())
        # (2) random-direction NCF null at matched norm
        nr = float(R[l].norm())
        e1, e2 = gram_schmidt_2(a, u)
        rr = torch.from_numpy(rng.standard_normal((n_perm, d)).astype(np.float32))
        rr = rr / (rr.norm(dim=-1, keepdim=True) + 1e-8) * nr
        span = (rr @ e1).unsqueeze(-1) * e1 + (rr @ e2).unsqueeze(-1) * e2
        ncf_null = ((rr - span).norm(dim=-1) ** 2 / (nr ** 2 + 1e-8)).numpy()
        r = R[l]; r_span = torch.dot(r, e1) * e1 + torch.dot(r, e2) * e2
        ncf_obs = float((r - r_span).norm() ** 2 / (nr ** 2 + 1e-8))
        rows.append({
            "layer": l,
            "proj_A_pval_signflip": pvalA, "proj_U_pval_signflip": pvalU,
            "ncf_obs": ncf_obs, "ncf_null_mean": float(ncf_null.mean()),
            "ncf_pval_below_chance": float((ncf_null <= ncf_obs).mean()),
        })
    return rows


# ── Summary ───────────────────────────────────────────────────────────────────

def generate_summary(main_df, effrank_df, esva_df, gbc_df, rho_df):
    n = len(main_df)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    indices = sorted(set(i for i in indices if 0 <= i < n))
    lines = ["=" * 70, "PERSONA INTERACTION METRICS SUMMARY", "=" * 70, ""]
    for i in indices:
        row = main_df.iloc[i]
        lines.append(f"Layer {int(row['layer'])}:")
        for c in [
            "norm_tau_A", "norm_tau_U", "norm_tau_AU", "eta",
            "cos_R_vs_tau_A", "cos_R_vs_tau_U",
            "alpha_A_given_U", "alpha_U_given_A",
            "alpha_A_given_U_span", "alpha_U_given_A_span",
            "ncf", "cai", "rna",
        ]:
            lines.append(f"  {c:<30s} = {row[c]:.4f}")
        if effrank_df is not None and i < len(effrank_df):
            lines.append(f"  {'EffRank':<30s} = {effrank_df.iloc[i]['effective_rank']:.2f}")
        if esva_df is not None and i < len(esva_df):
            lines.append(f"  {'ESVA':<30s} = {esva_df.iloc[i]['esva']:.4f}")
        if gbc_df is not None and i < len(gbc_df):
            lines.append(f"  {'GBC(harmfulness)':<30s} = {gbc_df.iloc[i]['gbc_harmfulness']:.4f}")
        if rho_df is not None and i < len(rho_df):
            rr = rho_df.iloc[i]["rho_rel"]
            if rr > 0.05:
                interp = "SIGNIFICANT — null ablation = non-identifiability"
            elif rr > 0.01:
                interp = "moderate"
            else:
                interp = "negligible — null ablation result mechanically explained"
            lines.append(f"  {'rho_rel':<30s} = {rr:.4f}  ({interp})")
        lines.append("")
    lines.append("NCF note: 'span' alpha scores remove NCF component from tau_AU.")
    lines.append("If alpha_*_span ≈ alpha_* then NCF barely contaminates amplification.")
    lines.append("If they differ substantially, the NCF was contributing to the score.")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compute persona interaction metrics (CSV output).")
    p.add_argument("--vec_dir",               default="persona_vectors/Qwen2.5-7B-Instruct")
    p.add_argument("--ai_trait",              default="orig_humorous")
    p.add_argument("--user_trait",            default="user_evil")
    p.add_argument("--joint_trait",           default="humorous_ai_user_evil")
    p.add_argument("--output_dir",            default="output/metrics")
    p.add_argument("--output_prefix",         default="")
    p.add_argument("--n_boot", type=int, default=2000,
                   help="bootstrap resamples for alpha/cos/NCF CIs (per_prompt_residue).")
    p.add_argument("--n_perm", type=int, default=2000,
                   help="permutations for sign-flip + random-direction NCF nulls.")
    p.add_argument("--pos_activation_matrix", default="",
                   help="[n_prompts, n_layers, d] pos-side real-generation activations (ESVA).")
    p.add_argument("--neg_activation_matrix", default="",
                   help="[n_prompts, n_layers, d] neg-side real-generation activations (ESVA).")
    p.add_argument("--per_prompt_residue",    default="",
                   help="[n_prompts, n_layers, d] per-prompt residue (EffRank, GBC, projection).")
    p.add_argument("--scored_csv",            default="",
                   help="CSV with 'harmfulness' column for GBC.")
    p.add_argument("--eval_hidden_states",    default="",
                   help="[n_prompts, n_layers, d] real-generation eval activations for rho_eff. "
                        "Must be captured under joint persona instruction, response tokens only.")
    # Cross-trait cosine diagnostic
    p.add_argument("--cross_trait_vec_dir",   default="",
                   help="Vec dir for cross-trait cosine. Defaults to --vec_dir if not set.")
    p.add_argument("--cross_trait_left",      default="",
                   help="Left-side trait for cross-trait cosine (e.g. 'user_evil').")
    p.add_argument("--cross_trait_right",     default="",
                   help="Right-side trait for cross-trait cosine (e.g. 'orig_evil'). "
                        "Both left and right must be set to compute cross-trait cosine.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.vec_dir)
    prefix = args.output_prefix or f"{vdir.name}_{args.joint_trait}"

    # ── Load steering vectors ─────────────────────────────────────────────────
    print("Loading vectors...")
    mu_A  = load_vec(str(vdir / f"{args.ai_trait}_response_avg_diff.pt"))
    mu_U  = load_vec(str(vdir / f"{args.user_trait}_response_avg_diff.pt"))
    mu_AU = load_vec(str(vdir / f"{args.joint_trait}_response_avg_diff.pt"))

    # CAA vectors are already contrastive (pos − neg), so mu_base ≈ 0 and
    # tau_X = mu_X directly. See Section 1.3 of the Persona Interaction Metrics doc.
    tau_A  = mu_A
    tau_U  = mu_U
    tau_AU = mu_AU - mu_A - mu_U   # Interaction residue R_{A,U}

    # Prefer the pre-computed residue .pt if it exists (should be identical)
    residue = load_opt(str(vdir / f"residue_{args.joint_trait}_response_avg_diff.pt"))
    if residue is None:
        residue = tau_AU
    else:
        # Sanity check: confirm the stored residue matches recomputed one
        delta = float((residue - tau_AU).norm().item())
        if delta > 1.0:
            print(f"  [WARNING] Loaded residue differs from mu_AU-mu_A-mu_U "
                  f"by norm {delta:.3f}. Using loaded residue.")

    n_layers, d = tau_A.shape
    print(f"  {n_layers} layers, d={d}")

    # Save tau vectors as npz for downstream scripts
    np.savez(
        str(out_dir / f"{prefix}_tau_vectors.npz"),
        tau_A=tau_A.numpy(), tau_U=tau_U.numpy(), tau_AU=tau_AU.numpy(),
        mu_A=mu_A.numpy(),  mu_U=mu_U.numpy(),  mu_AU=mu_AU.numpy(),
    )
    print("  Saved tau_vectors.npz")

    # ── Core per-layer metrics ────────────────────────────────────────────────
    print("Computing per-layer metrics...")
    main_df = pd.DataFrame(compute_per_layer(tau_A, tau_U, tau_AU))
    main_df.to_csv(str(out_dir / f"{prefix}_per_layer_metrics.csv"), index=False)
    print(f"  Saved per_layer_metrics.csv ({len(main_df)} rows)")

    # ── Optional: cross-trait cosine ─────────────────────────────────────────
    if args.cross_trait_left and args.cross_trait_right:
        ct_vdir = Path(args.cross_trait_vec_dir) if args.cross_trait_vec_dir else vdir
        print(f"  Cross-trait cosine: {args.cross_trait_left} vs {args.cross_trait_right}...")
        vec_left  = load_vec(str(ct_vdir / f"{args.cross_trait_left}_response_avg_diff.pt"))
        vec_right = load_vec(str(ct_vdir / f"{args.cross_trait_right}_response_avg_diff.pt"))
        ct_df = pd.DataFrame(compute_cross_trait_cosine(vec_left, vec_right))
        ct_path = str(out_dir / f"{prefix}_cross_cosine_{args.cross_trait_left}_vs_{args.cross_trait_right}.csv")
        ct_df.to_csv(ct_path, index=False)
        print(f"  Saved cross-trait cosine CSV")

    effrank_df = esva_df = gbc_df = rho_df = proj_df = None

    # ── EffRank, GBC, projection from per-prompt residue ─────────────────────
    ppr = load_opt(args.per_prompt_residue)
    if ppr is not None:
        n_pp, n_l_pp, _ = ppr.shape
        n_l_use = min(n_l_pp, n_layers)
        ppr_a   = ppr[:, :n_l_use, :]
        res_a   = residue[:n_l_use]

        print(f"  EffRank: {n_l_use} layers, {n_pp} prompts...")
        effrank_df = pd.DataFrame(compute_effrank(ppr_a))
        effrank_df.to_csv(str(out_dir / f"{prefix}_effrank.csv"), index=False)

        print("  Projection on residue direction (extraction-time proxy)...")
        proj_df = pd.DataFrame(compute_projection_on_residue(ppr_a, res_a))
        proj_df.to_csv(str(out_dir / f"{prefix}_projection_on_residue.csv"), index=False)

        print(f"  Bootstrap CIs (alpha/cos/NCF) over {n_pp} prompts x{args.n_boot}...")
        boot_df = pd.DataFrame(compute_bootstrap_ci(
            ppr_a, tau_A[:n_l_use], tau_U[:n_l_use], n_boot=args.n_boot))
        boot_df.to_csv(str(out_dir / f"{prefix}_bootstrap_ci.csv"), index=False)

        print(f"  Permutation null (sign-flip + random-direction NCF) x{args.n_perm}...")
        perm_df = pd.DataFrame(compute_permutation_null(
            ppr_a, tau_A[:n_l_use], tau_U[:n_l_use], n_perm=args.n_perm))
        perm_df.to_csv(str(out_dir / f"{prefix}_permutation_null.csv"), index=False)

        if args.scored_csv:
            scored = pd.read_csv(args.scored_csv)
            if "mode" in scored.columns:
                scored = scored[scored["mode"] == "baseline"]
            if len(scored) >= n_pp:
                harm = scored["harmfulness"].values[:n_pp].astype(np.float32)
                print(f"  GBC: {n_pp} prompts...")
                gbc_df = pd.DataFrame(compute_gbc(ppr_a, harm))
                gbc_df.to_csv(str(out_dir / f"{prefix}_gbc.csv"), index=False)
            else:
                print(f"  GBC skipped: scored CSV has {len(scored)} rows < {n_pp} prompts.")

    # ── ESVA from pos/neg activation matrices ─────────────────────────────────
    H_pos = load_opt(args.pos_activation_matrix)
    H_neg = load_opt(args.neg_activation_matrix)
    if H_pos is not None and H_neg is not None:
        n_l_esva = min(H_pos.shape[1], n_layers)
        print(f"  ESVA: {n_l_esva} layers...")
        esva_df = pd.DataFrame(
            compute_esva(H_pos[:, :n_l_esva, :], H_neg[:, :n_l_esva, :])
        )
        esva_df.to_csv(str(out_dir / f"{prefix}_esva.csv"), index=False)

    # ── rho_eff from eval hidden states ───────────────────────────────────────
    eval_hs = load_opt(args.eval_hidden_states)
    if eval_hs is not None:
        n_l_ev = min(eval_hs.shape[1], n_layers)
        print(f"  rho_eff: {n_l_ev} layers...")
        rho_df = pd.DataFrame(
            compute_rho_eff(eval_hs[:, :n_l_ev, :], residue[:n_l_ev])
        )
        rho_df.to_csv(str(out_dir / f"{prefix}_rho_eff.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = generate_summary(main_df, effrank_df, esva_df, gbc_df, rho_df)
    (out_dir / f"{prefix}_metrics_summary.txt").write_text(summary)
    print(f"\n{summary}")
    print(f"All outputs in: {out_dir}")


if __name__ == "__main__":
    main()