#!/usr/bin/env python3
"""
factorial_residue.py — Combine the 4 factorial cell vectors into tau_A, tau_U, R.

INPUT
-----
Four CAA vectors produced by the EXISTING persona_steering/generate_vec.py, one per
cell. Each generate_vec output is  v_cell = mu_cell(pos) - mu_baseline(neg)  with the
SAME shared baseline (the "neg" side of every cell JSON). The baseline cancels in
every contrast below, so v_cell behaves exactly like the cell mean up to a constant
that never survives a factorial difference.

File naming (matches generate_vec's "<trait>_response_avg_diff.pt" convention):
    <prefix>_pp_response_avg_diff.pt   # A+ , B+
    <prefix>_pn_response_avg_diff.pt   # A+ , B-
    <prefix>_np_response_avg_diff.pt   # A- , B+
    <prefix>_nn_response_avg_diff.pt   # A- , B-

Each is a tensor of shape [n_layers, d].

FACTORIAL ALGEBRA  (first index = AI axis, second index = B axis)
-----------------------------------------------------------------
    tau_A (AI main effect)   = 0.5 * [ (v_pp - v_np) + (v_pn - v_nn) ]
    tau_U (B  main effect)   = 0.5 * [ (v_pp - v_pn) + (v_np - v_nn) ]
    R     (interaction)      =        v_pp - v_np - v_pn + v_nn
    joint                    = tau_A + tau_U + R

Derivation sanity check (effect-coded 2x2: mu_ij = m + a_i + b_j + g_ij, with
Sum_i a_i = Sum_j b_j = Sum_i g_ij = Sum_j g_ij = 0):
    v_pp - v_np - v_pn + v_nn = 4 * g_{++}   -> the pure interaction, up to scale.
The shared baseline m and template noise appear in all four cells with a net
coefficient of (+1 -1 -1 +1) = 0 in R, and (+1 -1 +1 -1)=0 / (+1 +1 -1 -1)=0 in the
main effects, so they vanish. This is the whole point.

OUTPUT  (drop-in for metrics.py, which recomputes tau_AU = joint - tau_A - tau_U = R)
------------------------------------------------------------------------------------
    <out_prefix>_tauA_response_avg_diff.pt
    <out_prefix>_tauU_response_avg_diff.pt
    <out_prefix>_joint_response_avg_diff.pt

Then:
    python metrics.py \
        --vec_dir <vec_dir> \
        --ai_trait    <out_prefix>_tauA \
        --user_trait  <out_prefix>_tauU \
        --joint_trait <out_prefix>_joint \
        --output_dir output/metrics --output_prefix <out_prefix>

metrics.py will internally form tau_AU = joint - tau_A - tau_U = R exactly.

PER-PROMPT RESIDUE (optional, for EffRank/GBC under the factorial)
------------------------------------------------------------------
If you pass --per_prompt and the 4 per-prompt activation matrices exist
(generate_vec --save_activations writes "<trait>_pos_activation_matrix.pt" of shape
[n_prompts, n_layers, d]), this script also writes a per-prompt interaction residue
    R_i = ppr_pp[i] - ppr_np[i] - ppr_pn[i] + ppr_nn[i]
aligned across cells by prompt index. NOTE: alignment is positional, so the 4 cells
MUST have been extracted over the identical question x instruction ordering (they are,
if you used build_factorial_dataset.py unchanged). Saved as:
    <out_prefix>_per_prompt_residue.pt   -> feed to metrics.py --per_prompt_residue
"""

import argparse
from pathlib import Path

import torch


def load_vec(path):
    v = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(v):
        v = torch.tensor(v)
    return v.float()


def main():
    p = argparse.ArgumentParser(description="Combine 4 factorial cells into tau_A, tau_U, R.")
    p.add_argument("--vec_dir", required=True,
                   help="Dir containing the 4 cell <prefix>_{pp,pn,np,nn}_response_avg_diff.pt files.")
    p.add_argument("--prefix", required=True,
                   help="Cell file prefix, e.g. fac_humor_evil_user.")
    p.add_argument("--out_prefix", default=None,
                   help="Output prefix. Default: <prefix>_fac.")
    p.add_argument("--per_prompt", action="store_true",
                   help="Also build the per-prompt interaction residue from pos activation matrices.")
    p.add_argument("--pp_suffix", default="_pos_activation_matrix.pt",
                   help="Suffix of the per-prompt activation matrices (generate_vec --save_activations).")
    args = p.parse_args()

    vdir = Path(args.vec_dir)
    out_prefix = args.out_prefix or f"{args.prefix}_fac"

    # ── Load the four cell vectors ───────────────────────────────────────────
    cells = {}
    for cell in ["pp", "pn", "np", "nn"]:
        path = vdir / f"{args.prefix}_{cell}_response_avg_diff.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cell vector: {path}")
        cells[cell] = load_vec(str(path))
    shapes = {c: tuple(v.shape) for c, v in cells.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Cell vectors have mismatched shapes: {shapes}")
    print(f"  Loaded 4 cells, shape {next(iter(shapes.values()))}")

    v_pp, v_pn, v_np, v_nn = cells["pp"], cells["pn"], cells["np"], cells["nn"]

    # ── Factorial decomposition ──────────────────────────────────────────────
    tau_A = 0.5 * ((v_pp - v_np) + (v_pn - v_nn))
    tau_U = 0.5 * ((v_pp - v_pn) + (v_np - v_nn))
    R = v_pp - v_np - v_pn + v_nn
    joint = tau_A + tau_U + R

    torch.save(tau_A, str(vdir / f"{out_prefix}_tauA_response_avg_diff.pt"))
    torch.save(tau_U, str(vdir / f"{out_prefix}_tauU_response_avg_diff.pt"))
    torch.save(joint, str(vdir / f"{out_prefix}_joint_response_avg_diff.pt"))
    print(f"  Saved {out_prefix}_tauA / _tauU / _joint  (joint = tau_A + tau_U + R)")

    # Quick diagnostics so you can eyeball whether R is non-trivial.
    def per_layer_norm(x):
        return torch.linalg.vector_norm(x, dim=-1)
    nR, nA, nU = per_layer_norm(R), per_layer_norm(tau_A), per_layer_norm(tau_U)
    ratio = (nR / (nA + nU + 1e-8))
    print(f"  ||R|| / (||tau_A|| + ||tau_U||)  per-layer: "
          f"min={ratio.min():.3f} mean={ratio.mean():.3f} max={ratio.max():.3f}")

    # ── Optional per-prompt interaction residue ──────────────────────────────
    if args.per_prompt:
        mats = {}
        for cell in ["pp", "pn", "np", "nn"]:
            path = vdir / f"{args.prefix}_{cell}{args.pp_suffix}"
            if not path.exists():
                raise FileNotFoundError(
                    f"--per_prompt set but missing {path}. "
                    f"Re-run generate_vec with --save_activations for each cell.")
            mats[cell] = load_vec(str(path))  # [n_prompts, n_layers, d]
        n = min(m.shape[0] for m in mats.values())
        pp_R = (mats["pp"][:n] - mats["np"][:n] - mats["pn"][:n] + mats["nn"][:n])
        out_pp = vdir / f"{out_prefix}_per_prompt_residue.pt"
        torch.save(pp_R, str(out_pp))
        print(f"  Saved {out_pp.name}  shape {tuple(pp_R.shape)}  (feed to metrics.py --per_prompt_residue)")


if __name__ == "__main__":
    main()
