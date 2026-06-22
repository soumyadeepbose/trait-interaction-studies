#!/usr/bin/env python3
"""
refusal_cosines.py — Per-layer cosine battery of the refusal direction against the
factorial vectors.

Computes, per layer, the cosine of d_refusal with:
    R          = joint - tau_A - tau_U   (the interaction residue)
    R_perp     = component of R orthogonal to span{tau_A, tau_U}  (the "novel" part)
    tau_A, tau_U, joint

Headline check (instruct model, evil-AI setting): |cos(R, d_ref)| should exceed both
|cos(tau_A, d_ref)| and |cos(tau_U, d_ref)| at late layers, and cos(R_perp, d_ref) being
large says the novel component is itself a safety direction.

Usage:
    python refusal_cosines.py <vec_dir> <fac_prefix> <out_csv>

where <fac_prefix> is e.g. fac_humor_evil_user_fac, so the files read are
<vec_dir>/{<fac_prefix>_tauA,_tauU,_joint}_response_avg_diff.pt and
<vec_dir>/refusal_response_avg_diff.pt.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def load(vec_dir, name):
    p = Path(vec_dir) / f"{name}_response_avg_diff.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing vector: {p}")
    return torch.load(str(p), map_location="cpu", weights_only=False).float()


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def gram_schmidt_2(v1, v2, eps=1e-8):
    """Orthonormal basis of span{v1, v2} (matches metrics.py's Gram-Schmidt split)."""
    e1 = v1 / (v1.norm() + eps)
    v2o = v2 - (v2 @ e1) * e1
    n2 = v2o.norm()
    e2 = v2o / (n2 + eps) if n2 > eps else torch.zeros_like(e1)
    return e1, e2


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: python refusal_cosines.py <vec_dir> <fac_prefix> <out_csv>")
    vec_dir, fac, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]

    tau_A = load(vec_dir, f"{fac}_tauA")
    tau_U = load(vec_dir, f"{fac}_tauU")
    joint = load(vec_dir, f"{fac}_joint")
    d_ref = load(vec_dir, "refusal")
    R = joint - tau_A - tau_U

    n = min(R.shape[0], d_ref.shape[0])
    rows = []
    for l in range(n):
        e1, e2 = gram_schmidt_2(tau_A[l], tau_U[l])
        R_span = (R[l] @ e1) * e1 + (R[l] @ e2) * e2
        R_perp = R[l] - R_span
        rows.append(dict(
            layer=l,
            cos_R_refusal=cos(R[l], d_ref[l]),
            cos_Rperp_refusal=cos(R_perp, d_ref[l]),
            cos_tauA_refusal=cos(tau_A[l], d_ref[l]),
            cos_tauU_refusal=cos(tau_U[l], d_ref[l]),
            cos_joint_refusal=cos(joint[l], d_ref[l]),
        ))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({n} layers)")


if __name__ == "__main__":
    main()