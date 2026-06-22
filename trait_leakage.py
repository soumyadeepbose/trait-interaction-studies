#!/usr/bin/env python3
"""
trait_leakage.py -- cross-slot MAIN-EFFECT "trait leakage" (a.k.a. trait transfer).

Companion to metrics.py. Where the amplification score alpha measures the *interaction*
(via the residue R = mu_AU - mu_A - mu_U), this script answers the *main-effect* question:

    "Does conditioning the OTHER slot on a trait move the assistant's RESPONSE state
     along the assistant's OWN axis for that same trait?"

Both inputs are contrastive shifts measured on RESPONSE tokens, in the same space:
  - tau_source : the trait marginal extracted with the trait in the *other* slot
                 (e.g. user-evil  = tau_U from the humor-AI/evil-USER run)
  - tau_target : the trait marginal extracted with the trait in the *own* slot
                 (e.g. AI-evil    = tau_A from the evil-AI/humor-USER run)

Per-layer quantities (mirroring how the framework pairs cos(R,tau) with alpha):
  cos_leak   = cos(tau_source, tau_target)                  # directional, scale-free
  gamma_leak = <tau_source, tau_target> / ||tau_target||^2  # leakage GAIN, scale-aware,
               in units of the target slot's own-trait push:
                 gamma=1 -> other-slot instruction as potent as self-instruction (full leak)
                 gamma=0 -> no leakage ;  gamma<0 -> opposition / anti-leakage

Headline use (role-indexed safety):  compare instruct vs base via --base_*.
  Prediction:  gamma_evil(instruct) < gamma_evil(base)   (RLHF reduces user->assistant
  evil transfer), while a neutral trait (humor) leaks ~equally.  cos may stay high in
  both; the RLHF effect is expected to surface in gamma, not the angle.

Intervention hook:  --emit_shared writes s_l = gamma_l * tau_target[l] (the component of
the source shift lying on the target axis) as a [n_layers, d] .pt that ablation_study.py
can ADD (amplify leakage) or PROJECT OUT (ablate leakage).

CAUTION: source and target must use IDENTICAL trait text, differing only by slot, or
(1 - cos) mixes slot-effect with wording. The clean design is ONE factorial with the
trait flipped in both slots and the companion fixed; then source=tau_U, target=tau_A
share a --prefix.
"""
import argparse
import math
from pathlib import Path

import torch
import pandas as pd


def load_vec(vec_dir: Path, name: str) -> torch.Tensor:
    p = vec_dir / f"{name}_response_avg_diff.pt"
    if not p.exists():
        raise FileNotFoundError(f"missing vector: {p}")
    return torch.load(str(p), map_location="cpu", weights_only=False).float()


def per_layer_leakage(tau_source: torch.Tensor, tau_target: torch.Tensor):
    """tau_source, tau_target: [n_layers, d]. Returns list[dict] (one per layer)."""
    assert tau_source.shape == tau_target.shape, "source/target layer-dim mismatch"
    eps = 1e-8
    rows = []
    for l in range(tau_source.shape[0]):
        s, t = tau_source[l], tau_target[l]
        ns, nt = float(s.norm()), float(t.norm())
        dot = float(torch.dot(s, t))
        L = dot / (ns * nt + eps)
        rows.append(dict(
            layer=l,
            cos_leak=L,
            gamma_leak=dot / (nt ** 2 + eps),
            angle_deg=math.degrees(math.acos(max(-1.0, min(1.0, L)))),
            norm_source=ns,
            norm_target=nt,
        ))
    return rows


def shared_leakage_vector(tau_source: torch.Tensor, tau_target: torch.Tensor) -> torch.Tensor:
    """s_l = gamma_l * tau_target[l]: the part of the source shift on the target axis."""
    eps = 1e-8
    out = torch.zeros_like(tau_target)
    for l in range(tau_target.shape[0]):
        t = tau_target[l]
        g = float(torch.dot(tau_source[l], t)) / (float(t.norm()) ** 2 + eps)
        out[l] = g * t
    return out


def summarize(rows, layer_lo, layer_hi, label):
    sel = [r for r in rows if layer_lo <= r["layer"] <= layer_hi]
    mc = sum(r["cos_leak"] for r in sel) / len(sel)
    mg = sum(r["gamma_leak"] for r in sel) / len(sel)
    lines = [
        "=" * 64,
        f"TRAIT LEAKAGE: {label}",
        "=" * 64,
        f"(summary over layers [{layer_lo}, {layer_hi}]; endpoints L0/embedding and the",
        f" final layer/near-unembedding are excluded by default -- see --layer_lo/--layer_hi)",
        "",
        f"  mean cos_leak    = {mc:+.3f}   (angle {math.degrees(math.acos(max(-1,min(1,mc)))):.1f} deg)",
        f"  mean gamma_leak  = {mg:+.3f}   (1.0 = full transfer; 0 = none)",
        "",
        f"{'L':>3} {'cos':>8} {'gamma':>8} {'angle':>7} {'|src|':>8} {'|tgt|':>8}",
    ]
    for r in rows:
        lines.append(f"{r['layer']:>3} {r['cos_leak']:>8.3f} {r['gamma_leak']:>8.3f} "
                     f"{r['angle_deg']:>7.1f} {r['norm_source']:>8.2f} {r['norm_target']:>8.2f}")
    return "\n".join(lines), mc, mg


def main():
    ap = argparse.ArgumentParser(description="Cross-slot trait leakage / trait transfer.")
    ap.add_argument("--source_vec_dir", required=True,
                    help="dir holding the OTHER-slot trait marginal")
    ap.add_argument("--source_name", required=True,
                    help="basename of source marginal, e.g. fac_humor_evil_user_fac_tauU")
    ap.add_argument("--target_vec_dir", required=True,
                    help="dir holding the OWN-slot trait axis")
    ap.add_argument("--target_name", required=True,
                    help="basename of target axis, e.g. fac_evil_ai_humor_user_fac_tauA")
    ap.add_argument("--label", default="leakage")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer_lo", type=int, default=1)
    ap.add_argument("--layer_hi", type=int, default=-2,
                    help="inclusive upper layer; -2 = (n_layers-2), excludes near-unembedding")
    # optional base-vs-instruct delta (the headline)
    ap.add_argument("--base_source_vec_dir")
    ap.add_argument("--base_target_vec_dir")
    # intervention vector
    ap.add_argument("--emit_shared", action="store_true",
                    help="write shared-leakage steering vector s_l = gamma_l * tau_target[l]")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tau_src = load_vec(Path(args.source_vec_dir), args.source_name)
    tau_tgt = load_vec(Path(args.target_vec_dir), args.target_name)
    n = tau_src.shape[0]
    lo = args.layer_lo
    hi = args.layer_hi if args.layer_hi >= 0 else (n + args.layer_hi)

    rows = per_layer_leakage(tau_src, tau_tgt)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.label}_leakage.csv"
    df.to_csv(csv_path, index=False)
    summary, mc, mg = summarize(rows, lo, hi, args.label)

    # optional base comparison
    if args.base_source_vec_dir and args.base_target_vec_dir:
        b_src = load_vec(Path(args.base_source_vec_dir), args.source_name)
        b_tgt = load_vec(Path(args.base_target_vec_dir), args.target_name)
        b_rows = per_layer_leakage(b_src, b_tgt)
        _, b_mc, b_mg = summarize(b_rows, lo, hi, f"{args.label} (base)")
        pd.DataFrame(b_rows).to_csv(out_dir / f"{args.label}_leakage_base.csv", index=False)
        summary += (
            "\n\n" + "-" * 64 +
            f"\nBASE vs INSTRUCT (layers [{lo},{hi}]):"
            f"\n  cos:    base {b_mc:+.3f}  ->  instruct {mc:+.3f}   (delta {mc-b_mc:+.3f})"
            f"\n  gamma:  base {b_mg:+.3f}  ->  instruct {mg:+.3f}   (delta {mg-b_mg:+.3f})"
            f"\n  Prediction holds if gamma drops (RLHF reduces transfer of this trait).\n"
        )

    (out_dir / f"{args.label}_leakage_summary.txt").write_text(summary)
    print(summary)
    print(f"\n  saved {csv_path}")

    if args.emit_shared:
        s = shared_leakage_vector(tau_src, tau_tgt)
        sp = out_dir / f"{args.label}_shared_leak_response_avg_diff.pt"
        torch.save(s, str(sp))
        print(f"  saved shared-leakage steering vector {sp}  (add to amplify, project-out to ablate)")


if __name__ == "__main__":
    main()