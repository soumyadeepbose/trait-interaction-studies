#!/usr/bin/env python3
"""
ablation_stats.py — significance + effect size for an ablation run, done RIGHT.

The data are (a) PAIRED — baseline and ablate_head use the same (question,
instruction) prompts — and (b) CLUSTERED — 5 instruction variants share each of 64
questions, so rows are not independent. Naive t-tests / Mann-Whitney overstate
significance on both counts.

This script therefore:
  - pairs baseline vs ablate_head on (question_id, instruction_idx),
  - reports the mean / median / tail-rate (%>30, %>50) per arm,
  - computes a CLUSTER bootstrap CI on the paired mean difference (resampling the
    64 questions, not the 320 rows),
  - and runs a Wilcoxon signed-rank test on the per-question mean paired differences
    (distribution-free companion; appropriate for the skewed/bimodal evil scores),
  - also bootstraps the difference in TAIL RATE (%>30), the most paper-relevant
    quantity for an already-aligned model.

Usage:
    python ablation_stats.py \
      --csv output/ablation/A_role/pp/generations_scored.csv \
      --metric evilness --tail 30 --n_boot 5000
    # compare a specific coeff: --coeff 1.0
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def cluster_bootstrap(diffs_by_q, stat, n_boot, rng):
    """diffs_by_q: list of arrays (per-question paired diffs). Resample questions."""
    qs = list(diffs_by_q)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(qs), len(qs))
        pooled = np.concatenate([qs[i] for i in pick])
        boots.append(stat(pooled))
    return np.array(boots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="evilness")
    ap.add_argument("--baseline_mode", default="baseline")
    ap.add_argument("--treat_mode", default="ablate_head")
    ap.add_argument("--coeff", type=float, default=None,
                    help="treat-arm coeff to select (if multiple). baseline coeff ignored.")
    ap.add_argument("--tail", type=float, default=30.0, help="tail threshold (%>tail)")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df = pd.read_csv(args.csv)
    m = args.metric
    key = ["question_id", "instruction_idx"]

    base = df[df["mode"] == args.baseline_mode].copy()
    treat = df[df["mode"] == args.treat_mode].copy()
    if args.coeff is not None and "coeff" in treat:
        treat = treat[np.isclose(treat["coeff"].fillna(-1), args.coeff)]
    if len(base) == 0 or len(treat) == 0:
        raise SystemExit(f"empty arm: baseline={len(base)} treat={len(treat)} "
                         f"(modes present: {df['mode'].unique().tolist()})")

    merged = base[key + [m]].merge(treat[key + [m]], on=key, suffixes=("_base", "_abl"))
    if len(merged) == 0:
        raise SystemExit("no paired rows — check question_id/instruction_idx align across arms")
    merged["diff"] = merged[f"{m}_abl"] - merged[f"{m}_base"]
    print(f"Paired on (question_id, instruction_idx): {len(merged)} pairs "
          f"across {merged['question_id'].nunique()} questions\n")

    b, a = merged[f"{m}_base"], merged[f"{m}_abl"]
    def line(name, x):
        return (f"  {name:<10} mean={x.mean():6.2f}  median={x.median():6.2f}  "
                f"%>{args.tail:.0f}={100*(x>args.tail).mean():5.1f}%  %>50={100*(x>50).mean():5.1f}%")
    print(f"{m} per arm:")
    print(line("baseline", b))
    print(line(args.treat_mode, a))
    print(f"\n  mean paired diff (abl - base): {merged['diff'].mean():+.2f}")
    print(f"  relative reduction in mean:    {100*merged['diff'].mean()/b.mean():+.1f}%")

    # per-question paired diffs (for clustering)
    diffs_by_q = [g["diff"].values for _, g in merged.groupby("question_id")]
    tail_base_by_q = [(g[f"{m}_base"].values > args.tail).astype(float) for _, g in merged.groupby("question_id")]
    tail_abl_by_q = [(g[f"{m}_abl"].values > args.tail).astype(float) for _, g in merged.groupby("question_id")]

    # cluster bootstrap: mean paired diff
    bd = cluster_bootstrap(diffs_by_q, np.mean, args.n_boot, rng)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    print(f"\n  cluster-bootstrap 95% CI on mean paired diff: "
          f"[{lo:+.2f}, {hi:+.2f}]  ({'excludes 0 -> significant' if hi < 0 or lo > 0 else 'includes 0'})")

    # cluster bootstrap: tail-rate difference (the safety-relevant number)
    qs = list(range(len(diffs_by_q)))
    tail_boots = []
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(qs), len(qs))
        tb = np.concatenate([tail_base_by_q[i] for i in pick]).mean()
        ta = np.concatenate([tail_abl_by_q[i] for i in pick]).mean()
        tail_boots.append(100 * (ta - tb))
    tlo, thi = np.percentile(tail_boots, [2.5, 97.5])
    tb_obs = 100 * np.concatenate(tail_base_by_q).mean()
    ta_obs = 100 * np.concatenate(tail_abl_by_q).mean()
    print(f"  tail rate %>{args.tail:.0f}:  baseline={tb_obs:.1f}%  ablated={ta_obs:.1f}%  "
          f"Δ={ta_obs-tb_obs:+.1f}pp  CI[{tlo:+.1f}, {thi:+.1f}]")

    # Wilcoxon signed-rank on per-question mean paired diffs (distribution-free)
    perq = np.array([d.mean() for d in diffs_by_q])
    try:
        stat, p = wilcoxon(perq)
        print(f"\n  Wilcoxon signed-rank (per-question paired diffs, n={len(perq)}): "
              f"W={stat:.1f}, p={p:.4g}")
    except ValueError as e:
        print(f"\n  Wilcoxon could not run: {e}")


if __name__ == "__main__":
    main()