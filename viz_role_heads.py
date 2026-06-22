#!/usr/bin/env python3
"""
viz_role_heads.py — base-vs-instruct visualization of the role-selective heads.

Picks the top role-selective heads from the INSTRUCT per_head_attribution.csv
(localize_heads.py output), then for THOSE SAME head indices extracts, in both the
instruct and base models, the last-token attention mass over the {system, user,
assistant} token spans (Setting B, pp cell). Produces side-by-side comparisons.

Why same heads (not each model's own top-4): head index L.h is the same architectural
slot in both models and attention patterns are geometry-free, so they ARE comparable;
each model's suppression attribution is in its own evil-axis units (kept separate).

Models are loaded SEQUENTIALLY (one at a time) so two 7B models never co-reside.

Outputs:
  - attention_comparison.png : one panel per head; grouped bars (Base vs Instruct)
    over {system, user, assistant} span attention. Tests "does the suppressor attend
    to the system prompt (AI-persona declaration)?"
  - suppression_comparison.png : interaction attribution per head, Base vs Instruct
    (needs --base_attrib). Shows RLHF installed the suppression at these heads.
  - role_head_attention_compare.csv : the raw span numbers.

Usage:
    python viz_role_heads.py \
      --instruct_model Qwen/Qwen2.5-7B-Instruct --base_model Qwen/Qwen2.5-7B \
      --extract_dir_instruct persona_steering/eval_persona_extract/Qwen2.5-7B-Instruct \
      --extract_dir_base     persona_steering/eval_persona_extract/Qwen2.5-7B \
      --instruct_attrib output/localize_heads/Qwen2.5-7B-Instruct/per_head_attribution.csv \
      --base_attrib     output/localize_heads/Qwen2.5-7B/per_head_attribution.csv \
      --prefix_b fac_evil_ai_humor_user --top_k 4 --max_prompts 32 \
      --out_dir output/viz_role_heads
    # override head selection with e.g. --heads 19:5,19:12,20:3,19:20
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_heads(attrib_csv, top_k, override):
    if override:
        return [(int(x.split(":")[0]), int(x.split(":")[1])) for x in override.split(",")]
    df = pd.read_csv(attrib_csv).sort_values("role_selectivity", ascending=False)
    return [(int(r["layer"]), int(r["head"])) for _, r in df.head(top_k).iterrows()]


def span_attention(model, tok, prompts, heads, device, max_prompts):
    """Mean last-token attention over {system,user,assistant} spans, per head."""
    imstart = tok.convert_tokens_to_ids("<|im_start|>")
    acc = {hd: {"system": [], "user": [], "assistant": []} for hd in heads}
    n = min(len(prompts), max_prompts)
    for i in range(n):
        ids = tok(prompts[i], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        seq = ids[0].tolist()
        marks = [j for j, t in enumerate(seq) if t == imstart]
        if len(marks) < 3:
            continue
        spans = {"system": list(range(marks[0], marks[1])),
                 "user": list(range(marks[1], marks[2])),
                 "assistant": list(range(marks[2], len(seq)))}
        with torch.no_grad():
            o = model(ids, output_attentions=True, use_cache=False)
        for (L, h) in heads:
            a = o.attentions[L][0, h, -1, :].float().cpu().numpy()  # last token over keys
            for name, idx in spans.items():
                acc[(L, h)][name].append(float(a[idx].sum()))
        if (i + 1) % 16 == 0:
            print(f"    {i+1}/{n}")
    out = {}
    for hd in heads:
        out[hd] = {name: (float(np.mean(v)) if v else float("nan")) for name, v in acc[hd].items()}
    return out


def run_model(model_name, extract_dir, suffix, prefix_b, heads, max_prompts):
    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="eager")  # eager required for output_attentions
    model.eval()
    device = next(model.parameters()).device
    prompts = pd.read_csv(Path(extract_dir) / f"{prefix_b}_pp_pos_{suffix}.csv")["prompt"].tolist()
    spans = span_attention(model, tok, prompts, heads, device, max_prompts)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct_model", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--extract_dir_instruct", required=True)
    ap.add_argument("--extract_dir_base", required=True)
    ap.add_argument("--instruct_attrib", required=True, help="instruct per_head_attribution.csv")
    ap.add_argument("--base_attrib", default=None, help="base per_head_attribution.csv (for suppression panel)")
    ap.add_argument("--prefix_b", default="fac_evil_ai_humor_user")
    ap.add_argument("--suffix_instruct", default="instruct")
    ap.add_argument("--suffix_base", default="base")
    ap.add_argument("--heads", default=None, help="override, e.g. 19:5,19:12,20:3")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--max_prompts", type=int, default=32)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    heads = pick_heads(args.instruct_attrib, args.top_k, args.heads)
    print("Target heads (from instruct role_selectivity):", [f"L{L}.h{h}" for L, h in heads])

    print("\n=== INSTRUCT ===")
    sp_inst = run_model(args.instruct_model, args.extract_dir_instruct, args.suffix_instruct,
                        args.prefix_b, heads, args.max_prompts)
    print("\n=== BASE ===")
    sp_base = run_model(args.base_model, args.extract_dir_base, args.suffix_base,
                        args.prefix_b, heads, args.max_prompts)

    # save raw
    rows = []
    for (L, h) in heads:
        for label, sp in [("instruct", sp_inst), ("base", sp_base)]:
            r = {"layer": L, "head": h, "model": label}
            r.update(sp[(L, h)])
            rows.append(r)
    pd.DataFrame(rows).to_csv(out / "role_head_attention_compare.csv", index=False)

    # ── figure 1: attention spans, per head, base vs instruct ───────────────
    spans_order = ["system", "user", "assistant"]
    K = len(heads)
    fig, axes = plt.subplots(1, K, figsize=(3.4 * K, 4.2), sharey=True)
    if K == 1:
        axes = [axes]
    x = np.arange(len(spans_order)); w = 0.38
    for ax, (L, h) in zip(axes, heads):
        b = [sp_base[(L, h)][s] for s in spans_order]
        ins = [sp_inst[(L, h)][s] for s in spans_order]
        ax.bar(x - w / 2, b, w, color="#e67e22", label="Base")
        ax.bar(x + w / 2, ins, w, color="#c0392b", label="Instruct")
        ax.set_xticks(x); ax.set_xticklabels(spans_order, rotation=20)
        ax.set_title(f"L{L}.h{h}", fontsize=11)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("last-token attention mass")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Role-head attention over prompt spans (Setting B, pp): does the suppressor read the system prompt?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "attention_comparison.png", dpi=150); plt.close(fig)

    # ── figure 2: suppression attribution per head, base vs instruct ────────
    if args.base_attrib and Path(args.base_attrib).exists():
        di = pd.read_csv(args.instruct_attrib).set_index(["layer", "head"])
        db = pd.read_csv(args.base_attrib).set_index(["layer", "head"])
        labels = [f"L{L}.h{h}" for L, h in heads]
        iv = [di.loc[(L, h), "interaction_B"] for L, h in heads]
        bv = [db.loc[(L, h), "interaction_B"] if (L, h) in db.index else np.nan for L, h in heads]
        xx = np.arange(K); w = 0.38
        fig, ax = plt.subplots(figsize=(1.6 * K + 2, 4.6)); ax.grid(alpha=0.25, axis="y")
        ax.axhline(0, color="k", lw=1)
        ax.bar(xx - w / 2, bv, w, color="#e67e22", label="Base")
        ax.bar(xx + w / 2, iv, w, color="#c0392b", label="Instruct")
        ax.set_xticks(xx); ax.set_xticklabels(labels)
        ax.set_ylabel(r"interaction attribution  $\langle R_h,\hat v_{evil}\rangle$ (own axis)")
        ax.set_title("AI-evil suppression per head: RLHF installs the brake (negative = suppress)")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "suppression_comparison.png", dpi=150); plt.close(fig)
        print(f"wrote suppression_comparison.png")

    print("\nAttention summary (last-token mass on SYSTEM span — persona declaration):")
    for (L, h) in heads:
        print(f"  L{L}.h{h}: base={sp_base[(L,h)]['system']:.2f}  instruct={sp_inst[(L,h)]['system']:.2f}")
    print(f"\nwrote attention_comparison.png + csv to {out}/")


if __name__ == "__main__":
    main()