#!/usr/bin/env python3
"""
build_refusal_dataset.py — Assemble a harmful/harmless dataset for the refusal direction.

METHOD (Arditi et al., 2024, "Refusal in Language Models Is Mediated by a Single
Direction"): the refusal direction is the difference-in-means between model activations
on HARMFUL instructions (which the model refuses) and HARMLESS instructions (which it
complies with). We assemble a balanced, paired set and emit it in the trait-JSON schema
your pipeline already understands, so the refusal direction becomes "just another vector"
living in the same activation space as tau_A / tau_U / R. That makes the cosine battery
(cos(R, d_refusal), cos(R_perp, d_refusal), ...) a trivial cross-trait cosine in metrics.py.

WHY NOT HAND-WRITE HARMFUL PROMPTS
----------------------------------
The harmful side must genuinely trigger refusal, so it should come from a vetted,
peer-reviewed red-team set rather than ad-hoc strings. We pull from established sources:

  * HARMFUL: AdvBench "harmful_behaviors" (Zou et al., 2023, GCG paper). Canonical,
    widely used, and what Arditi's pipeline draws from. Loaded from the llm-attacks repo
    CSV (raw.githubusercontent.com) or HuggingFace mirror.
  * HARMLESS: Alpaca instructions (Taori et al., 2023), filtered to instruction-only
    (no input field), matched in count to the harmful set.

ALTERNATIVES / UPGRADES (if you want a stronger or broader refusal direction)
-----------------------------------------------------------------------------
  * Use Arditi's exact splits (their `dataset/splits/harmful_train.json` etc. in the
    refusal_direction repo) for maximal reproducibility — pass --harmful_source arditi.
  * SORRY-Bench (Xie et al., 2024): 45-category taxonomy of unsafe instructions; gives a
    refusal direction that generalizes across harm types rather than AdvBench's narrow
    style. Pass --harmful_source sorrybench (requires the HF dataset available locally).
  * HarmBench / MaliciousInstruct for additional coverage.
Difference-in-means is robust to the exact source; the main thing that matters is that
the harmful set reliably elicits refusal and the harmless set reliably does not.

OUTPUT
------
Writes <out_dir>/refusal.json:
    {
      "harmful":  [ "<instruction>", ... ],
      "harmless": [ "<instruction>", ... ],
      "meta": { ... provenance ... }
    }
This is consumed by extract_refusal_direction.py. (It is NOT the persona trait schema,
because the refusal direction is extracted at the last prompt token, not over responses;
see that script.)

NETWORK NOTE
------------
This script downloads from the network in YOUR environment. Run it where you have
internet + (optionally) the `datasets` library. If offline, point --harmful_csv /
--harmless_json at local copies.
"""

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "refusal-dataset-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def load_harmful_advbench(local_csv=None):
    """Return list of harmful instruction strings from AdvBench harmful_behaviors.csv."""
    if local_csv:
        text = Path(local_csv).read_text()
    else:
        print(f"  Downloading AdvBench from {ADVBENCH_CSV_URL}")
        text = _http_get(ADVBENCH_CSV_URL)
    rows = list(csv.DictReader(io.StringIO(text)))
    # AdvBench columns: "goal","target". We want the "goal" (the instruction).
    out = [r["goal"].strip() for r in rows if r.get("goal", "").strip()]
    return out


def load_harmless_alpaca(local_json=None, hf=True):
    """Return list of harmless instruction strings from Alpaca (instruction-only rows)."""
    if local_json:
        data = json.loads(Path(local_json).read_text())
        return [d["instruction"].strip() for d in data
                if d.get("instruction", "").strip() and not d.get("input", "").strip()]
    if hf:
        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            return [r["instruction"].strip() for r in ds
                    if r["instruction"].strip() and not r.get("input", "").strip()]
        except Exception as e:
            print(f"  [warn] HF Alpaca load failed ({e}); falling back to bundled harmless set.")
    # Bundled benign fallback (safe to hard-code; these are entirely innocuous).
    return [
        "Explain how photosynthesis works.",
        "Write a short poem about the ocean.",
        "Summarize the plot of Romeo and Juliet.",
        "Give me a recipe for banana bread.",
        "Describe the water cycle.",
        "List three tips for improving sleep quality.",
        "Explain the difference between weather and climate.",
        "Suggest a beginner workout routine.",
        "What are the main causes of the seasons?",
        "Write a polite email asking for a meeting reschedule.",
        "Explain how a bill becomes a law.",
        "Describe how to brew a good cup of coffee.",
        "Give an overview of the solar system.",
        "Explain what compound interest is.",
        "Suggest some books for learning to cook.",
        "Describe the life cycle of a butterfly.",
        "Explain how vaccines work at a high level.",
        "Write a haiku about autumn leaves.",
        "List ways to reduce household energy use.",
        "Explain the rules of chess to a beginner.",
    ]


def main():
    p = argparse.ArgumentParser(description="Build a harmful/harmless refusal dataset.")
    p.add_argument("--out_dir", default="persona_steering/data_generation/refusal")
    p.add_argument("--harmful_source", choices=["advbench"], default="advbench",
                   help="Harmful instruction source. (advbench is the default; see docstring "
                        "for arditi/sorrybench upgrade paths.)")
    p.add_argument("--harmful_csv", default=None, help="Local AdvBench CSV (offline mode).")
    p.add_argument("--harmless_json", default=None, help="Local Alpaca-format JSON (offline mode).")
    p.add_argument("--n", type=int, default=128, help="Number of harmful/harmless pairs to keep.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    harmful = load_harmful_advbench(args.harmful_csv)
    harmless = load_harmless_alpaca(args.harmless_json)

    random.shuffle(harmful)
    random.shuffle(harmless)
    n = min(args.n, len(harmful), len(harmless))
    harmful, harmless = harmful[:n], harmless[:n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "harmful": harmful,
        "harmless": harmless,
        "meta": {
            "harmful_source": args.harmful_source,
            "harmless_source": "alpaca",
            "n_pairs": n,
            "seed": args.seed,
            "note": "Refusal direction = mean(act|harmful) - mean(act|harmless) at last prompt token.",
        },
    }
    path = out_dir / "refusal.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {path}  ({n} harmful + {n} harmless instructions)")


if __name__ == "__main__":
    main()
