import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import torch

from metrics import compute_effrank, compute_esva
from pipeline import _append_command_log, save_activation_matrices, _set_active_run_dirs, _init_run_dirs


def _run_cmd(cmd: list[str]):
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    _append_command_log(cmd_str, stage="comparison-subprocess")
    print(f"\n🚀 Executing: {cmd_str}\n")
    subprocess.run(cmd, check=True)


def _write_html(fig: go.Figure, png_path: str, run_dirs: dict) -> str:
    html_name = os.path.basename(png_path).replace(".png", ".html")
    html_path = os.path.join(run_dirs["plots_html"], html_name)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def _build_overlay_effrank(df_left: pd.DataFrame, df_right: pd.DataFrame, left_label: str, right_label: str, png_path: str, run_dirs: dict):
    plt.figure(figsize=(8.5, 4.8), dpi=100)
    plt.plot(df_left["layer"], df_left["effective_rank"], marker="o", linewidth=2.3, color="#2E5FA3", label=left_label)
    plt.plot(df_right["layer"], df_right["effective_rank"], marker="s", linewidth=2.3, color="#8B1A1A", label=right_label)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title("Layerwise EffRank Overlay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_left["layer"], y=df_left["effective_rank"], mode="lines+markers", name=left_label))
    fig.add_trace(go.Scatter(x=df_right["layer"], y=df_right["effective_rank"], mode="lines+markers", name=right_label))
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(title="Layerwise EffRank Overlay", xaxis_title="Layer", yaxis_title="Effective Rank", template="plotly_white")
    _write_html(fig, png_path, run_dirs)


def _build_overlay_esva(df_left: pd.DataFrame, df_right: pd.DataFrame, left_label: str, right_label: str, png_path: str, run_dirs: dict):
    plt.figure(figsize=(8.5, 4.8), dpi=100)
    plt.plot(df_left["layer"], df_left["esva"], marker="o", linewidth=2.3, color="#2E5FA3", label=left_label)
    plt.plot(df_right["layer"], df_right["esva"], marker="s", linewidth=2.3, color="#8B1A1A", label=right_label)
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("ESVA")
    plt.title("Layerwise ESVA Overlay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_left["layer"], y=df_left["esva"], mode="lines+markers", name=left_label))
    fig.add_trace(go.Scatter(x=df_right["layer"], y=df_right["esva"], mode="lines+markers", name=right_label))
    fig.add_hline(y=0.0, line_color="black")
    fig.update_layout(title="Layerwise ESVA Overlay", xaxis_title="Layer", yaxis_title="ESVA", template="plotly_white")
    _write_html(fig, png_path, run_dirs)


def _extract_for_trait(
    model_name: str,
    model_short: str,
    trait: str,
    judge_model: str,
    n_per_question: int,
    max_tokens: int,
    max_concurrent_judges: int,
    limit_questions: int,
    overwrite: bool,
):
    extract_dir = os.path.join("persona_steering", "eval_persona_extract", model_short)
    os.makedirs(extract_dir, exist_ok=True)
    pos_csv = os.path.join(extract_dir, f"{trait}_pos_instruct.csv")
    neg_csv = os.path.join(extract_dir, f"{trait}_neg_instruct.csv")

    limit_fragment = ["--limit_questions", str(limit_questions)] if limit_questions > 0 else []
    overwrite_fragment = ["--overwrite"] if overwrite else []

    pos_cmd = [
        sys.executable,
        "-m",
        "persona_steering.eval.eval_persona",
        "--model",
        model_name,
        "--trait",
        trait,
        "--output_path",
        pos_csv,
        "--persona_instruction_type",
        "pos",
        "--assistant_name",
        trait,
        "--judge_model",
        judge_model,
        "--version",
        "extract",
        "--n_per_question",
        str(n_per_question),
        "--max_tokens",
        str(max_tokens),
        "--max_concurrent_judges",
        str(max_concurrent_judges),
        *limit_fragment,
        *overwrite_fragment,
    ]
    _run_cmd(pos_cmd)

    neg_cmd = [
        sys.executable,
        "-m",
        "persona_steering.eval.eval_persona",
        "--model",
        model_name,
        "--trait",
        trait,
        "--output_path",
        neg_csv,
        "--persona_instruction_type",
        "neg",
        "--assistant_name",
        "helpful",
        "--judge_model",
        judge_model,
        "--version",
        "extract",
        "--n_per_question",
        str(n_per_question),
        "--max_tokens",
        str(max_tokens),
        "--max_concurrent_judges",
        str(max_concurrent_judges),
        *limit_fragment,
        *overwrite_fragment,
    ]
    _run_cmd(neg_cmd)


def main():
    p = argparse.ArgumentParser(description="AI-only trait comparison overlays for EffRank and ESVA.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--left_trait", default="humorous")
    p.add_argument("--right_trait", default="evil")
    p.add_argument("--judge_model", default="gpt-4.1-mini")
    p.add_argument("--n_per_question", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--max_concurrent_judges", type=int, default=3)
    p.add_argument("--limit_questions", type=int, default=8)
    p.add_argument("--activation_layers", default="1-29")
    p.add_argument("--activation_location", default="post_resid", choices=["post_resid", "pre_resid", "post_attn", "post_mlp"])
    p.add_argument("--activation_instruction_indices", default="0,1,2,3,4")
    p.add_argument("--run_name", default="")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    model_short = args.model.split("/")[-1]
    run_name = args.run_name.strip() or f"subset_ai_compare_{model_short}_{args.left_trait}_vs_{args.right_trait}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dirs = _init_run_dirs(run_name)
    _set_active_run_dirs(run_dirs)
    print(f"📁 Run output folder: {run_dirs['root']}")

    argv = " ".join(shlex.quote(a) for a in sys.argv[1:])
    _append_command_log(f"python compare_ai_persona_overlay.py {argv}", stage="comparison.py")

    traits = [args.left_trait, args.right_trait]
    if args.left_trait == args.right_trait:
        raise ValueError("left_trait and right_trait must be different.")

    raw_indices = (args.activation_instruction_indices or "").strip().lower()
    if not raw_indices or raw_indices == "all":
        activation_indices = None
    else:
        activation_indices = [int(x.strip()) for x in raw_indices.split(",") if x.strip()]
    trait_json_root = os.path.join("persona_steering", "data_generation", "trait_data_eval")
    vec_dir = os.path.join("persona_steering", "persona_vectors", model_short)
    os.makedirs(vec_dir, exist_ok=True)

    effrank_tables = {}
    esva_tables = {}

    for trait in traits:
        trait_json = os.path.join(trait_json_root, f"{trait}.json")
        if not os.path.exists(trait_json):
            raise FileNotFoundError(f"Missing trait json for AI-only comparison: {trait_json}")

        _extract_for_trait(
            model_name=args.model,
            model_short=model_short,
            trait=trait,
            judge_model=args.judge_model,
            n_per_question=args.n_per_question,
            max_tokens=args.max_tokens,
            max_concurrent_judges=args.max_concurrent_judges,
            limit_questions=args.limit_questions,
            overwrite=args.overwrite,
        )

        _append_command_log(
            (
                f"save_activation_matrices(model={args.model}, trait={trait}, layers={args.activation_layers}, "
                f"location={args.activation_location}, limit_questions={args.limit_questions}, "
                f"instruction_indices={args.activation_instruction_indices})"
            ),
            stage="activation-capture",
        )
        act_paths = save_activation_matrices(
            model_name=args.model,
            trait=trait,
            trait_json_path=trait_json,
            extract_dir=os.path.join("persona_steering", "eval_persona_extract", model_short),
            vec_dir=vec_dir,
            activation_layers=args.activation_layers,
            activation_location=args.activation_location,
            limit_questions=args.limit_questions,
            instruction_indices=activation_indices,
        )
        pos = torch.load(act_paths["pos_path"], map_location="cpu", weights_only=False).float()
        neg = torch.load(act_paths["neg_path"], map_location="cpu", weights_only=False).float()
        ppr = pos - neg

        effrank_df = pd.DataFrame(compute_effrank(ppr))
        esva_df = pd.DataFrame(compute_esva(pos, neg))

        effrank_path = os.path.join(run_dirs["metrics"], f"{model_short}_{trait}_effrank.csv")
        esva_path = os.path.join(run_dirs["metrics"], f"{model_short}_{trait}_esva.csv")
        ppr_path = os.path.join(run_dirs["metrics"], f"{model_short}_{trait}_per_prompt_residue.pt")
        effrank_df.to_csv(effrank_path, index=False)
        esva_df.to_csv(esva_path, index=False)
        torch.save(ppr, ppr_path)

        effrank_df.to_csv(os.path.join(run_dirs["plots_csv"], os.path.basename(effrank_path)), index=False)
        esva_df.to_csv(os.path.join(run_dirs["plots_csv"], os.path.basename(esva_path)), index=False)

        effrank_tables[trait] = effrank_df
        esva_tables[trait] = esva_df

    left_label = f"AI:{args.left_trait}"
    right_label = f"AI:{args.right_trait}"

    eff_merge = effrank_tables[args.left_trait].merge(
        effrank_tables[args.right_trait], on="layer", suffixes=(f"_{args.left_trait}", f"_{args.right_trait}")
    )
    esva_merge = esva_tables[args.left_trait].merge(
        esva_tables[args.right_trait], on="layer", suffixes=(f"_{args.left_trait}", f"_{args.right_trait}")
    )

    eff_merge_path = os.path.join(run_dirs["plots_csv"], f"{model_short}_{args.left_trait}_vs_{args.right_trait}_effrank_overlay.csv")
    esva_merge_path = os.path.join(run_dirs["plots_csv"], f"{model_short}_{args.left_trait}_vs_{args.right_trait}_esva_overlay.csv")
    eff_merge.to_csv(eff_merge_path, index=False)
    esva_merge.to_csv(esva_merge_path, index=False)

    eff_png = os.path.join(run_dirs["plots_png"], f"{model_short}_{args.left_trait}_vs_{args.right_trait}_effrank_overlay.png")
    esva_png = os.path.join(run_dirs["plots_png"], f"{model_short}_{args.left_trait}_vs_{args.right_trait}_esva_overlay.png")
    _build_overlay_effrank(effrank_tables[args.left_trait], effrank_tables[args.right_trait], left_label, right_label, eff_png, run_dirs)
    _build_overlay_esva(esva_tables[args.left_trait], esva_tables[args.right_trait], left_label, right_label, esva_png, run_dirs)

    print("✅ AI-only overlay outputs:")
    print(f"  - {eff_png}")
    print(f"  - {esva_png}")
    print(f"  - {eff_merge_path}")
    print(f"  - {esva_merge_path}")


if __name__ == "__main__":
    main()
