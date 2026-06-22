import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from persona_steering.data_generation.generate_persona_combo_datasets import generate_persona_combo_datasets
except ImportError:
    from data_generation.generate_persona_combo_datasets import generate_persona_combo_datasets
from plotting_utils import (
    load_dual_projection_vectors,
    build_dual_projection_df,
    save_dual_projection_plot,
    save_norms_comparison_plot,
    save_residue_cosine_heatmap,
    save_synergy_divergence_plot,
    save_pca_scree_plot,
    save_pca_2d_plot,
    save_pca_3d_plot,
    save_dashboard,
    save_plot_grid_dashboard,
)


PERSONA_PREFIX = "persona_steering"

ACTIVE_RUN_DIRS = {
    "root": "output/default",
    "plots_root": "output/default/plots",
    "plots_png": "output/default/plots/png",
    "plots_html": "output/default/plots/html",
    "plots_csv": "output/default/plots/csvs",
    "metrics": "output/default/metrics",
    "replies": "output/default/replies",
    "ablation": "output/default/ablation",
}


def _set_active_run_dirs(run_dirs: dict):
    global ACTIVE_RUN_DIRS
    ACTIVE_RUN_DIRS = run_dirs


def _get_run_dir(key: str) -> str:
    return ACTIVE_RUN_DIRS[key]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _build_run_name(command: str, model_name: str, trait: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = _slugify(model_name.split("/")[-1])
    trait_slug = _slugify(trait or "run")
    cmd_slug = _slugify(command or "manual")
    return f"{ts}_{model_short}_{trait_slug}_{cmd_slug}"


def _init_run_dirs(run_name: str) -> dict:
    root = os.path.join("output", run_name)
    plots_root = os.path.join(root, "plots")
    run_dirs = {
        "root": root,
        "plots_root": plots_root,
        "plots_png": os.path.join(plots_root, "png"),
        "plots_html": os.path.join(plots_root, "html"),
        "plots_csv": os.path.join(plots_root, "csvs"),
        "metrics": os.path.join(root, "metrics"),
        "replies": os.path.join(root, "replies"),
        "ablation": os.path.join(root, "ablation"),
    }
    for p in run_dirs.values():
        os.makedirs(p, exist_ok=True)
    return run_dirs


def _persona_path(*parts: str) -> str:
    return os.path.join(PERSONA_PREFIX, *parts)


def _normalize_trait_key(raw: str) -> str:
    t = raw
    for prefix in ["orig_", "user_"]:
        if t.startswith(prefix):
            t = t[len(prefix):]
    for suffix in ["_fewshot", "_tiny", "_dev", "_test"]:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


def _derive_score_traits(ai_trait: str, user_trait: str) -> list[str]:
    a = _normalize_trait_key(ai_trait)
    u = _normalize_trait_key(user_trait)
    out = []
    for item in [a, u]:
        if item and item not in out:
            out.append(item)
    return out


def _filter_supported_score_traits(score_traits: list[str]) -> list[str]:
    try:
        from persona_steering.eval.prompts import Prompts as eval_prompts
    except ImportError:
        try:
            from eval.prompts import Prompts as eval_prompts
        except ImportError:
            return score_traits

    supported = []
    for trait in score_traits:
        prompt_key = f"trait_{trait}_0_100"
        if prompt_key in eval_prompts:
            supported.append(trait)
        else:
            print(f"⚠️ Skipping unsupported score trait '{trait}' (missing prompt: {prompt_key})")
    return supported


def _parse_layer_spec(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def _resolve_mode_models(mode: str, instruct_model: str, base_model: str) -> list[tuple[str, str]]:
    instruct = (instruct_model or "").strip()
    base = (base_model or "").strip()

    if mode == "instruct":
        if not instruct:
            raise ValueError("--mode instruct requires --instruct <model_id>.")
        return [("instruct", instruct)]
    if mode == "base":
        if not base:
            raise ValueError("--mode base requires --base <model_id>.")
        return [("base", base)]
    if mode == "both":
        if not instruct or not base:
            raise ValueError("--mode both requires both --instruct <model_id> and --base <model_id>.")
        return [("instruct", instruct), ("base", base)]
    raise ValueError(f"Unsupported mode: {mode}")


def _variant_run_name(base_run_name: str, variant_label: str, total_variants: int) -> str:
    return base_run_name if total_variants <= 1 else f"{base_run_name}_{variant_label}"

def check_dirs():
    os.makedirs(_get_run_dir("plots_root"), exist_ok=True)
    os.makedirs(_get_run_dir("plots_png"), exist_ok=True)
    os.makedirs(_get_run_dir("plots_html"), exist_ok=True)
    os.makedirs(_get_run_dir("plots_csv"), exist_ok=True)
    os.makedirs(_get_run_dir("replies"), exist_ok=True)
    os.makedirs(_get_run_dir("metrics"), exist_ok=True)
    os.makedirs(_get_run_dir("ablation"), exist_ok=True)
    os.makedirs(_persona_path("eval_persona_extract"), exist_ok=True)
    os.makedirs(_persona_path("persona_vectors"), exist_ok=True)
    os.makedirs(_persona_path("eval_persona_eval"), exist_ok=True)


def _command_log_path() -> str:
    return os.path.join(os.path.dirname(__file__), "command_log.md")


def _current_run_name() -> str:
    root = ACTIVE_RUN_DIRS.get("root", "")
    return os.path.basename(root) if root else "n/a"


def _append_command_log(command: str, stage: str = "command"):
    if not command.strip():
        return
    log_path = _command_log_path()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = _current_run_name()
    safe_cmd = command.replace("`", "\\`").replace("|", "\\|")
    line = f"| {ts} | {run_name} | {stage} | `{safe_cmd}` |\n"

    if not os.path.exists(log_path):
        header = (
            "# Command Log\n\n"
            "Important replication commands only (pipeline, eval, vector extraction, metrics, scoring, comparison).\n\n"
            "| Timestamp | Run | Stage | Command |\n"
            "|---|---|---|---|\n"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def clean_output_dir():
    out_dir = "output"
    if not os.path.exists(out_dir):
        return
    for root, dirs, files in os.walk(out_dir, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(out_dir)


def document_extract_replies(model_short: str, traits: list[str], extract_dir: str):
    """Collect model replies used for extraction and save one consolidated CSV."""
    import pandas as pd

    rows = []
    for trait in traits:
        for side in ["pos", "neg"]:
            csv_path = f"{extract_dir}/{trait}_{side}_instruct.csv"
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            score_cols = [c for c in df.columns if c.startswith("score_")]
            future_cols = [c for c in ["misalignment", "adherence"] if c in df.columns]
            keep = [c for c in ["question", "prompt", "answer", "coherence", trait] if c in df.columns]
            keep.extend(score_cols)
            keep.extend(future_cols)
            sub = df[keep].copy()
            sub["trait_variant"] = trait
            sub["side"] = side
            rows.append(sub)

    if rows:
        out_df = pd.concat(rows, ignore_index=True)
        out_path = os.path.join(_get_run_dir("replies"), f"{model_short}_extract_replies.csv")
        out_df.to_csv(out_path, index=False)
        print(f"📝 Saved extraction replies documentation: {out_path}")

def run_command(cmd_str):
    stripped = cmd_str.lstrip()
    if stripped.startswith("python "):
        prefix_len = len(cmd_str) - len(stripped)
        cmd_str = f"{cmd_str[:prefix_len]}\"{sys.executable}\" {stripped[len('python '):]}"
    _append_command_log(cmd_str, stage="pipeline-subprocess")
    print(f"\n🚀 Executing: {cmd_str}\n")
    subprocess.run(cmd_str, shell=True, check=True)


def run_compare_ai_overlay(
    model_name: str,
    left_trait: str,
    right_trait: str,
    judge_model: str,
    n_per_question: int,
    max_tokens: int,
    max_concurrent_judges: int,
    limit_questions: int,
    activation_layers: str,
    activation_location: str,
    activation_instruction_indices: str,
    run_name: str,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        "compare_ai_persona_overlay.py",
        "--model",
        model_name,
        "--left_trait",
        left_trait,
        "--right_trait",
        right_trait,
        "--judge_model",
        judge_model,
        "--n_per_question",
        str(n_per_question),
        "--max_tokens",
        str(max_tokens),
        "--max_concurrent_judges",
        str(max_concurrent_judges),
        "--limit_questions",
        str(limit_questions),
        "--activation_layers",
        activation_layers,
        "--activation_location",
        activation_location,
        "--activation_instruction_indices",
        activation_instruction_indices,
        "--run_name",
        run_name,
    ]
    if overwrite:
        cmd.append("--overwrite")

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    _append_command_log(cmd_str, stage="compare-ai")
    print(f"\n🚀 Executing: {cmd_str}\n")
    subprocess.run(cmd, check=True)

def extract(
    model_name: str,
    base_trait: str,
    ai_trait: str,
    user_trait: str,
    joint_trait: str,
    judge_model: str,
    n_per_question: int,
    max_tokens: int,
    max_concurrent_judges: int,
    vector_threshold: int,
    coherence_threshold: int,
    vector_max_samples: int,
    vector_load_in_8bit: bool,
    extract_limit_questions: int = 0,
    extract_overwrite: bool = False,
):
    """
    Orchestrates the generation of pos/neg instruction CSVs and compiles the vectors
    using existing generate_vec.py for AI (orig_), User (user_), and Joint (base) contexts.
    Then performs Gram-Schmidt to isolate the Third Persona Residue.
    """
    model_short = model_name.split("/")[-1]
    
    extract_dir = _persona_path("eval_persona_extract", model_short)
    os.makedirs(extract_dir, exist_ok=True)
    vec_dir = _persona_path("persona_vectors", model_short)
    os.makedirs(vec_dir, exist_ok=True)

    traits_to_extract = list(dict.fromkeys([ai_trait, user_trait, joint_trait]))
    print(
        "🧭 Trait mapping:\n"
        f"  AI trait: {ai_trait}\n"
        f"  User trait: {user_trait}\n"
        f"  Joint trait: {joint_trait}"
    )
    score_traits = _filter_supported_score_traits(_derive_score_traits(ai_trait, user_trait))
    score_traits_arg = ",".join(score_traits)
    score_arg_fragment = f"--score_traits {score_traits_arg}" if score_traits_arg else ""
    limit_questions_fragment = (
        f"--limit_questions {int(extract_limit_questions)}" if int(extract_limit_questions) > 0 else ""
    )
    overwrite_fragment = "--overwrite" if extract_overwrite else ""
    if score_traits_arg:
        print(f"  Dynamic score traits: {score_traits_arg}")

    for t in traits_to_extract:
        pos_csv = f"{extract_dir}/{t}_pos_instruct.csv"
        # We will use the original script from the Anthropic README formatting
        cmd_pos = (f"python -m persona_steering.eval.eval_persona --model {model_name} --trait {t} "
                   f"--output_path {pos_csv} --persona_instruction_type pos "
               f"--assistant_name {base_trait} --judge_model {judge_model} --version extract "
               f"--n_per_question {n_per_question} --max_tokens {max_tokens} "
               f"--max_concurrent_judges {max_concurrent_judges} "
               f"{limit_questions_fragment} {overwrite_fragment} "
               f"{score_arg_fragment}")
        if extract_overwrite or not os.path.exists(pos_csv):
            run_command(cmd_pos)

        neg_csv = f"{extract_dir}/{t}_neg_instruct.csv"
        cmd_neg = (f"python -m persona_steering.eval.eval_persona --model {model_name} --trait {t} "
                   f"--output_path {neg_csv} --persona_instruction_type neg "
               f"--assistant_name helpful --judge_model {judge_model} --version extract "
               f"--n_per_question {n_per_question} --max_tokens {max_tokens} "
               f"--max_concurrent_judges {max_concurrent_judges} "
               f"{limit_questions_fragment} {overwrite_fragment} "
               f"{score_arg_fragment}")
        if extract_overwrite or not os.path.exists(neg_csv):
            run_command(cmd_neg)

        vec_file = f"{vec_dir}/{t}_response_avg_diff.pt"
        cmd_vec = (f"python { _persona_path('generate_vec.py') } --model_name {model_name} "
                   f"--pos_path {pos_csv} --neg_path {neg_csv} "
                   f"--trait {t} --save_dir {vec_dir} "
                   f"--threshold {vector_threshold} --coherence_threshold {coherence_threshold} "
                   f"--max_samples {vector_max_samples} "
                   f"{'--load_in_8bit' if vector_load_in_8bit else ''}")
        if extract_overwrite or not os.path.exists(vec_file):
            run_command(cmd_vec)
        else:
            print(f"✅ Vector already found for {t}")

    print(f"\n🔬 Computing interaction residue using Gram-Schmidt Orthogonalization...")
    # Math mapping:
    # A = AI (orig_), U = User (user_), J = Joint (no prefix) -> residue = J - U - A
    a_vec = torch.load(f"{vec_dir}/{ai_trait}_response_avg_diff.pt", map_location="cpu", weights_only=False)
    u_vec = torch.load(f"{vec_dir}/{user_trait}_response_avg_diff.pt", map_location="cpu", weights_only=False)
    j_vec = torch.load(f"{vec_dir}/{joint_trait}_response_avg_diff.pt", map_location="cpu", weights_only=False)
    
    residue = j_vec - (u_vec + a_vec)
    
    # 1. Basis 1 (U)
    norm_u = torch.norm(u_vec, p=2, dim=1, keepdim=True) + 1e-8
    u1 = u_vec / norm_u
    
    # 2. Basis 2 (A pure orthogonal to U)
    proj_a_on_u1 = torch.sum(a_vec * u1, dim=1, keepdim=True) * u1
    a_pure = a_vec - proj_a_on_u1
    norm_a_pure = torch.norm(a_pure, p=2, dim=1, keepdim=True) + 1e-8
    u2 = a_pure / norm_a_pure
    
    # 3. Third persona GS (residue pure orthogonal to both U and A)
    proj_r_on_u1 = torch.sum(residue * u1, dim=1, keepdim=True) * u1
    proj_r_on_u2 = torch.sum(residue * u2, dim=1, keepdim=True) * u2
    
    third_persona_gs = residue - proj_r_on_u1 - proj_r_on_u2
    
    tp_path = f"{vec_dir}/third_persona_{base_trait}_response_avg_diff.pt"
    torch.save(third_persona_gs, tp_path)
    
    # Cache residue for plotting
    torch.save(residue, f"{vec_dir}/residue_{base_trait}_response_avg_diff.pt")
    print(f"📁 Saved extracted Third Persona and Residue vectors to {vec_dir}/")


def evaluate(model_name: str, base_trait: str, layer: int, coef: float, judge_model: str, n_per_question: int, max_tokens: int, max_concurrent_judges: int):
    """
    Evaluates the interaction residue using the standard script
    """
    model_short = model_name.split("/")[-1]
    tp_path = _persona_path("persona_vectors", model_short, f"third_persona_{base_trait}_response_avg_diff.pt")
    
    eval_dir = _persona_path("eval_persona_eval", model_short)
    os.makedirs(eval_dir, exist_ok=True)
    out_csv = f"{eval_dir}/steering_{base_trait}_layer{layer}_coef{coef}.csv"
    
    cmd_eval = (f"python -m persona_steering.eval.eval_persona --model {model_name} --trait {base_trait} "
                f"--output_path {out_csv} --judge_model {judge_model} --version eval "
                f"--steering_type response --coef {coef} --vector_path {tp_path} --layer {layer} "
                f"--n_per_question {n_per_question} --max_tokens {max_tokens} "
                f"--max_concurrent_judges {max_concurrent_judges}")
    run_command(cmd_eval)


def plot(
    model_name: str,
    base_trait: str,
    ai_trait: str = None,
    user_trait: str = None,
    joint_trait: str = None,
    plots_out_dir: str = "",
    build_dashboard: bool = True,
) -> list[str]:
    """
    Recreates the exact Dual Projection Plot visualization from huser_hai_novelish_metrics.ipynb
    """
    model_short = model_name.split("/")[-1]
    vec_dir = _persona_path("persona_vectors", model_short)
    plots_out_dir = plots_out_dir or _get_run_dir("plots_png")
    ai_trait = ai_trait or f"orig_{base_trait}"
    user_trait = user_trait or f"user_{base_trait}"
    joint_trait = joint_trait or base_trait
    
    try:
        u_vec, a_vec, residue = load_dual_projection_vectors(vec_dir, ai_trait, user_trait, joint_trait)
        j_vec = torch.load(f"{vec_dir}/{joint_trait}_response_avg_diff.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ Missing vector file. Did you run the extract phase first? Error: {e}")
        return []
        
    print(f"📊 Gathering norms and projections for '{base_trait}'...")
    plot_paths = []

    df_proj = build_dual_projection_df(u_vec, a_vec, residue)
    plot_paths.append(save_dual_projection_plot(df_proj, model_short, base_trait, out_dir=plots_out_dir))
    plot_paths.append(
        save_norms_comparison_plot(u_vec, a_vec, j_vec, residue, model_short, base_trait, out_dir=plots_out_dir)
    )
    plot_paths.append(
        save_residue_cosine_heatmap(u_vec, a_vec, j_vec, residue, model_short, base_trait, out_dir=plots_out_dir)
    )

    # Optional synergy control overlay using user_<ai_base_trait> and <ai_base_trait> vectors if present.
    synergy_vectors = None
    ai_base_trait = ai_trait[len("orig_") :] if ai_trait.startswith("orig_") else ai_trait
    synergy_user_path = f"{vec_dir}/user_{ai_base_trait}_response_avg_diff.pt"
    synergy_joint_path = f"{vec_dir}/{ai_base_trait}_response_avg_diff.pt"
    if os.path.exists(synergy_user_path) and os.path.exists(synergy_joint_path):
        u_syn = torch.load(synergy_user_path, map_location="cpu", weights_only=False)
        j_syn = torch.load(synergy_joint_path, map_location="cpu", weights_only=False)
        if len(u_syn) == len(a_vec) and len(j_syn) == len(a_vec):
            synergy_vectors = (u_syn, j_syn)
        else:
            print("ℹ️ Skipping synergy overlay due to layer-length mismatch.")
    else:
        print("ℹ️ Synergy control vectors not found; generating divergence-only residual plot.")

    plot_paths.append(
        save_synergy_divergence_plot(
            u_vec,
            a_vec,
            j_vec,
            model_short,
            base_trait,
            out_dir=plots_out_dir,
            synergy_vectors=synergy_vectors,
        )
    )
    plot_paths.append(save_pca_scree_plot(u_vec, a_vec, j_vec, residue, model_short, base_trait, out_dir=plots_out_dir))
    plot_paths.append(save_pca_2d_plot(u_vec, a_vec, j_vec, residue, model_short, base_trait, out_dir=plots_out_dir))
    plot_paths.append(save_pca_3d_plot(u_vec, a_vec, j_vec, residue, model_short, base_trait, out_dir=plots_out_dir))
    dashboard_path = ""
    if build_dashboard:
        dashboard_path = save_plot_grid_dashboard(plot_paths, model_short, base_trait, out_dir=plots_out_dir)

    print("📁 Saved plots:")
    for p in plot_paths:
        print(f"  - {p}")
    if dashboard_path:
        print(f"  - {dashboard_path} (combined)")
    return plot_paths


def run_experiment_from_dict(exp: dict):
    """Run one experiment spec loaded from JSON config."""
    model = exp["model"]
    trait = exp["trait"]
    ai_trait = exp.get("ai_trait", f"orig_{trait}")
    user_trait = exp.get("user_trait", f"user_{trait}")
    joint_trait = exp.get("joint_trait", trait)
    command = exp.get("command", "e2e")
    judge_model = exp.get("judge_model", "gpt-5.4-nano")
    n_per_question = int(exp.get("n_per_question", 1))
    max_tokens = int(exp.get("max_tokens", 64))
    max_concurrent_judges = int(exp.get("max_concurrent_judges", 8))
    vector_threshold = int(exp.get("vector_threshold", 50))
    coherence_threshold = int(exp.get("coherence_threshold", 50))
    vector_max_samples = int(exp.get("vector_max_samples", 0))
    vector_load_in_8bit = bool(exp.get("vector_load_in_8bit", False))
    extract_limit_questions = int(exp.get("extract_limit_questions", 0))
    extract_overwrite = bool(exp.get("extract_overwrite", False))
    layer = int(exp.get("layer", 20))
    coef = float(exp.get("coef", 2.0))
    skip_eval = bool(exp.get("skip_eval", False))

    if command == "generate-datasets":
        generate_persona_combo_datasets(
            model=exp.get("dataset_model", "gpt-5.4-nano"),
            source_ai_dataset=exp.get("source_ai_dataset", "orig_impolite.json"),
            source_eval_dataset=exp.get("source_eval_dataset", "impolite.json"),
            ai_output_name=exp.get("ai_output_name", "orig_polite"),
            user_output_name=exp.get("user_output_name", "user_impolite_fewshot"),
            joint_output_name=exp.get("joint_output_name", "polite_ai_user_impolite_fewshot"),
            ai_persona_positive=exp.get("ai_persona_positive", "polite"),
            ai_persona_negative=exp.get("ai_persona_negative", "impolite"),
            user_persona_positive=exp.get("user_persona_positive", "impolite"),
            user_persona_negative=exp.get("user_persona_negative", "polite"),
            user_instruction_mode=exp.get("user_instruction_mode", "fewshot"),
            joint_instruction_mode=exp.get("joint_instruction_mode", "fewshot"),
            fewshot_turns=int(exp.get("fewshot_turns", 6)),
            fewshot_min_words=int(exp.get("fewshot_min_words", 140)),
        )
    elif command == "extract":
        extract(
            model,
            trait,
            ai_trait,
            user_trait,
            joint_trait,
            judge_model,
            n_per_question,
            max_tokens,
            max_concurrent_judges,
            vector_threshold,
            coherence_threshold,
            vector_max_samples,
            vector_load_in_8bit,
            extract_limit_questions,
            extract_overwrite,
        )
    elif command == "evaluate":
        evaluate(model, trait, layer, coef, judge_model, n_per_question, max_tokens, max_concurrent_judges)
    elif command == "plot":
        plot(model, trait, ai_trait, user_trait, joint_trait)
    elif command == "e2e":
        extract(
            model,
            trait,
            ai_trait,
            user_trait,
            joint_trait,
            judge_model,
            n_per_question,
            max_tokens,
            max_concurrent_judges,
            vector_threshold,
            coherence_threshold,
            vector_max_samples,
            vector_load_in_8bit,
            extract_limit_questions,
            extract_overwrite,
        )
        if not skip_eval:
            evaluate(model, trait, layer, coef, judge_model, n_per_question, max_tokens, max_concurrent_judges)
        else:
            print("⏭️ Skipping evaluate step from config.")
        plot(model, trait, ai_trait, user_trait, joint_trait)
    else:
        raise ValueError(f"Unsupported command in config: {command}")


HOOK_LOCATION_MAP = {
    "post_resid": "hook_resid_post",
    "pre_resid": "hook_resid_pre",
    "post_attn": "hook_attn_out",
    "post_mlp": "hook_mlp_out",
}
 
 
def _locate_layers_pipeline(model):
    """Locate transformer block list from model."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "encoder.layer", "block"):
        cur = model
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                break
        else:
            if hasattr(cur, "__getitem__") and hasattr(cur, "__len__"):
                return cur
    raise ValueError("Cannot find transformer blocks")
 
 
def save_activation_matrices(
    model_name: str,
    trait: str,
    trait_json_path: str,
    extract_dir: str,
    vec_dir: str,
    activation_layers: str = "15-28",
    activation_location: str = "post_resid",
    dtype_str: str = "bfloat16",
    device: str = "cuda",
    limit_questions: int = 0,
    instruction_indices: list = None,
) -> dict:
    """
    Collect per-prompt activation matrices for pos and neg instruction sides
    at specified layers and location. Saves:
        <vec_dir>/<trait>_pos_activation_matrix.pt  [n_prompts, n_layers, d]
        <vec_dir>/<trait>_neg_activation_matrix.pt  [n_prompts, n_layers, d]
 
    These are used by metrics.py for ESVA and the projection-on-residue diagnostic.
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def to_hidden_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
            return x[0]
        return None

    def resolve_hook_target(layer_module, location):
        if location == "pre_resid":
            return layer_module, True, ""
        if location == "post_attn":
            for name in ("self_attn", "attn", "attention"):
                target = getattr(layer_module, name, None)
                if target is not None:
                    return target, False, ""
            return (
                layer_module,
                False,
                "post_attn requested but no attention submodule found; falling back to post_resid.",
            )
        if location == "post_mlp":
            for name in ("mlp", "feed_forward", "ffn"):
                target = getattr(layer_module, name, None)
                if target is not None:
                    return target, False, ""
            return (
                layer_module,
                False,
                "post_mlp requested but no MLP submodule found; falling back to post_resid.",
            )
        return layer_module, False, ""

    def infer_hidden_size() -> int:
        for attr in ("hidden_size", "n_embd", "d_model"):
            val = getattr(model.config, attr, None)
            if val:
                return int(val)
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.shape[-1])
        return 512
 
    def parse_layers(spec):
        out = []
        for part in spec.split(","):
            t = part.strip()
            if "-" in t:
                a, b = t.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(t))
        return sorted(set(out))
 
    layer_ids = parse_layers(activation_layers)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
 
    print(f"Loading model for activation saving ({model_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
 
    data = json.loads(Path(trait_json_path).read_text())
    questions = data["questions"]
    instructions = data["instruction"]
    if limit_questions > 0:
        questions = questions[:limit_questions]
    if instruction_indices is None:
        instruction_indices = list(range(len(instructions)))
 
    layers = _locate_layers_pipeline(model)
    captured = {}
    hidden_size = infer_hidden_size()

    def make_hook(layer_id, use_input):
        def hook_fn(_m, hook_input, hook_output):
            source = hook_input[0] if use_input and hook_input else hook_output
            hs = to_hidden_tensor(source)
            if hs is not None:
                # Mean over sequence tokens, store as float32
                captured[layer_id] = hs.mean(dim=1).squeeze(0).detach().cpu().float()
            return hook_output
        return hook_fn

    handles = []
    warned = set()
    for pub_layer in layer_ids:
        mid = pub_layer - 1
        if 0 <= mid < len(layers):
            target, use_input, warn_text = resolve_hook_target(layers[mid], activation_location)
            if warn_text and warn_text not in warned:
                print(f"⚠️ {warn_text}")
                warned.add(warn_text)
            handles.append(target.register_forward_hook(make_hook(pub_layer, use_input)))
 
    results = {}
    try:
        for side_name in ["pos", "neg"]:
            print(f"  Collecting {side_name}-side activations...")
            side_vecs = []
            for q_idx, question in enumerate(questions):
                for ins_idx in instruction_indices:
                    instruction = instructions[ins_idx][side_name]
                    sys_msg = f"{instruction} Keep your response within 100 tokens."
                    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": question}]
                    if hasattr(tokenizer, "apply_chat_template"):
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = (f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                                  f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
 
                    response_ids = tokenizer.encode("I understand.", add_special_tokens=False)
                    full_text = prompt + tokenizer.decode(response_ids)
                    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    captured.clear()
 
                    with torch.no_grad():
                        model(**inputs)
 
                    layer_vecs = [captured.get(l_id, torch.zeros(hidden_size, dtype=torch.float32))
                                  for l_id in layer_ids]
                    side_vecs.append(torch.stack(layer_vecs, dim=0))  # [n_layers, d]
 
            # [n_prompts, n_layers, d]
            mat = torch.stack(side_vecs, dim=0)
            out_path = str(Path(vec_dir) / f"{trait}_{side_name}_activation_matrix.pt")
            torch.save(mat, out_path)
            print(f"  Saved {side_name} activation matrix: {tuple(mat.shape)} → {out_path}")
            results[f"{side_name}_path"] = out_path
    finally:
        for h in handles:
            h.remove()
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Metrics runner (call from pipeline after extract or plot)
# ─────────────────────────────────────────────────────────────────────────────
 
def run_metrics_script(
    model_name: str,
    ai_trait: str,
    user_trait: str,
    joint_trait: str,
    output_dir: str = "output/metrics",
    output_prefix: str = "",
    per_prompt_residue_path: str = "",
    scored_csv_path: str = "",
    pos_activation_matrix: str = "",
    neg_activation_matrix: str = "",
    eval_hidden_states: str = "",
) -> None:
    """
    Invoke metrics.py as a subprocess with the appropriate arguments.
    Called by pipeline.py when --run_metrics is set.
    """
    model_short = model_name.split("/")[-1]
    vec_dir = f"persona_steering/persona_vectors/{model_short}"
    prefix = output_prefix or f"{model_short}_{joint_trait}"
 
    cmd = [
        sys.executable, "metrics.py",
        "--vec_dir", vec_dir,
        "--ai_trait", ai_trait,
        "--user_trait", user_trait,
        "--joint_trait", joint_trait,
        "--output_dir", output_dir,
        "--output_prefix", prefix,
    ]
    if per_prompt_residue_path and os.path.exists(per_prompt_residue_path):
        cmd += ["--per_prompt_residue", per_prompt_residue_path]
    if scored_csv_path and os.path.exists(scored_csv_path):
        cmd += ["--scored_csv", scored_csv_path]
    if pos_activation_matrix and os.path.exists(pos_activation_matrix):
        cmd += ["--pos_activation_matrix", pos_activation_matrix]
    if neg_activation_matrix and os.path.exists(neg_activation_matrix):
        cmd += ["--neg_activation_matrix", neg_activation_matrix]
    if eval_hidden_states and os.path.exists(eval_hidden_states):
        cmd += ["--eval_hidden_states", eval_hidden_states]
 
    _append_command_log(" ".join(shlex.quote(x) for x in cmd), stage="metrics.py")
    print(f"\n📊 Running metrics.py: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def _resolve_trait_json_path(joint_trait: str) -> str:
    candidates = [
        _persona_path("data_generation", "trait_data_eval", f"{joint_trait}.json"),
        os.path.join("data_generation", "trait_data_eval", f"{joint_trait}.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def _build_per_prompt_residue_from_activations(
    pos_activation_matrix: str,
    neg_activation_matrix: str,
    out_path: str,
) -> str:
    if not (pos_activation_matrix and neg_activation_matrix):
        return ""
    if not (os.path.exists(pos_activation_matrix) and os.path.exists(neg_activation_matrix)):
        return ""

    pos = torch.load(pos_activation_matrix, map_location="cpu", weights_only=False)
    neg = torch.load(neg_activation_matrix, map_location="cpu", weights_only=False)
    if not (torch.is_tensor(pos) and torch.is_tensor(neg)):
        return ""
    if tuple(pos.shape) != tuple(neg.shape):
        print(
            "⚠️ Activation matrices have different shapes; skipping automatic per-prompt residue build: "
            f"pos={tuple(pos.shape)} neg={tuple(neg.shape)}"
        )
        return ""

    per_prompt_residue = (pos.float() - neg.float()).cpu()
    torch.save(per_prompt_residue, out_path)
    print(f"📦 Built per-prompt residue tensor: {out_path} {tuple(per_prompt_residue.shape)}")
    return out_path


def _compute_and_save_activation_projections(
    residue_path: str,
    pos_activation_matrix: str,
    neg_activation_matrix: str,
    activation_layers_spec: str,
    out_csv: str,
    out_tensor_path: str,
) -> str:
    if not (os.path.exists(residue_path) and os.path.exists(pos_activation_matrix) and os.path.exists(neg_activation_matrix)):
        return ""

    residue = torch.load(residue_path, map_location="cpu", weights_only=False)
    pos = torch.load(pos_activation_matrix, map_location="cpu", weights_only=False)
    neg = torch.load(neg_activation_matrix, map_location="cpu", weights_only=False)
    if not (torch.is_tensor(residue) and torch.is_tensor(pos) and torch.is_tensor(neg)):
        return ""
    residue = residue.float()
    pos = pos.float()
    neg = neg.float()
    if pos.shape != neg.shape:
        print(f"⚠️ pos/neg activation matrix shape mismatch; skipping projection metric: {tuple(pos.shape)} vs {tuple(neg.shape)}")
        return ""

    layer_ids = _parse_layer_spec(activation_layers_spec)
    n_layers_act = pos.shape[1]
    if len(layer_ids) != n_layers_act:
        layer_ids = list(range(1, n_layers_act + 1))

    unit_dirs = []
    residue_norms = []
    for lid in layer_ids:
        idx = lid - 1
        if idx < 0 or idx >= residue.shape[0]:
            unit_dirs.append(torch.zeros(residue.shape[1], dtype=torch.float32))
            residue_norms.append(0.0)
            continue
        r = residue[idx]
        rn = float(torch.linalg.vector_norm(r).item())
        residue_norms.append(rn)
        unit_dirs.append(r / (rn + 1e-8) if rn > 0 else torch.zeros_like(r))
    u = torch.stack(unit_dirs, dim=0)  # [n_layers_act, d]

    signed_pos = (pos * u.unsqueeze(0)).sum(dim=-1)  # [n_prompts, n_layers_act]
    signed_neg = (neg * u.unsqueeze(0)).sum(dim=-1)
    signed_delta = signed_pos - signed_neg

    pos_norm = pos.norm(dim=-1)
    neg_norm = neg.norm(dim=-1)
    norm_abs_pos = signed_pos.abs() / (pos_norm + 1e-8)
    norm_abs_neg = signed_neg.abs() / (neg_norm + 1e-8)

    import pandas as pd

    df = pd.DataFrame(
        {
            "layer": layer_ids,
            "residue_norm": residue_norms,
            "mean_signed_pos": signed_pos.mean(dim=0).cpu().numpy(),
            "mean_signed_neg": signed_neg.mean(dim=0).cpu().numpy(),
            "mean_signed_delta": signed_delta.mean(dim=0).cpu().numpy(),
            "mean_abs_pos": signed_pos.abs().mean(dim=0).cpu().numpy(),
            "mean_abs_neg": signed_neg.abs().mean(dim=0).cpu().numpy(),
            "mean_abs_delta": signed_delta.abs().mean(dim=0).cpu().numpy(),
            "mean_norm_abs_pos": norm_abs_pos.mean(dim=0).cpu().numpy(),
            "mean_norm_abs_neg": norm_abs_neg.mean(dim=0).cpu().numpy(),
        }
    )
    df.to_csv(out_csv, index=False)
    torch.save(
        {
            "layer_ids": layer_ids,
            "signed_pos": signed_pos,
            "signed_neg": signed_neg,
            "signed_delta": signed_delta,
            "norm_abs_pos": norm_abs_pos,
            "norm_abs_neg": norm_abs_neg,
        },
        out_tensor_path,
    )
    print(f"📈 Saved activation-on-residue projection metric: {out_csv}")
    return out_csv


def _run_optional_post_steps(
    args,
    model_name: str,
    model_short: str,
    base_trait: str,
    ai_trait: str,
    user_trait: str,
    joint_trait: str,
    initial_plot_paths: list[str] | None = None,
):
    vec_dir = _persona_path("persona_vectors", model_short)
    extract_dir = _persona_path("eval_persona_extract", model_short)
    metrics_dir = _get_run_dir("metrics")
    plots_png_dir = _get_run_dir("plots_png")
    plots_csv_dir = _get_run_dir("plots_csv")
    combined_plot_paths = list(initial_plot_paths or [])

    pos_activation_matrix = ""
    neg_activation_matrix = ""

    if args.save_activations:
        trait_json_path = _resolve_trait_json_path(joint_trait)
        if not trait_json_path:
            print(
                "⚠️ Could not find trait JSON for activation capture. "
                f"Expected {joint_trait}.json under persona_steering/data_generation/trait_data_eval/."
            )
        else:
            activation_instruction_indices = None
            raw_indices = (args.activation_instruction_indices or "").strip().lower()
            if raw_indices and raw_indices != "all":
                activation_instruction_indices = [int(x.strip()) for x in raw_indices.split(",") if x.strip()]
            activation_outputs = save_activation_matrices(
                model_name=model_name,
                trait=joint_trait,
                trait_json_path=trait_json_path,
                extract_dir=extract_dir,
                vec_dir=vec_dir,
                activation_layers=args.activation_layers,
                activation_location=args.activation_location,
                limit_questions=args.activation_limit_questions,
                instruction_indices=activation_instruction_indices,
            )
            pos_activation_matrix = activation_outputs.get("pos_path", "")
            neg_activation_matrix = activation_outputs.get("neg_path", "")

    if not pos_activation_matrix:
        pos_candidate = os.path.join(vec_dir, f"{joint_trait}_pos_activation_matrix.pt")
        if os.path.exists(pos_candidate):
            pos_activation_matrix = pos_candidate
    if not neg_activation_matrix:
        neg_candidate = os.path.join(vec_dir, f"{joint_trait}_neg_activation_matrix.pt")
        if os.path.exists(neg_candidate):
            neg_activation_matrix = neg_candidate

    activation_projection_csv = ""
    if pos_activation_matrix and neg_activation_matrix:
        residue_path = os.path.join(vec_dir, f"residue_{joint_trait}_response_avg_diff.pt")
        activation_projection_csv = _compute_and_save_activation_projections(
            residue_path=residue_path,
            pos_activation_matrix=pos_activation_matrix,
            neg_activation_matrix=neg_activation_matrix,
            activation_layers_spec=args.activation_layers,
            out_csv=os.path.join(metrics_dir, f"{model_short}_{joint_trait}_activation_projection.csv"),
            out_tensor_path=os.path.join(metrics_dir, f"{model_short}_{joint_trait}_activation_projection.pt"),
        )

    if args.run_metrics:
        per_prompt_residue_path = args.metrics_per_prompt_residue.strip() if args.metrics_per_prompt_residue else ""
        if not per_prompt_residue_path:
            auto_ppr_path = os.path.join(metrics_dir, f"{model_short}_{joint_trait}_per_prompt_residue.pt")
            per_prompt_residue_path = _build_per_prompt_residue_from_activations(
                pos_activation_matrix,
                neg_activation_matrix,
                auto_ppr_path,
            )

        scored_csv_path = args.metrics_scored_csv.strip() if args.metrics_scored_csv else ""
        if not scored_csv_path:
            run_scored_csv = os.path.join(_get_run_dir("replies"), f"{joint_trait}_pos_scored.csv")
            if os.path.exists(run_scored_csv):
                scored_csv_path = run_scored_csv

        if not scored_csv_path:
            fallback_eval_csv = _persona_path(
                "eval_persona_eval",
                model_short,
                f"steering_{base_trait}_layer{args.layer}_coef{args.coef}.csv",
            )
            if os.path.exists(fallback_eval_csv):
                scored_csv_path = fallback_eval_csv

        eval_hidden_states_path = args.metrics_eval_hidden_states.strip() if args.metrics_eval_hidden_states else ""
        if not eval_hidden_states_path:
            run_ablation_hidden = os.path.join(_get_run_dir("ablation"), "eval_hidden_states.pt")
            legacy_ablation_hidden = os.path.join("output", "ablation", "eval_hidden_states.pt")
            if os.path.exists(run_ablation_hidden):
                eval_hidden_states_path = run_ablation_hidden
            elif os.path.exists(legacy_ablation_hidden):
                eval_hidden_states_path = legacy_ablation_hidden

        run_metrics_script(
            model_name=model_name,
            ai_trait=ai_trait,
            user_trait=user_trait,
            joint_trait=joint_trait,
            output_dir=metrics_dir,
            output_prefix=args.metrics_output_prefix,
            per_prompt_residue_path=per_prompt_residue_path,
            scored_csv_path=scored_csv_path,
            pos_activation_matrix=pos_activation_matrix,
            neg_activation_matrix=neg_activation_matrix,
            eval_hidden_states=eval_hidden_states_path,
        )

        metrics_prefix = args.metrics_output_prefix or f"{model_short}_{joint_trait}"
        os.makedirs(plots_csv_dir, exist_ok=True)
        for name in sorted(os.listdir(metrics_dir)):
            if not name.endswith(".csv"):
                continue
            src = os.path.join(metrics_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(plots_csv_dir, name))

        metrics_csv_path = os.path.join(metrics_dir, f"{metrics_prefix}_per_layer_metrics.csv")
        if os.path.exists(metrics_csv_path):
            metrics_plot_paths, _ = save_dashboard(
                metrics_csv_path=metrics_csv_path,
                model_short=model_short,
                base_trait=base_trait,
                out_dir=plots_png_dir,
                effrank_csv_path=os.path.join(metrics_dir, f"{metrics_prefix}_effrank.csv"),
                esva_csv_path=os.path.join(metrics_dir, f"{metrics_prefix}_esva.csv"),
                gbc_csv_path=os.path.join(metrics_dir, f"{metrics_prefix}_gbc.csv"),
                projection_on_residue_csv_path=os.path.join(metrics_dir, f"{metrics_prefix}_projection_on_residue.csv"),
                rho_eff_csv_path=os.path.join(metrics_dir, f"{metrics_prefix}_rho_eff.csv"),
                activation_projection_csv_path=activation_projection_csv,
                build_dashboard=False,
            )
            combined_plot_paths.extend(metrics_plot_paths)

    if args.run_metrics and combined_plot_paths:
        dashboard_path = save_plot_grid_dashboard(combined_plot_paths, model_short, base_trait, out_dir=plots_png_dir)
        if dashboard_path:
            print(f"📊 Saved combined dashboard: {dashboard_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Projection on residue tracking (for existing plot() in pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_and_save_projection_on_residue(
    residue: torch.Tensor,      # [n_layers, d]
    per_prompt_residue: torch.Tensor,  # [n_prompts, n_layers, d]
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Compute mean abs projection of per-prompt residues onto the mean residue direction.
    This is a quick approximation of rho_eff using extraction-time activations.
    Saves a JSON and a plot.
    """
    import json
    import numpy as np
 
    n_prompts, n_layers, d = per_prompt_residue.shape
    mean_abs_proj = []
    mean_norm_proj = []
 
    for l_idx in range(n_layers):
        r = residue[l_idx]
        r_norm = float(torch.linalg.vector_norm(r).item())
        if r_norm < 1e-8:
            mean_abs_proj.append(0.0)
            mean_norm_proj.append(0.0)
            continue
        r_hat = r / r_norm
        h = per_prompt_residue[:, l_idx, :]
        projs = (h @ r_hat).abs()
        h_norms = h.norm(dim=-1)
        mean_abs_proj.append(float(projs.mean().item()))
        mean_norm_proj.append(float((projs / (h_norms + 1e-8)).mean().item()))
 
    data = {
        "layer": list(range(n_layers)),
        "mean_abs_proj": mean_abs_proj,
        "mean_norm_proj": mean_norm_proj,
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = f"{out_dir}/{model_short}_{base_trait}_projection_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
 
    # Quick plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(data["layer"], data["mean_norm_proj"], marker="o", color="#8B1A1A", linewidth=2,
             label="mean |<h, r_hat>| / ||h|| (proxy for rho_rel)")
    plt.axhline(0.05, color="orange", linewidth=0.8, linestyle="--", label="0.05 (significant)")
    plt.axhline(0.01, color="red", linewidth=0.8, linestyle=":", label="0.01 (negligible below)")
    plt.xlabel("Layer")
    plt.ylabel("Normalized projection")
    plt.title("Projection of Per-Prompt Residues onto Mean Residue Direction\n"
              "(proxy for rho_rel — key diagnostic for ablation significance)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"{out_dir}/{model_short}_{base_trait}_projection_diagnostic.png"
    plt.savefig(png_path)
    plt.close()
 
    print(f"  Saved projection diagnostic: {png_path}")
    return png_path


def run_config(config_path: str):
    """Run one or more experiments from a JSON config file."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("Config has no experiments[].")

    for i, exp in enumerate(experiments, start=1):
        print(f"\n🧪 Running experiment {i}/{len(experiments)}: {exp.get('name', 'unnamed')}\n")
        run_experiment_from_dict(exp)

    print("\n✅ Config experiments completed.")

def main():
    parser = argparse.ArgumentParser(description="Full Persona Vectors Residue Pipeline via CLI Subprocesses")
    
    # Global Configs
    parser.add_argument("--model", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--instruct", type=str, default="", help="Instruct model ID (required for --mode instruct/both)")
    parser.add_argument("--base", type=str, default="", help="Base model ID (required for --mode base/both)")
    parser.add_argument("--mode", type=str, default="instruct", choices=["instruct", "base", "both"],
                        help="Select which model variant(s) to run.")
    parser.add_argument("--trait", type=str, default="humorous", help="Base trait name mapping to the datasets (e.g. humorous, impolite)")
    parser.add_argument("--ai_trait", type=str, default=None, help="Trait file for AI persona vector (default: orig_<trait>)")
    parser.add_argument("--user_trait", type=str, default=None, help="Trait file for user persona vector (default: user_<trait>)")
    parser.add_argument("--joint_trait", type=str, default=None, help="Trait file for joint persona vector (default: <trait>)")
    parser.add_argument("--layer", type=int, default=20, help="Layer to inject/extract vectors from.")
    parser.add_argument("--coef", type=float, default=2.0, help="Steering coefficient for evaluation.")
    parser.add_argument("--judge_model", type=str, default="gpt-5.4-nano", help="Model used to judge extraction/evaluation")
    parser.add_argument("--n_per_question", type=int, default=1, help="Samples per question (use 1 for tiny/cheap runs)")
    parser.add_argument("--max_tokens", type=int, default=64, help="Generation max tokens per answer")
    parser.add_argument("--max_concurrent_judges", type=int, default=8, help="Concurrent OpenAI judge requests")
    parser.add_argument("--vector_threshold", type=int, default=50, help="Trait threshold for filtering rows in vector extraction")
    parser.add_argument("--coherence_threshold", type=int, default=50, help="Coherence threshold for filtering rows in vector extraction")
    parser.add_argument("--vector_max_samples", type=int, default=0, help="Cap number of effective samples used for vector extraction (0 = all)")
    parser.add_argument("--vector_load_in_8bit", action="store_true", help="Load model in 8-bit for vector extraction to reduce VRAM")
    parser.add_argument("--extract_limit_questions", type=int, default=0,
                        help="Limit extract/eval_persona questions for tiny subset runs (0 = all).")
    parser.add_argument("--extract_overwrite", action="store_true",
                        help="Force rerun of extract CSV/vector generation even if files already exist.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation in e2e and proceed directly to plotting")
    parser.add_argument("--clean_output", action="store_true", help="Delete output folder before run")
    parser.add_argument("--keep_output", action="store_true", help="Do not auto-clean output folder before run")
    parser.add_argument("--run_name", type=str, default="", help="Optional run folder name under output/. If omitted, an auto name is used.")
    parser.add_argument("--config", type=str, default=None, help="JSON config file path for multi-experiment runs")
    parser.add_argument("--dataset_model", type=str, default="gpt-5.4-nano", help="OpenAI model used for dataset generation")
    parser.add_argument("--source_ai_dataset", type=str, default="orig_impolite.json", help="Source AI dataset to flip for AI-positive dataset")
    parser.add_argument("--source_eval_dataset", type=str, default="impolite.json", help="Source eval dataset to borrow questions/eval prompt")
    parser.add_argument("--ai_output_name", type=str, default="orig_polite", help="Output name for generated AI dataset (without .json)")
    parser.add_argument("--user_output_name", type=str, default="user_impolite_fewshot", help="Output name for generated user dataset (without .json)")
    parser.add_argument("--joint_output_name", type=str, default="polite_ai_user_impolite_fewshot", help="Output name for generated joint dataset (without .json)")
    parser.add_argument("--ai_persona_positive", type=str, default="polite", help="Positive AI persona for generated datasets")
    parser.add_argument("--ai_persona_negative", type=str, default="impolite", help="Negative AI persona for generated datasets")
    parser.add_argument("--user_persona_positive", type=str, default="impolite", help="Positive user persona for generated datasets")
    parser.add_argument("--user_persona_negative", type=str, default="polite", help="Negative user persona for generated datasets")
    parser.add_argument("--user_instruction_mode", type=str, default="fewshot", choices=["simple", "fewshot"], help="Instruction mode for generated user dataset")
    parser.add_argument("--joint_instruction_mode", type=str, default="fewshot", choices=["simple", "fewshot"], help="Instruction mode for generated joint dataset")
    parser.add_argument("--fewshot_turns", type=int, default=6, help="Minimum few-shot dialogue turns per instruction")
    parser.add_argument("--fewshot_min_words", type=int, default=140, help="Minimum words per instruction in few-shot mode")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save per-prompt activation matrices to disk after extract. "
                             "Required for ESVA, EffRank, rho_eff, and GBC metrics.")
    parser.add_argument("--activation_layers", type=str, default="15-28",
                        help="Layer spec for activation saving (e.g. '15-28' or '20,24,28').")
    parser.add_argument("--activation_location", type=str, default="post_resid",
                        choices=["post_resid", "pre_resid", "post_attn", "post_mlp"],
                        help="Where in the transformer block to capture activations. "
                             "post_resid = after residual addition (default, same as CAA). "
                             "pre_resid = before residual addition. "
                             "post_attn = after attention block. "
                             "post_mlp = after MLP block.")
    parser.add_argument("--activation_limit_questions", type=int, default=0,
                        help="Limit number of questions used for activation capture (0 = all).")
    parser.add_argument("--activation_instruction_indices", type=str, default="all",
                        help="Comma-separated instruction indices for activation capture, or 'all'.")
    parser.add_argument("--run_metrics", action="store_true",
                        help="Run metrics.py after extract/plot to compute all interaction metrics.")
    parser.add_argument("--metrics_output_prefix", type=str, default="",
                        help="Prefix for metrics output files (default: model_short_joint_trait).")
    parser.add_argument("--metrics_per_prompt_residue", type=str, default="",
                        help="Optional path to per-prompt residue tensor (.pt) for EffRank/GBC/projection metrics.")
    parser.add_argument("--metrics_scored_csv", type=str, default="",
                        help="Optional scored CSV path with harmfulness column for GBC.")
    parser.add_argument("--metrics_eval_hidden_states", type=str, default="",
                        help="Optional eval hidden states path (.pt) for rho_eff/rho_rel.")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    subparsers.add_parser("extract", help="Gen vectors (A, U, J) and Math Residue via subprocesses")
    subparsers.add_parser("evaluate", help="Eval Third Persona via eval.eval_persona")
    subparsers.add_parser("plot", help="Plot Projections and Benchmarks matching Notebook")
    subparsers.add_parser("e2e", help="End to end run (Extract -> Evaluate -> Plot)")
    compare_ai_parser = subparsers.add_parser("compare-ai", help="AI-only humorous-vs-evil overlay comparison")
    compare_ai_parser.add_argument("--left_trait", type=str, default="humorous", help="Left AI trait for overlay")
    compare_ai_parser.add_argument("--right_trait", type=str, default="evil", help="Right AI trait for overlay")
    compare_ai_parser.add_argument("--compare_limit_questions", type=int, default=8,
                                   help="Limit questions for compare-ai extraction (0 = all).")
    compare_ai_parser.add_argument("--compare_overwrite", action="store_true",
                                   help="Force overwrite in compare-ai extraction and activation capture.")
    run_cfg_parser = subparsers.add_parser("run-config", help="Run experiments from a JSON config file")
    run_cfg_parser.add_argument("--config_file", type=str, required=True, help="Path to pipeline experiment config JSON")
    subparsers.add_parser("generate-datasets", help="Generate AI/User/Joint persona datasets via OpenAI")

    args = parser.parse_args()
    if args.clean_output:
        clean_output_dir()

    if args.command == "run-config":
        config_path = getattr(args, "config_file", None) or args.config
        if not config_path:
            raise ValueError("Please pass --config_file path/to/config.json")
        run_config(config_path)
        return

    if args.command == "generate-datasets":
        generate_persona_combo_datasets(
            model=args.dataset_model,
            source_ai_dataset=args.source_ai_dataset,
            source_eval_dataset=args.source_eval_dataset,
            ai_output_name=args.ai_output_name,
            user_output_name=args.user_output_name,
            joint_output_name=args.joint_output_name,
            ai_persona_positive=args.ai_persona_positive,
            ai_persona_negative=args.ai_persona_negative,
            user_persona_positive=args.user_persona_positive,
            user_persona_negative=args.user_persona_negative,
            user_instruction_mode=args.user_instruction_mode,
            joint_instruction_mode=args.joint_instruction_mode,
            fewshot_turns=args.fewshot_turns,
            fewshot_min_words=args.fewshot_min_words,
        )
        return

    if args.command is None:
        parser.print_help()
        return

    if args.model.strip():
        raise ValueError("--model is deprecated. Use --instruct/--base with --mode instruct|base|both.")

    model_variants = _resolve_mode_models(args.mode, args.instruct, args.base)
    ai_trait = args.ai_trait or f"orig_{args.trait}"
    user_trait = args.user_trait or f"user_{args.trait}"
    joint_trait = args.joint_trait or args.trait

    run_trait = f"{args.left_trait}_vs_{args.right_trait}_ai" if args.command == "compare-ai" else joint_trait
    run_name = args.run_name.strip() if args.run_name else ""
    if not run_name:
        run_name = _build_run_name(args.command, model_variants[0][1], run_trait)

    argv = " ".join(shlex.quote(a) for a in sys.argv[1:])

    for variant_label, model_name in model_variants:
        variant_run_name = _variant_run_name(run_name, variant_label, len(model_variants))
        run_dirs = _init_run_dirs(variant_run_name)
        _set_active_run_dirs(run_dirs)
        print(f"📁 Run output folder: {run_dirs['root']}")
        check_dirs()
        _append_command_log(f"python pipeline.py {argv}", stage="pipeline.py")

        model_short = model_name.split("/")[-1]

        if args.command == "extract":
            extract(
                model_name,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                args.judge_model,
                args.n_per_question,
                args.max_tokens,
                args.max_concurrent_judges,
                args.vector_threshold,
                args.coherence_threshold,
                args.vector_max_samples,
                args.vector_load_in_8bit,
                args.extract_limit_questions,
                args.extract_overwrite,
            )
            extract_dir = _persona_path("eval_persona_extract", model_short)
            document_extract_replies(model_short, [ai_trait, user_trait, joint_trait], extract_dir)
            _run_optional_post_steps(args, model_name, model_short, args.trait, ai_trait, user_trait, joint_trait)
        elif args.command == "evaluate":
            evaluate(
                model_name,
                args.trait,
                args.layer,
                args.coef,
                args.judge_model,
                args.n_per_question,
                args.max_tokens,
                args.max_concurrent_judges,
            )
        elif args.command == "plot":
            plot_paths = plot(
                model_name,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                plots_out_dir=_get_run_dir("plots_png"),
                build_dashboard=not args.run_metrics,
            )
            _run_optional_post_steps(
                args,
                model_name,
                model_short,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                initial_plot_paths=plot_paths,
            )
        elif args.command == "e2e":
            extract(
                model_name,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                args.judge_model,
                args.n_per_question,
                args.max_tokens,
                args.max_concurrent_judges,
                args.vector_threshold,
                args.coherence_threshold,
                args.vector_max_samples,
                args.vector_load_in_8bit,
                args.extract_limit_questions,
                args.extract_overwrite,
            )
            extract_dir = _persona_path("eval_persona_extract", model_short)
            document_extract_replies(model_short, [ai_trait, user_trait, joint_trait], extract_dir)
            if args.skip_eval:
                print("⏭️ Skipping evaluate step (--skip_eval enabled).")
            else:
                evaluate(
                    model_name,
                    args.trait,
                    args.layer,
                    args.coef,
                    args.judge_model,
                    args.n_per_question,
                    args.max_tokens,
                    args.max_concurrent_judges,
                )
            plot_paths = plot(
                model_name,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                plots_out_dir=_get_run_dir("plots_png"),
                build_dashboard=not args.run_metrics,
            )
            _run_optional_post_steps(
                args,
                model_name,
                model_short,
                args.trait,
                ai_trait,
                user_trait,
                joint_trait,
                initial_plot_paths=plot_paths,
            )
        elif args.command == "compare-ai":
            run_compare_ai_overlay(
                model_name=model_name,
                left_trait=args.left_trait,
                right_trait=args.right_trait,
                judge_model=args.judge_model,
                n_per_question=args.n_per_question,
                max_tokens=args.max_tokens,
                max_concurrent_judges=args.max_concurrent_judges,
                limit_questions=args.compare_limit_questions,
                activation_layers=args.activation_layers,
                activation_location=args.activation_location,
                activation_instruction_indices=args.activation_instruction_indices,
                run_name=variant_run_name,
                overwrite=args.compare_overwrite,
            )
        else:
            parser.print_help()
            return

    if args.command == "e2e":
        print("\n✅ End-To-End Delegated Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
