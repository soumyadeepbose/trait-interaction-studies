import argparse
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
PERSONA_ROOT = PROJECT_ROOT / "persona_steering"


def resolve_input_path(path_str: str) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(PROJECT_ROOT / raw)
        if not str(raw).startswith("persona_steering/"):
            candidates.append(PERSONA_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layer_spec(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid range '{token}': end < start")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("Layer spec produced empty list")
    return sorted(set(out))


def load_hf_model_and_tokenizer(model_name: str, dtype: str, device: str, load_in_8bit: bool):
    if load_in_8bit and dtype != "float16":
        raise ValueError("--load_in_8bit is only supported when --dtype=float16")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    model_kwargs = {"trust_remote_code": True}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not load_in_8bit:
        model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def _normalize_question_id(value: object) -> str:
    key = str(value).strip()
    if not key:
        return key
    # Normalize common pos/neg markers so paired rows share the same id.
    key = key.replace("_pos_", "_").replace("_neg_", "_")
    key = key.replace("-pos-", "-").replace("-neg-", "-")
    return key


def _build_question_occurrence_key(df: pd.DataFrame, question_col: str) -> pd.Series:
    q = df[question_col].astype(str).str.strip()
    occ = q.groupby(q).cumcount().astype(str)
    return q + "||" + occ


def _build_join_key(df: pd.DataFrame, question_col: str) -> pd.Series:
    if "question_id" in df.columns:
        return df["question_id"].map(_normalize_question_id).astype(str)
    return _build_question_occurrence_key(df, question_col)


def _align_by_keys(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    pos_keys: pd.Series,
    neg_keys: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos_tmp = pos_df.copy()
    neg_tmp = neg_df.copy()

    pos_tmp["_join_key"] = pos_keys.astype(str)
    neg_tmp["_join_key"] = neg_keys.astype(str)
    pos_tmp["_join_occ"] = pos_tmp.groupby("_join_key").cumcount()
    neg_tmp["_join_occ"] = neg_tmp.groupby("_join_key").cumcount()
    pos_tmp["_row_order"] = np.arange(len(pos_tmp))

    merged = pos_tmp.merge(
        neg_tmp,
        on=["_join_key", "_join_occ"],
        how="inner",
        suffixes=("_pos", "_neg"),
        sort=False,
    )

    if merged.empty:
        return pos_df.iloc[0:0].copy(), neg_df.iloc[0:0].copy()

    order_col = "_row_order_pos" if "_row_order_pos" in merged.columns else "_row_order"
    merged = merged.sort_values(order_col).reset_index(drop=True)

    pos_cols = [f"{col}_pos" for col in pos_df.columns]
    neg_cols = [f"{col}_neg" for col in neg_df.columns]

    aligned_pos = merged[pos_cols].copy()
    aligned_neg = merged[neg_cols].copy()
    aligned_pos.columns = pos_df.columns
    aligned_neg.columns = neg_df.columns
    return aligned_pos.reset_index(drop=True), aligned_neg.reset_index(drop=True)


def align_pos_neg_rows(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    question_col: str,
    prompt_col: str,
    answer_col: str,
    max_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for col in [question_col, prompt_col, answer_col]:
        if col not in pos_df.columns:
            raise KeyError(f"Missing column '{col}' in positive CSV")
        if col not in neg_df.columns:
            raise KeyError(f"Missing column '{col}' in negative CSV")

    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.iloc[:n].reset_index(drop=True)
    neg_df = neg_df.iloc[:n].reset_index(drop=True)

    pos_keys = _build_join_key(pos_df, question_col)
    neg_keys = _build_join_key(neg_df, question_col)
    aligned_pos, aligned_neg = _align_by_keys(pos_df, neg_df, pos_keys, neg_keys)

    # Fallback alignment if ids exist but pairing quality is poor.
    if len(aligned_pos) < 2 and "question_id" in pos_df.columns and "question_id" in neg_df.columns:
        fallback_pos_keys = _build_question_occurrence_key(pos_df, question_col)
        fallback_neg_keys = _build_question_occurrence_key(neg_df, question_col)
        fallback_pos, fallback_neg = _align_by_keys(pos_df, neg_df, fallback_pos_keys, fallback_neg_keys)
        if len(fallback_pos) > len(aligned_pos):
            aligned_pos, aligned_neg = fallback_pos, fallback_neg

    mismatch_count = int(n - len(aligned_pos))

    pos_df = aligned_pos
    neg_df = aligned_neg

    if max_samples > 0:
        pos_df = pos_df.iloc[:max_samples].reset_index(drop=True)
        neg_df = neg_df.iloc[:max_samples].reset_index(drop=True)

    info = {
        "initial_paired_rows": n,
        "mismatch_dropped_rows": mismatch_count,
        "final_aligned_rows": len(pos_df),
    }
    return pos_df, neg_df, info


def get_response_avg_by_layer(
    model,
    tokenizer,
    prompt_text: str,
    answer_text: str,
    layer_ids: list[int],
) -> torch.Tensor:
    text = f"{prompt_text}{answer_text}"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_vectors = []
    for layer in layer_ids:
        hs = outputs.hidden_states[layer]  # [1, seq, d]
        if prompt_len >= hs.shape[1]:
            response_tokens = hs[:, -1:, :]
        else:
            response_tokens = hs[:, prompt_len:, :]
        layer_vectors.append(response_tokens.mean(dim=1).squeeze(0).detach().cpu().to(torch.float32))

    del outputs
    return torch.stack(layer_vectors, dim=0)  # [L, d]


def pca_svd(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    n, d = matrix.shape
    if n < 2:
        raise ValueError("Need at least 2 samples for PCA")

    k = min(n_components, n, d)
    if k < 1:
        raise ValueError("No valid components for PCA")

    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean

    _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k, :]
    coords = centered @ components.T

    denom = max(n - 1, 1)
    explained = (singular_vals**2) / denom
    explained_ratio = explained / (explained.sum() + 1e-12)

    return coords.astype(np.float32), components.astype(np.float32), explained_ratio.astype(np.float32), mean.astype(np.float32)


def validate_layer_ids(layer_ids: list[int], model) -> None:
    model_layer_rows = int(model.config.num_hidden_layers) + 1
    for layer in layer_ids:
        if layer < 0 or layer >= model_layer_rows:
            raise ValueError(
                f"Layer {layer} is invalid for model with {model_layer_rows} hidden-state rows. "
                f"Valid range: [0, {model_layer_rows - 1}]"
            )


def _project_with_pca(matrix: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = matrix - mean
    projected = centered @ components.T
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])), mode="constant")
    return projected[:, :3].astype(np.float32)


def _parse_prompt_instruction_key(question_id: object) -> tuple[str, int, int]:
    qid = str(question_id).strip()
    match = re.search(r"_(\d+)_(?:pos|neg)_(\d+)$", qid)
    if match is None:
        raise ValueError(
            f"Failed to parse question_id='{qid}'. Expected suffix pattern like '<...>_<qidx>_pos_<instidx>' or '<...>_<qidx>_neg_<instidx>'."
        )
    question_idx = int(match.group(1))
    instruction_idx = int(match.group(2))
    key = f"{question_idx}_{instruction_idx}"
    return key, question_idx, instruction_idx


def _build_family_row_index(
    df: pd.DataFrame,
    question_col: str,
) -> tuple[dict[str, list[int]], dict[str, dict]]:
    if "question_id" not in df.columns:
        raise KeyError("Missing required column 'question_id' for four-group animation mode")
    if question_col not in df.columns:
        raise KeyError(f"Missing required column '{question_col}' for four-group animation mode")

    key_to_rows: dict[str, list[int]] = {}
    key_meta: dict[str, dict] = {}

    for row_idx in range(len(df)):
        key, q_idx, inst_idx = _parse_prompt_instruction_key(df.iloc[row_idx]["question_id"])
        key_to_rows.setdefault(key, []).append(row_idx)
        if key not in key_meta:
            key_meta[key] = {
                "key": key,
                "question_idx": q_idx,
                "instruction_idx": inst_idx,
                "question": str(df.iloc[row_idx][question_col]),
            }

    return key_to_rows, key_meta


def _compute_family_steering_tensor(
    family_name: str,
    model,
    tokenizer,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    pos_key_to_rows: dict[str, list[int]],
    neg_key_to_rows: dict[str, list[int]],
    ordered_keys: list[str],
    prompt_col: str,
    answer_col: str,
    layer_ids: list[int],
    repeat_agg: str,
) -> torch.Tensor:
    family_vectors: list[torch.Tensor] = []

    for key in tqdm(ordered_keys, desc=f"Collecting {family_name} vectors"):
        pos_rows = pos_key_to_rows[key]
        neg_rows = neg_key_to_rows[key]

        if repeat_agg == "first":
            pos_rows = pos_rows[:1]
            neg_rows = neg_rows[:1]

        pos_layer_vecs = []
        for row_idx in pos_rows:
            pos_layer_vecs.append(
                get_response_avg_by_layer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=str(pos_df.iloc[row_idx][prompt_col]),
                    answer_text=str(pos_df.iloc[row_idx][answer_col]),
                    layer_ids=layer_ids,
                )
            )

        neg_layer_vecs = []
        for row_idx in neg_rows:
            neg_layer_vecs.append(
                get_response_avg_by_layer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=str(neg_df.iloc[row_idx][prompt_col]),
                    answer_text=str(neg_df.iloc[row_idx][answer_col]),
                    layer_ids=layer_ids,
                )
            )

        pos_mean = torch.stack(pos_layer_vecs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_layer_vecs, dim=0).mean(dim=0)
        family_vectors.append(pos_mean - neg_mean)

    return torch.stack(family_vectors, dim=0)


def _project_group_tensors(
    group_tensors: dict[str, torch.Tensor],
    pca_fit_mode: str,
) -> tuple[dict[str, np.ndarray], dict]:
    group_order = list(group_tensors.keys())
    first_shape = None
    for tensor in group_tensors.values():
        if first_shape is None:
            first_shape = tensor.shape
        elif tensor.shape != first_shape:
            raise ValueError("All group tensors must have the same shape [N, L, D]")

    if first_shape is None:
        raise ValueError("group_tensors is empty")

    n_samples, n_layers, hidden_dim = first_shape

    if pca_fit_mode == "global":
        flat_by_group = {
            group: tensor.reshape(n_samples * n_layers, hidden_dim).cpu().numpy()
            for group, tensor in group_tensors.items()
        }
        fit_matrix = np.concatenate([flat_by_group[group] for group in group_order], axis=0)
        _, components, explained_ratio, mean = pca_svd(fit_matrix, n_components=3)

        coords_by_group = {}
        for group in group_order:
            projected = _project_with_pca(flat_by_group[group], components, mean)
            coords_by_group[group] = projected.reshape(n_samples, n_layers, 3)

        pca_info = {
            "fit_mode": "global",
            "explained_ratio": explained_ratio.tolist(),
            "components_shape": list(components.shape),
        }
        return coords_by_group, pca_info

    if pca_fit_mode == "per_layer":
        coords_by_group = {
            group: np.zeros((n_samples, n_layers, 3), dtype=np.float32)
            for group in group_order
        }
        per_layer_explained_ratio = []

        for layer_local_idx in range(n_layers):
            layer_mats = {
                group: group_tensors[group][:, layer_local_idx, :].cpu().numpy()
                for group in group_order
            }
            fit_matrix = np.concatenate([layer_mats[group] for group in group_order], axis=0)
            _, components, explained_ratio, mean = pca_svd(fit_matrix, n_components=3)
            per_layer_explained_ratio.append(explained_ratio.tolist())

            for group in group_order:
                coords_by_group[group][:, layer_local_idx, :] = _project_with_pca(
                    layer_mats[group], components, mean
                )

        pca_info = {
            "fit_mode": "per_layer",
            "per_layer_explained_ratio": per_layer_explained_ratio,
        }
        return coords_by_group, pca_info

    raise ValueError(f"Invalid --pca_fit_mode: {pca_fit_mode}")


def _build_four_group_coords_df(
    coords_by_group: dict[str, np.ndarray],
    ordered_keys: list[str],
    key_meta: dict[str, dict],
    layer_ids: list[int],
) -> pd.DataFrame:
    group_order = ["ai", "user", "joint", "residue"]
    records = []

    for sample_idx, key in enumerate(ordered_keys):
        meta = key_meta[key]
        for layer_local_idx, layer in enumerate(layer_ids):
            for group in group_order:
                coords = coords_by_group[group][sample_idx, layer_local_idx, :]
                records.append(
                    {
                        "sample_idx": int(sample_idx),
                        "key": key,
                        "question_idx": int(meta["question_idx"]),
                        "instruction_idx": int(meta["instruction_idx"]),
                        "question": str(meta["question"]),
                        "group": group,
                        "layer": int(layer),
                        "pc1": float(coords[0]),
                        "pc2": float(coords[1]),
                        "pc3": float(coords[2]),
                    }
                )

    return pd.DataFrame(records)


def write_plotly_four_group_animation(
    coords_df: pd.DataFrame,
    html_path: Path,
) -> None:
    group_order = ["ai", "user", "joint", "residue"]

    fig = px.scatter_3d(
        coords_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="group",
        symbol="group",
        animation_frame="layer",
        category_orders={"group": group_order},
        hover_data=["sample_idx", "question_idx", "instruction_idx", "key"],
        opacity=0.82,
        title="Layer-wise 3D PCA: AI / User / Joint / Residue",
    )

    x_vals = coords_df["pc1"].to_numpy(dtype=np.float32)
    y_vals = coords_df["pc2"].to_numpy(dtype=np.float32)
    z_vals = coords_df["pc3"].to_numpy(dtype=np.float32)
    x_margin = max(1e-6, 0.05 * (float(x_vals.max()) - float(x_vals.min())))
    y_margin = max(1e-6, 0.05 * (float(y_vals.max()) - float(y_vals.min())))
    z_margin = max(1e-6, 0.05 * (float(z_vals.max()) - float(z_vals.min())))

    fig.update_layout(
        scene={
            "xaxis": {"title": "PC1", "range": [float(x_vals.min()) - x_margin, float(x_vals.max()) + x_margin]},
            "yaxis": {"title": "PC2", "range": [float(y_vals.min()) - y_margin, float(y_vals.max()) + y_margin]},
            "zaxis": {"title": "PC3", "range": [float(z_vals.min()) - z_margin, float(z_vals.max()) + z_margin]},
        },
        legend_title_text="Group",
    )
    fig.write_html(str(html_path))


def write_four_group_layerwise_gif(
    coords_df: pd.DataFrame,
    gif_path: Path,
    fps: int,
    dpi: int,
) -> dict:
    layers = sorted(int(x) for x in coords_df["layer"].dropna().unique().tolist())
    if not layers:
        raise ValueError("No layers found for four-group GIF animation")

    group_order = ["ai", "user", "joint", "residue"]
    colors = {
        "ai": "#1f77b4",
        "user": "#ff7f0e",
        "joint": "#2ca02c",
        "residue": "#d62728",
    }
    markers = {
        "ai": "o",
        "user": "^",
        "joint": "s",
        "residue": "D",
    }

    x_vals = coords_df["pc1"].to_numpy(dtype=np.float32)
    y_vals = coords_df["pc2"].to_numpy(dtype=np.float32)
    z_vals = coords_df["pc3"].to_numpy(dtype=np.float32)

    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    z_min, z_max = float(z_vals.min()), float(z_vals.max())

    x_margin = max(1e-6, 0.05 * (x_max - x_min))
    y_margin = max(1e-6, 0.05 * (y_max - y_min))
    z_margin = max(1e-6, 0.05 * (z_max - z_min))

    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    def setup_axes() -> None:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.view_init(elev=22, azim=40)

    def update(frame_idx: int):
        ax.cla()
        setup_axes()
        layer = layers[frame_idx]
        subset_layer = coords_df[coords_df["layer"] == layer]

        for group in group_order:
            group_df = subset_layer[subset_layer["group"] == group]
            if group_df.empty:
                continue
            ax.scatter(
                group_df["pc1"].to_numpy(dtype=np.float32),
                group_df["pc2"].to_numpy(dtype=np.float32),
                group_df["pc3"].to_numpy(dtype=np.float32),
                c=colors[group],
                marker=markers[group],
                s=20,
                alpha=0.82,
                label=group,
            )

        ax.set_title(f"Layer-wise 3D PCA (Layer {layer})")
        ax.legend(loc="upper left")
        return []

    setup_axes()
    interval_ms = int(1000 / max(fps, 1))
    anim = FuncAnimation(fig, update, frames=len(layers), interval=interval_ms, blit=False)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    return {
        "gif": str(gif_path),
        "fps": int(fps),
        "dpi": int(dpi),
    }


def write_residue_layer_scree_plots(
    residue_tensor: torch.Tensor,
    layer_ids: list[int],
    out_dir: Path,
    max_components: int,
) -> dict:
    scree_dir = out_dir / "residue_scree_plots"
    scree_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    layer_entries = []

    for local_idx, layer in enumerate(layer_ids):
        x_layer = residue_tensor[:, local_idx, :].cpu().numpy()
        n_comp = max(1, min(int(max_components), x_layer.shape[0], x_layer.shape[1]))
        _, _, ratio, _ = pca_svd(x_layer, n_components=n_comp)

        components = np.arange(1, len(ratio) + 1, dtype=np.int32)
        cumulative = np.cumsum(ratio)

        df = pd.DataFrame(
            {
                "layer": int(layer),
                "component": components,
                "explained_ratio": ratio,
                "cumulative_explained_ratio": cumulative,
            }
        )
        csv_path = scree_dir / f"residue_scree_layer{int(layer)}.csv"
        df.to_csv(csv_path, index=False)

        fig_mpl, ax = plt.subplots(figsize=(8, 5), dpi=140)
        ax.plot(components, ratio, marker="o", linewidth=1.8, markersize=3.5, color="#1f77b4", label="Explained Ratio")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(components, cumulative, linestyle="--", linewidth=1.8, color="#ff7f0e", label="Cumulative Ratio")
        ax2.set_ylabel("Cumulative Explained Ratio")
        ax2.set_ylim(0.0, 1.02)

        ax.set_title(f"Residual Scree Plot (Layer {int(layer)})")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        png_path = scree_dir / f"residue_scree_layer{int(layer)}.png"
        fig_mpl.tight_layout()
        fig_mpl.savefig(png_path)
        plt.close(fig_mpl)

        fig_plotly = go.Figure()
        fig_plotly.add_trace(
            go.Scatter(
                x=components,
                y=ratio,
                mode="lines+markers",
                name="Explained Ratio",
            )
        )
        fig_plotly.add_trace(
            go.Scatter(
                x=components,
                y=cumulative,
                mode="lines+markers",
                name="Cumulative Ratio",
                yaxis="y2",
            )
        )
        fig_plotly.update_layout(
            title=f"Residual Scree Plot (Layer {int(layer)})",
            xaxis_title="Principal Component",
            yaxis={"title": "Explained Variance Ratio"},
            yaxis2={
                "title": "Cumulative Explained Ratio",
                "overlaying": "y",
                "side": "right",
                "range": [0.0, 1.02],
            },
            legend={"x": 0.02, "y": 0.98},
        )
        html_path = scree_dir / f"residue_scree_layer{int(layer)}.html"
        fig_plotly.write_html(str(html_path))

        rows.extend(df.to_dict(orient="records"))
        layer_entries.append(
            {
                "layer": int(layer),
                "csv": str(csv_path),
                "png": str(png_path),
                "html": str(html_path),
                "pc1_explained_ratio": float(ratio[0]) if len(ratio) > 0 else 0.0,
                "pc1_to_pc3_sum": float(ratio[:3].sum()),
            }
        )

    long_csv_path = scree_dir / "residue_scree_all_layers_long.csv"
    pd.DataFrame(rows).to_csv(long_csv_path, index=False)

    manifest = {
        "layers": layer_ids,
        "max_components": int(max_components),
        "entries": layer_entries,
        "long_csv": str(long_csv_path),
    }
    manifest_path = scree_dir / "residue_scree_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "scree_dir": str(scree_dir),
        "manifest": str(manifest_path),
        "long_csv": str(long_csv_path),
        "n_layers": len(layer_entries),
    }


def run_four_group_layer_animation(
    args,
    model,
    tokenizer,
    layer_ids: list[int],
    out_dir: Path,
) -> None:
    required_paths = {
        "ai_pos_csv": args.ai_pos_csv,
        "ai_neg_csv": args.ai_neg_csv,
        "user_pos_csv": args.user_pos_csv,
        "user_neg_csv": args.user_neg_csv,
    }
    missing_args = [name for name, value in required_paths.items() if not value]
    if missing_args:
        raise ValueError(
            "--make_four_group_animation requires these arguments: " + ", ".join(f"--{x}" for x in missing_args)
        )

    family_csvs = {
        "ai": (Path(args.ai_pos_csv), Path(args.ai_neg_csv)),
        "user": (Path(args.user_pos_csv), Path(args.user_neg_csv)),
        "joint": (Path(args.pos_csv), Path(args.neg_csv)),
    }

    family_data: dict[str, dict] = {}
    family_key_sets = []

    for family, (pos_path, neg_path) in family_csvs.items():
        if not pos_path.exists():
            raise FileNotFoundError(f"{family} positive CSV not found: {pos_path}")
        if not neg_path.exists():
            raise FileNotFoundError(f"{family} negative CSV not found: {neg_path}")

        pos_df = pd.read_csv(pos_path)
        neg_df = pd.read_csv(neg_path)

        for col in [args.question_col, args.prompt_col, args.answer_col, "question_id"]:
            if col not in pos_df.columns:
                raise KeyError(f"Missing column '{col}' in {pos_path}")
            if col not in neg_df.columns:
                raise KeyError(f"Missing column '{col}' in {neg_path}")

        pos_key_to_rows, pos_meta = _build_family_row_index(pos_df, question_col=args.question_col)
        neg_key_to_rows, _ = _build_family_row_index(neg_df, question_col=args.question_col)

        paired_keys = set(pos_key_to_rows.keys()) & set(neg_key_to_rows.keys())
        if not paired_keys:
            raise ValueError(f"No paired prompt/instruction keys for family '{family}'")

        family_data[family] = {
            "pos_df": pos_df,
            "neg_df": neg_df,
            "pos_key_to_rows": pos_key_to_rows,
            "neg_key_to_rows": neg_key_to_rows,
            "meta": pos_meta,
            "paired_keys": paired_keys,
            "pos_path": str(pos_path),
            "neg_path": str(neg_path),
        }
        family_key_sets.append(paired_keys)

    common_keys = set.intersection(*family_key_sets)
    if len(common_keys) < 2:
        raise ValueError(
            f"Need at least 2 common prompt/instruction keys across ai/user/joint families, got {len(common_keys)}"
        )

    key_meta = {
        key: family_data["joint"]["meta"].get(
            key,
            family_data["ai"]["meta"].get(key, family_data["user"]["meta"][key]),
        )
        for key in common_keys
    }

    ordered_keys = sorted(
        common_keys,
        key=lambda key: (
            int(key_meta[key]["question_idx"]),
            int(key_meta[key]["instruction_idx"]),
            key,
        ),
    )

    if args.max_samples > 0:
        ordered_keys = ordered_keys[: args.max_samples]

    ai_tensor = _compute_family_steering_tensor(
        family_name="AI",
        model=model,
        tokenizer=tokenizer,
        pos_df=family_data["ai"]["pos_df"],
        neg_df=family_data["ai"]["neg_df"],
        pos_key_to_rows=family_data["ai"]["pos_key_to_rows"],
        neg_key_to_rows=family_data["ai"]["neg_key_to_rows"],
        ordered_keys=ordered_keys,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
        layer_ids=layer_ids,
        repeat_agg=args.repeat_agg,
    )

    user_tensor = _compute_family_steering_tensor(
        family_name="User",
        model=model,
        tokenizer=tokenizer,
        pos_df=family_data["user"]["pos_df"],
        neg_df=family_data["user"]["neg_df"],
        pos_key_to_rows=family_data["user"]["pos_key_to_rows"],
        neg_key_to_rows=family_data["user"]["neg_key_to_rows"],
        ordered_keys=ordered_keys,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
        layer_ids=layer_ids,
        repeat_agg=args.repeat_agg,
    )

    joint_tensor = _compute_family_steering_tensor(
        family_name="Joint",
        model=model,
        tokenizer=tokenizer,
        pos_df=family_data["joint"]["pos_df"],
        neg_df=family_data["joint"]["neg_df"],
        pos_key_to_rows=family_data["joint"]["pos_key_to_rows"],
        neg_key_to_rows=family_data["joint"]["neg_key_to_rows"],
        ordered_keys=ordered_keys,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
        layer_ids=layer_ids,
        repeat_agg=args.repeat_agg,
    )

    residue_tensor = joint_tensor - ai_tensor - user_tensor

    group_tensors = {
        "ai": ai_tensor,
        "user": user_tensor,
        "joint": joint_tensor,
        "residue": residue_tensor,
    }

    torch.save(ai_tensor, out_dir / "ai_prompt_instruction_steering.pt")
    torch.save(user_tensor, out_dir / "user_prompt_instruction_steering.pt")
    torch.save(joint_tensor, out_dir / "joint_prompt_instruction_steering.pt")
    torch.save(residue_tensor, out_dir / "residue_prompt_instruction_steering.pt")

    residue_scree_info = write_residue_layer_scree_plots(
        residue_tensor=residue_tensor,
        layer_ids=layer_ids,
        out_dir=out_dir,
        max_components=args.residue_scree_max_components,
    )

    coords_by_group, pca_info = _project_group_tensors(
        group_tensors=group_tensors,
        pca_fit_mode=args.pca_fit_mode,
    )
    coords_df = _build_four_group_coords_df(
        coords_by_group=coords_by_group,
        ordered_keys=ordered_keys,
        key_meta=key_meta,
        layer_ids=layer_ids,
    )
    coords_csv_path = out_dir / "four_group_layerwise_pca_coords.csv"
    coords_df.to_csv(coords_csv_path, index=False)

    plotly_html_path = out_dir / "four_group_layerwise_pca_animated_3d.html"
    write_plotly_four_group_animation(coords_df, plotly_html_path)

    gif_info = write_four_group_layerwise_gif(
        coords_df=coords_df,
        gif_path=out_dir / "four_group_layerwise_pca_animation.gif",
        fps=args.animation_fps,
        dpi=args.animation_dpi,
    )

    question_indices = sorted({int(key_meta[key]["question_idx"]) for key in ordered_keys})
    instruction_indices = sorted({int(key_meta[key]["instruction_idx"]) for key in ordered_keys})
    expected_count = len(question_indices) * len(instruction_indices)

    family_stats = {}
    for family in ["ai", "user", "joint"]:
        pos_counts = [
            len(family_data[family]["pos_key_to_rows"][key])
            for key in ordered_keys
        ]
        neg_counts = [
            len(family_data[family]["neg_key_to_rows"][key])
            for key in ordered_keys
        ]
        family_stats[family] = {
            "pos_csv": family_data[family]["pos_path"],
            "neg_csv": family_data[family]["neg_path"],
            "avg_pos_rows_per_key": float(np.mean(pos_counts)),
            "avg_neg_rows_per_key": float(np.mean(neg_counts)),
        }

    summary = {
        "mode": "four_group_layerwise_animation",
        "model_name": args.model_name,
        "layers": layer_ids,
        "pca_fit_mode": args.pca_fit_mode,
        "repeat_agg": args.repeat_agg,
        "matrix_rows_per_group": len(ordered_keys),
        "n_prompts": len(question_indices),
        "instruction_indices": instruction_indices,
        "n_instruction_indices": len(instruction_indices),
        "expected_n_prompts_times_instruction_indices": int(expected_count),
        "actual_common_keys": int(len(ordered_keys)),
        "family_stats": family_stats,
        "pca_info": pca_info,
        "artifacts": {
            "coords_csv": str(coords_csv_path),
            "plotly_html": str(plotly_html_path),
            "gif": gif_info["gif"],
            "ai_tensor": str(out_dir / "ai_prompt_instruction_steering.pt"),
            "user_tensor": str(out_dir / "user_prompt_instruction_steering.pt"),
            "joint_tensor": str(out_dir / "joint_prompt_instruction_steering.pt"),
            "residue_tensor": str(out_dir / "residue_prompt_instruction_steering.pt"),
            "residue_scree": residue_scree_info,
        },
        "output_dir": str(out_dir),
    }

    summary_path = out_dir / "four_group_layerwise_pca_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved four-group layer-wise PCA animation artifacts:")
    print(f"- coords CSV: {coords_csv_path}")
    print(f"- plotly HTML: {plotly_html_path}")
    print(f"- GIF: {gif_info['gif']}")
    print(f"- residue scree dir: {residue_scree_info['scree_dir']}")
    print(f"- summary: {summary_path}")


def write_plotly_outputs(fig, html_path: Path, png_path: Path) -> None:
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        # PNG export depends on optional backends (for example kaleido).
        pass


def write_layerwise_animation(
    all_df: pd.DataFrame,
    gif_path: Path,
    mp4_path: Path | None,
    fps: int,
    dpi: int,
) -> dict:
    layers = sorted(int(x) for x in all_df["layer"].dropna().unique().tolist())
    if not layers:
        raise ValueError("No layers found for animation")

    x_vals = all_df["pc1"].to_numpy(dtype=np.float32)
    y_vals = all_df["pc2"].to_numpy(dtype=np.float32)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())

    x_margin = max(1e-6, 0.05 * (x_max - x_min))
    y_margin = max(1e-6, 0.05 * (y_max - y_min))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)

    instruction_values = sorted(
        int(v) for v in all_df["instruction_idx"].dropna().astype(int).unique().tolist()
    )
    cmap = plt.get_cmap("tab10")
    color_lookup = {val: cmap(i % 10) for i, val in enumerate(instruction_values)}
    default_color = (0.3, 0.3, 0.3, 0.9)

    scatter = ax.scatter([], [], s=24, alpha=0.85)
    title = ax.set_title("")

    def update(frame_idx: int):
        layer = layers[frame_idx]
        subset = all_df[all_df["layer"] == layer]
        xy = subset[["pc1", "pc2"]].to_numpy(dtype=np.float32)
        scatter.set_offsets(xy)

        colors = []
        for value in subset["instruction_idx"].tolist():
            if pd.isna(value):
                colors.append(default_color)
            else:
                colors.append(color_lookup.get(int(value), default_color))
        scatter.set_color(colors)
        title.set_text(f"Prompt-wise PCA Layer Animation (Layer {layer})")
        return scatter, title

    interval_ms = int(1000 / max(fps, 1))
    anim = FuncAnimation(fig, update, frames=len(layers), interval=interval_ms, blit=False)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=dpi)

    saved_mp4: str | None = None
    if mp4_path is not None:
        try:
            anim.save(str(mp4_path), fps=fps, dpi=dpi)
            saved_mp4 = str(mp4_path)
        except Exception:
            saved_mp4 = None

    plt.close(fig)
    return {
        "gif": str(gif_path),
        "mp4": saved_mp4,
        "fps": int(fps),
        "dpi": int(dpi),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Advanced prompt-wise PCA and 3D layer-evolution analysis for pos-neg persona residues"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--pos_csv", type=str, required=True)
    parser.add_argument("--neg_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/promptwise_pca")

    parser.add_argument("--layers", type=str, default="15-28")
    parser.add_argument("--single_layer", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--question_col", type=str, default="question")
    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--answer_col", type=str, default="answer")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--make_layer_animation", action="store_true")
    parser.add_argument("--animation_fps", type=int, default=2)
    parser.add_argument("--animation_dpi", type=int, default=140)

    parser.add_argument("--make_four_group_animation", action="store_true")
    parser.add_argument("--ai_pos_csv", type=str, default=None)
    parser.add_argument("--ai_neg_csv", type=str, default=None)
    parser.add_argument("--user_pos_csv", type=str, default=None)
    parser.add_argument("--user_neg_csv", type=str, default=None)
    parser.add_argument("--pca_fit_mode", type=str, default="global", choices=["global", "per_layer"])
    parser.add_argument("--repeat_agg", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--residue_scree_max_components", type=int, default=100)

    args = parser.parse_args()
    set_seed(args.seed)

    args.pos_csv = str(resolve_input_path(args.pos_csv))
    args.neg_csv = str(resolve_input_path(args.neg_csv))
    if args.ai_pos_csv:
        args.ai_pos_csv = str(resolve_input_path(args.ai_pos_csv))
    if args.ai_neg_csv:
        args.ai_neg_csv = str(resolve_input_path(args.ai_neg_csv))
    if args.user_pos_csv:
        args.user_pos_csv = str(resolve_input_path(args.user_pos_csv))
    if args.user_neg_csv:
        args.user_neg_csv = str(resolve_input_path(args.user_neg_csv))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_ids = parse_layer_spec(args.layers)

    model, tokenizer = load_hf_model_and_tokenizer(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        load_in_8bit=bool(args.load_in_8bit),
    )
    validate_layer_ids(layer_ids=layer_ids, model=model)

    if args.make_four_group_animation:
        run_four_group_layer_animation(
            args=args,
            model=model,
            tokenizer=tokenizer,
            layer_ids=layer_ids,
            out_dir=out_dir,
        )
        return

    pos_df = pd.read_csv(args.pos_csv)
    neg_df = pd.read_csv(args.neg_csv)

    pos_df, neg_df, align_info = align_pos_neg_rows(
        pos_df=pos_df,
        neg_df=neg_df,
        question_col=args.question_col,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
        max_samples=args.max_samples,
    )

    if len(pos_df) < 2:
        raise ValueError("Need at least 2 aligned pos/neg rows for prompt-wise PCA analysis")

    if args.single_layer not in layer_ids:
        raise ValueError(
            f"--single_layer={args.single_layer} must be inside --layers ({layer_ids})"
        )

    prompt_diffs = []
    prompt_meta_rows = []

    for i in tqdm(range(len(pos_df)), desc="Collecting prompt-wise diffs"):
        pos_prompt = str(pos_df.iloc[i][args.prompt_col])
        pos_answer = str(pos_df.iloc[i][args.answer_col])
        neg_prompt = str(neg_df.iloc[i][args.prompt_col])
        neg_answer = str(neg_df.iloc[i][args.answer_col])

        pos_layer_vecs = get_response_avg_by_layer(
            model=model,
            tokenizer=tokenizer,
            prompt_text=pos_prompt,
            answer_text=pos_answer,
            layer_ids=layer_ids,
        )
        neg_layer_vecs = get_response_avg_by_layer(
            model=model,
            tokenizer=tokenizer,
            prompt_text=neg_prompt,
            answer_text=neg_answer,
            layer_ids=layer_ids,
        )

        diff = pos_layer_vecs - neg_layer_vecs  # [L, d]
        prompt_diffs.append(diff)

        prompt_meta_rows.append(
            {
                "sample_idx": i,
                "question": str(pos_df.iloc[i][args.question_col]),
                "instruction_idx": int(pos_df.iloc[i]["instruction_idx"]) if "instruction_idx" in pos_df.columns else -1,
                "prompt": pos_prompt,
            }
        )

    prompt_diffs_tensor = torch.stack(prompt_diffs, dim=0)  # [N, L, d]
    n_samples, n_layers_used, hidden_dim = prompt_diffs_tensor.shape

    torch.save(prompt_diffs_tensor, out_dir / "promptwise_pos_minus_neg_diff.pt")

    prompt_meta_df = pd.DataFrame(prompt_meta_rows)
    prompt_meta_df.to_csv(out_dir / "prompt_metadata.csv", index=False)

    # Single-layer prompt-wise PCA
    single_layer_local_idx = layer_ids.index(args.single_layer)
    x_single = prompt_diffs_tensor[:, single_layer_local_idx, :].cpu().numpy()
    coords_single, components_single, ratio_single, mean_single = pca_svd(x_single, n_components=3)

    single_df = prompt_meta_df.copy()
    single_df["layer"] = args.single_layer
    single_df["pc1"] = coords_single[:, 0]
    single_df["pc2"] = coords_single[:, 1] if coords_single.shape[1] > 1 else 0.0
    single_df["pc3"] = coords_single[:, 2] if coords_single.shape[1] > 2 else 0.0
    single_df.to_csv(out_dir / f"single_layer_{args.single_layer}_promptwise_pca.csv", index=False)

    np.savez(
        out_dir / f"single_layer_{args.single_layer}_pca_artifacts.npz",
        components=components_single,
        explained_ratio=ratio_single,
        mean=mean_single,
        layer=np.array([args.single_layer], dtype=np.int32),
    )

    fig_single = px.scatter_3d(
        single_df,
        x="pc1",
        y="pc2",
        z="pc3",
        hover_data=["sample_idx", "instruction_idx", "question"],
        title=f"Prompt-wise PCA at Layer {args.single_layer}",
    )
    write_plotly_outputs(
        fig_single,
        out_dir / f"single_layer_{args.single_layer}_promptwise_pca_3d.html",
        out_dir / f"single_layer_{args.single_layer}_promptwise_pca_3d.png",
    )

    # All-layer PCA and layerwise evolution trajectory
    x_all = prompt_diffs_tensor.reshape(n_samples * n_layers_used, hidden_dim).cpu().numpy()
    coords_all, components_all, ratio_all, mean_all = pca_svd(x_all, n_components=3)

    sample_index = np.repeat(np.arange(n_samples), n_layers_used)
    layer_index = np.tile(np.array(layer_ids, dtype=np.int32), n_samples)

    all_df = pd.DataFrame(
        {
            "sample_idx": sample_index,
            "layer": layer_index,
            "pc1": coords_all[:, 0],
            "pc2": coords_all[:, 1] if coords_all.shape[1] > 1 else 0.0,
            "pc3": coords_all[:, 2] if coords_all.shape[1] > 2 else 0.0,
        }
    )
    all_df = all_df.merge(prompt_meta_df[["sample_idx", "instruction_idx", "question"]], on="sample_idx", how="left")
    all_df.to_csv(out_dir / "all_layers_promptwise_pca_coords.csv", index=False)

    layer_traj_df = (
        all_df.groupby("layer", as_index=False)[["pc1", "pc2", "pc3"]]
        .mean()
        .sort_values("layer")
        .reset_index(drop=True)
    )
    layer_traj_df.to_csv(out_dir / "layerwise_mean_trajectory_3d.csv", index=False)

    np.savez(
        out_dir / "all_layers_pca_artifacts.npz",
        components=components_all,
        explained_ratio=ratio_all,
        mean=mean_all,
        layers=np.array(layer_ids, dtype=np.int32),
    )

    fig_all = px.scatter_3d(
        all_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="layer",
        hover_data=["sample_idx", "instruction_idx", "question"],
        title="Prompt-wise PCA Across Layers",
        opacity=0.55,
    )
    fig_all.add_trace(
        go.Scatter3d(
            x=layer_traj_df["pc1"],
            y=layer_traj_df["pc2"],
            z=layer_traj_df["pc3"],
            mode="lines+markers+text",
            text=[str(int(x)) for x in layer_traj_df["layer"]],
            textposition="top center",
            name="Layerwise Mean Trajectory",
            line={"width": 6},
            marker={"size": 5},
        )
    )
    write_plotly_outputs(
        fig_all,
        out_dir / "all_layers_promptwise_pca_3d_with_trajectory.html",
        out_dir / "all_layers_promptwise_pca_3d_with_trajectory.png",
    )

    # Optional per-layer PCA explained ratio summary for diagnostics.
    per_layer_stats = []
    for local_idx, layer in enumerate(layer_ids):
        x_layer = prompt_diffs_tensor[:, local_idx, :].cpu().numpy()
        _, _, ratio_layer, _ = pca_svd(x_layer, n_components=3)
        per_layer_stats.append(
            {
                "layer": int(layer),
                "pc1_explained_ratio": float(ratio_layer[0]) if len(ratio_layer) > 0 else 0.0,
                "pc2_explained_ratio": float(ratio_layer[1]) if len(ratio_layer) > 1 else 0.0,
                "pc3_explained_ratio": float(ratio_layer[2]) if len(ratio_layer) > 2 else 0.0,
                "pc1_to_pc3_sum": float(ratio_layer[:3].sum()),
            }
        )

    per_layer_df = pd.DataFrame(per_layer_stats)
    per_layer_df.to_csv(out_dir / "per_layer_promptwise_pca_explained_ratio.csv", index=False)

    layer_animation = None
    if args.make_layer_animation:
        layer_animation = write_layerwise_animation(
            all_df=all_df,
            gif_path=out_dir / "layerwise_prompt_pca_animation.gif",
            mp4_path=out_dir / "layerwise_prompt_pca_animation.mp4",
            fps=args.animation_fps,
            dpi=args.animation_dpi,
        )

    summary = {
        "model_name": args.model_name,
        "pos_csv": args.pos_csv,
        "neg_csv": args.neg_csv,
        "align_info": align_info,
        "layers": layer_ids,
        "single_layer": int(args.single_layer),
        "n_samples": int(n_samples),
        "n_layers_used": int(n_layers_used),
        "hidden_dim": int(hidden_dim),
        "single_layer_explained_ratio": ratio_single.tolist(),
        "all_layers_explained_ratio": ratio_all.tolist(),
        "layer_animation": layer_animation,
        "output_dir": str(out_dir),
    }

    (out_dir / "promptwise_pca_summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved prompt-wise PCA artifacts:")
    print(f"- tensor: {out_dir / 'promptwise_pos_minus_neg_diff.pt'}")
    print(f"- single-layer PCA CSV: {out_dir / f'single_layer_{args.single_layer}_promptwise_pca.csv'}")
    print(f"- all-layer PCA CSV: {out_dir / 'all_layers_promptwise_pca_coords.csv'}")
    print(f"- 3D trajectory CSV: {out_dir / 'layerwise_mean_trajectory_3d.csv'}")
    if layer_animation is not None:
        print(f"- layer animation GIF: {layer_animation['gif']}")
        if layer_animation["mp4"] is None:
            print("- layer animation MP4: skipped (ffmpeg unavailable or save failed)")
        else:
            print(f"- layer animation MP4: {layer_animation['mp4']}")
    print(f"- summary: {out_dir / 'promptwise_pca_summary.json'}")


if __name__ == "__main__":
    main()
