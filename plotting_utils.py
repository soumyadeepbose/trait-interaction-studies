import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import math



def _write_plotly_html(fig, png_path: str) -> str:
    png_dir = os.path.dirname(png_path)
    if os.path.basename(png_dir) == "png":
        html_dir = os.path.join(os.path.dirname(png_dir), "html")
    else:
        html_dir = png_dir
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, os.path.basename(png_path).replace(".png", ".html"))
    fig.write_html(html_path, include_plotlyjs="cdn")
    return html_path


def _resolve_plot_csv_dir(out_dir: str) -> str:
    if os.path.basename(out_dir) == "png":
        return os.path.join(os.path.dirname(out_dir), "csvs")
    return os.path.join(out_dir, "csvs")


def _write_plot_csv(df: pd.DataFrame, png_path: str, out_dir: str) -> str:
    csv_dir = _resolve_plot_csv_dir(out_dir)
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = os.path.basename(png_path).replace(".png", ".csv")
    csv_path = os.path.join(csv_dir, csv_name)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_dual_projection_vectors(vec_dir: str, ai_trait: str, user_trait: str, joint_trait: str):
    """Load vectors needed for dual projection plotting."""
    u_vec = torch.load(
        f"{vec_dir}/{user_trait}_response_avg_diff.pt",
        map_location="cpu",
        weights_only=False,
    )
    a_vec = torch.load(
        f"{vec_dir}/{ai_trait}_response_avg_diff.pt",
        map_location="cpu",
        weights_only=False,
    )
    residue = torch.load(
        f"{vec_dir}/residue_{joint_trait}_response_avg_diff.pt",
        map_location="cpu",
        weights_only=False,
    )
    return u_vec, a_vec, residue


def build_dual_projection_df(u_vec: torch.Tensor, a_vec: torch.Tensor, residue: torch.Tensor) -> pd.DataFrame:
    """Compute layerwise scalar projections and benchmark norms."""
    norm_u = torch.norm(u_vec, p=2, dim=1)
    dot_prod_u = torch.sum(residue * u_vec, dim=1)
    scalar_proj_u = dot_prod_u / (norm_u + 1e-8)

    norm_a = torch.norm(a_vec, p=2, dim=1)
    dot_prod_a = torch.sum(residue * a_vec, dim=1)
    scalar_proj_a = dot_prod_a / (norm_a + 1e-8)

    layers = np.arange(len(scalar_proj_u))
    return pd.DataFrame(
        {
            "layer": layers,
            "proj_user": scalar_proj_u.detach().cpu().numpy(),
            "neg_norm_user": -1 * norm_u.detach().cpu().numpy(),
            "proj_ai": scalar_proj_a.detach().cpu().numpy(),
            "neg_norm_ai": -1 * norm_a.detach().cpu().numpy(),
        }
    )


def save_dual_projection_plot(df_proj: pd.DataFrame, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Save notebook-style dual projection plot and return output path."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 6), dpi=100)

    sns.lineplot(
        data=df_proj,
        x="layer",
        y="proj_user",
        color="#D32F2F",
        linewidth=2.5,
        marker="o",
        label="Proj. of Residue onto User Vector",
    )
    sns.lineplot(
        data=df_proj,
        x="layer",
        y="neg_norm_user",
        color="#FFCDD2",
        linestyle="--",
        linewidth=2,
        label="- ||User Vector|| (Benchmark)",
    )

    sns.lineplot(
        data=df_proj,
        x="layer",
        y="proj_ai",
        color="#1976D2",
        linewidth=2.5,
        marker="s",
        label="Proj. of Residue onto AI Vector",
    )
    sns.lineplot(
        data=df_proj,
        x="layer",
        y="neg_norm_ai",
        color="#BBDEFB",
        linestyle="--",
        linewidth=2,
        label="- ||AI Vector|| (Benchmark)",
    )

    plt.axhline(0, color="black", linewidth=1.5, ls="-")
    plt.xlabel("Layer")
    plt.ylabel("Projection / Negative Norm")
    plt.title(f"Interaction Residue Projections for '{base_trait.capitalize()}'")
    plt.legend(loc="best", fontsize="small")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plot_path = f"{out_dir}/{model_short}_{base_trait}_dual_projection.png"
    plt.savefig(plot_path)
    plt.close()
    _write_plot_csv(df_proj, plot_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_proj["layer"], y=df_proj["proj_user"], mode="lines+markers", name="Proj. of Residue onto User Vector", line=dict(color="#D32F2F")))
    fig.add_trace(go.Scatter(x=df_proj["layer"], y=df_proj["neg_norm_user"], mode="lines", name="- ||User Vector|| (Benchmark)", line=dict(color="#FFCDD2", dash="dash")))
    fig.add_trace(go.Scatter(x=df_proj["layer"], y=df_proj["proj_ai"], mode="lines+markers", name="Proj. of Residue onto AI Vector", line=dict(color="#1976D2")))
    fig.add_trace(go.Scatter(x=df_proj["layer"], y=df_proj["neg_norm_ai"], mode="lines", name="- ||AI Vector|| (Benchmark)", line=dict(color="#BBDEFB", dash="dash")))
    fig.update_layout(
        title=f"Interaction Residue Projections for '{base_trait.capitalize()}'",
        xaxis_title="Layer",
        yaxis_title="Projection / Negative Norm",
        template="plotly_white",
    )
    _write_plotly_html(fig, plot_path)
    return plot_path


def save_similarity_tile_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Notebook-style tiled heatmap of layerwise cosine similarity between AI and User vectors."""
    os.makedirs(out_dir, exist_ok=True)
    cos_vals = F.cosine_similarity(a_vec, u_vec, dim=1).detach().cpu().numpy()
    df = pd.DataFrame({"layer": np.arange(len(cos_vals)), "cosine_similarity": cos_vals})

    n = len(df)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid_data = np.full((rows * cols,), np.nan)
    grid_data[:n] = df["cosine_similarity"].values
    grid_data = grid_data.reshape(rows, cols)

    plt.figure(figsize=(6, 6), dpi=100)
    ax = sns.heatmap(
        grid_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.44,
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.5,
        linecolor="white",
    )

    for i in range(n):
        r, c = divmod(i, cols)
        layer_num = df.iloc[i]["layer"]
        ax.text(c + 0.05, r + 0.1, str(layer_num), fontsize=7, color="black", ha="left", va="top", weight="bold")

    plt.title("Layerwise Similarity: AI vs User Steering Vectors", fontsize=12)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_similarity_tiles.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)
    return out_path


def save_norms_comparison_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Line plot comparing norms of base vectors and residue."""
    os.makedirs(out_dir, exist_ok=True)
    l = len(residue)
    df_norms = pd.DataFrame(
        {
            "layer": np.arange(l),
            "residue_norm": residue.norm(dim=1).cpu().numpy(),
            "ai_norm": a_vec.norm(dim=1).cpu().numpy(),
            "user_norm": u_vec.norm(dim=1).cpu().numpy(),
            "joint_norm": j_vec.norm(dim=1).cpu().numpy(),
        }
    )

    plt.figure(figsize=(8, 5), dpi=100)
    sns.lineplot(data=df_norms, x="layer", y="ai_norm", label="AI", marker="o", color="tab:blue")
    sns.lineplot(data=df_norms, x="layer", y="user_norm", label="User", marker="s", color="tab:orange")
    sns.lineplot(data=df_norms, x="layer", y="joint_norm", label="Joint", marker="^", color="tab:green")
    sns.lineplot(data=df_norms, x="layer", y="residue_norm", label="Residue", marker="X", color="tab:red", linestyle="--", linewidth=2.5)
    plt.title("Magnitude Analysis: Base Vectors vs Interaction Residue")
    plt.xlabel("Layer")
    plt.ylabel("L2 Norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_norms_comparison.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df_norms, out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_norms["layer"], y=df_norms["ai_norm"], mode="lines+markers", name="AI"))
    fig.add_trace(go.Scatter(x=df_norms["layer"], y=df_norms["user_norm"], mode="lines+markers", name="User"))
    fig.add_trace(go.Scatter(x=df_norms["layer"], y=df_norms["joint_norm"], mode="lines+markers", name="Joint"))
    fig.add_trace(go.Scatter(x=df_norms["layer"], y=df_norms["residue_norm"], mode="lines+markers", name="Residue", line=dict(dash="dash")))
    fig.update_layout(
        title="Magnitude Analysis: Base Vectors vs Interaction Residue",
        xaxis_title="Layer",
        yaxis_title="L2 Norm",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def save_residue_cosine_heatmap(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Line plot of residue cosine similarity against AI/User/Joint vectors across layers."""
    os.makedirs(out_dir, exist_ok=True)
    cos_res_a = [F.cosine_similarity(residue[layer], a_vec[layer], dim=0).item() for layer in range(len(residue))]
    cos_res_u = [F.cosine_similarity(residue[layer], u_vec[layer], dim=0).item() for layer in range(len(residue))]
    cos_res_j = [F.cosine_similarity(residue[layer], j_vec[layer], dim=0).item() for layer in range(len(residue))]

    df = pd.DataFrame(
        {
            "layer": np.arange(len(residue)),
            "cos_res_ai": cos_res_a,
            "cos_res_user": cos_res_u,
            "cos_res_joint": cos_res_j,
        }
    )

    plt.figure(figsize=(8, 5), dpi=100)
    sns.lineplot(data=df, x="layer", y="cos_res_ai", marker="o", color="tab:blue", label="Residue vs AI")
    sns.lineplot(data=df, x="layer", y="cos_res_user", marker="s", color="tab:orange", label="Residue vs User")
    sns.lineplot(data=df, x="layer", y="cos_res_joint", marker="^", color="tab:green", label="Residue vs Joint")
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Layerwise Similarity: Residue vs Persona Vectors")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_residue_cosine_lineplot.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["layer"], y=df["cos_res_ai"], mode="lines+markers", name="Residue vs AI"))
    fig.add_trace(go.Scatter(x=df["layer"], y=df["cos_res_user"], mode="lines+markers", name="Residue vs User"))
    fig.add_trace(go.Scatter(x=df["layer"], y=df["cos_res_joint"], mode="lines+markers", name="Residue vs Joint"))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        title="Layerwise Similarity: Residue vs Persona Vectors",
        xaxis_title="Layer",
        yaxis_title="Cosine Similarity",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def _directional_residue_norm(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor) -> np.ndarray:
    """Compute || normalize(J) - (normalize(A) + normalize(U)) || per layer."""
    a_normed = F.normalize(a_vec, p=2, dim=1)
    u_normed = F.normalize(u_vec, p=2, dim=1)
    j_normed = F.normalize(j_vec, p=2, dim=1)
    residue_dir = j_normed - (a_normed + u_normed)
    return torch.norm(residue_dir, p=2, dim=1).detach().cpu().numpy()


def save_synergy_divergence_plot(
    u_vec: torch.Tensor,
    a_vec: torch.Tensor,
    j_vec: torch.Tensor,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
    synergy_vectors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> str:
    """
    Plot directional residue norm for divergence (main case) and optional synergy control.

    synergy_vectors: optional tuple (user_aligned_vec, joint_aligned_vec), with AI vector fixed.
    """
    os.makedirs(out_dir, exist_ok=True)
    divergence_norm = _directional_residue_norm(u_vec, a_vec, j_vec)
    df = pd.DataFrame({"layer": np.arange(len(divergence_norm)), "divergence_norm": divergence_norm})

    if synergy_vectors is not None:
        u_synergy, j_synergy = synergy_vectors
        synergy_norm = _directional_residue_norm(u_synergy, a_vec, j_synergy)
        df["synergy_norm"] = synergy_norm

    plt.figure(figsize=(8, 5), dpi=100)
    sns.lineplot(data=df, x="layer", y="divergence_norm", marker="o", color="tab:red", label="Divergence (Conflict)")

    if "synergy_norm" in df.columns:
        sns.lineplot(
            data=df,
            x="layer",
            y="synergy_norm",
            marker="s",
            color="tab:blue",
            linestyle="--",
            label="Synergy (Aligned)",
        )

    plt.xlabel("Layer")
    plt.ylabel("Directional Residue Norm (Unit Space)")
    plt.title("Synergy vs Divergence Residual Norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = f"{out_dir}/{model_short}_{base_trait}_synergy_divergence_residue_norm.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["layer"], y=df["divergence_norm"], mode="lines+markers", name="Divergence (Conflict)"))
    if "synergy_norm" in df.columns:
        fig.add_trace(go.Scatter(x=df["layer"], y=df["synergy_norm"], mode="lines+markers", name="Synergy (Aligned)", line=dict(dash="dash")))
    fig.update_layout(
        title="Synergy vs Divergence Residual Norm",
        xaxis_title="Layer",
        yaxis_title="Directional Residue Norm (Unit Space)",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def _build_pca_points(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack all vectors as points for PCA diagnostics."""
    points = []
    meta = []
    for layer in range(len(u_vec)):
        points.append(a_vec[layer].detach().cpu().numpy())
        meta.append({"layer": layer, "vector_type": "AI"})

        points.append(u_vec[layer].detach().cpu().numpy())
        meta.append({"layer": layer, "vector_type": "User"})

        points.append(j_vec[layer].detach().cpu().numpy())
        meta.append({"layer": layer, "vector_type": "Joint"})

        points.append(residue[layer].detach().cpu().numpy())
        meta.append({"layer": layer, "vector_type": "Residue"})

    return np.vstack(points), pd.DataFrame(meta)


def _fit_pca(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD; returns scores and explained variance ratio."""
    centered = data - data.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T
    denom = max(centered.shape[0] - 1, 1)
    explained = (s**2) / denom
    ratio = explained / (explained.sum() + 1e-12)
    return scores, ratio


def save_pca_scree_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots", max_components: int = 10) -> str:
    """Save PCA scree plot for vector interpretability."""
    os.makedirs(out_dir, exist_ok=True)
    points, _ = _build_pca_points(u_vec, a_vec, j_vec, residue)
    _, ratio = _fit_pca(points)
    k = min(max_components, len(ratio))

    df = pd.DataFrame({"component": np.arange(1, k + 1), "explained_ratio": ratio[:k]})
    plt.figure(figsize=(7, 4.5), dpi=100)
    sns.lineplot(data=df, x="component", y="explained_ratio", marker="o", color="tab:purple")
    plt.bar(df["component"], df["explained_ratio"], alpha=0.25, color="tab:purple")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"{out_dir}/{model_short}_{base_trait}_pca_scree.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["component"], y=df["explained_ratio"], name="Explained Ratio", opacity=0.35))
    fig.add_trace(go.Scatter(x=df["component"], y=df["explained_ratio"], mode="lines+markers", name="Scree"))
    fig.update_layout(
        title="PCA Scree Plot",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def save_pca_2d_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Save 2D PCA projection over all vector points."""
    os.makedirs(out_dir, exist_ok=True)
    points, meta = _build_pca_points(u_vec, a_vec, j_vec, residue)
    scores, ratio = _fit_pca(points)

    meta = meta.copy()
    meta["pc1"] = scores[:, 0]
    meta["pc2"] = scores[:, 1]

    plt.figure(figsize=(7, 6), dpi=100)
    sns.scatterplot(
        data=meta,
        x="pc1",
        y="pc2",
        hue="vector_type",
        style="vector_type",
        alpha=0.75,
        s=45,
    )
    plt.xlabel(f"PC1 ({ratio[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({ratio[1] * 100:.1f}% var)")
    plt.title("PCA 2D Projection of Persona/Residue Vectors")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = f"{out_dir}/{model_short}_{base_trait}_pca_2d.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(meta[["layer", "vector_type", "pc1", "pc2"]], out_path, out_dir)

    fig = px.scatter(
        meta,
        x="pc1",
        y="pc2",
        color="vector_type",
        symbol="vector_type",
        hover_data=["layer"],
        title="PCA 2D Projection of Persona/Residue Vectors",
    )
    fig.update_layout(
        xaxis_title=f"PC1 ({ratio[0] * 100:.1f}% var)",
        yaxis_title=f"PC2 ({ratio[1] * 100:.1f}% var)",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def save_pca_3d_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, residue: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Save 3D PCA projection over all vector points."""
    os.makedirs(out_dir, exist_ok=True)
    points, meta = _build_pca_points(u_vec, a_vec, j_vec, residue)
    scores, ratio = _fit_pca(points)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "AI": "tab:blue",
        "User": "tab:orange",
        "Joint": "tab:green",
        "Residue": "tab:red",
    }

    for vector_type, color in colors.items():
        idx = meta["vector_type"] == vector_type
        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            scores[idx, 2],
            c=color,
            label=vector_type,
            alpha=0.7,
            s=25,
        )

    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}% var)")
    ax.set_zlabel(f"PC3 ({ratio[2] * 100:.1f}% var)")
    ax.set_title("PCA 3D Projection of Persona/Residue Vectors")
    ax.legend(loc="best")
    plt.tight_layout()

    out_path = f"{out_dir}/{model_short}_{base_trait}_pca_3d.png"
    plt.savefig(out_path)
    plt.close(fig)

    meta_plotly = meta.copy()
    meta_plotly["pc1"] = scores[:, 0]
    meta_plotly["pc2"] = scores[:, 1]
    meta_plotly["pc3"] = scores[:, 2]
    _write_plot_csv(meta_plotly[["layer", "vector_type", "pc1", "pc2", "pc3"]], out_path, out_dir)
    fig3d = px.scatter_3d(
        meta_plotly,
        x="pc1",
        y="pc2",
        z="pc3",
        color="vector_type",
        symbol="vector_type",
        hover_data=["layer"],
        title="PCA 3D Projection of Persona/Residue Vectors",
    )
    fig3d.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({ratio[0] * 100:.1f}% var)",
            yaxis_title=f"PC2 ({ratio[1] * 100:.1f}% var)",
            zaxis_title=f"PC3 ({ratio[2] * 100:.1f}% var)",
        ),
        template="plotly_white",
    )
    _write_plotly_html(fig3d, out_path)
    return out_path


def save_interaction_dynamics_plot(u_vec: torch.Tensor, a_vec: torch.Tensor, j_vec: torch.Tensor, model_short: str, base_trait: str, out_dir: str = "output/plots") -> str:
    """Dual-axis dynamics plot: cos(AI,User) and normalized directional residue norm."""
    os.makedirs(out_dir, exist_ok=True)
    cos_sim = F.cosine_similarity(a_vec, u_vec, dim=1).detach().cpu().numpy()
    a_normed = F.normalize(a_vec, p=2, dim=1)
    u_normed = F.normalize(u_vec, p=2, dim=1)
    j_normed = F.normalize(j_vec, p=2, dim=1)
    residue_dir = j_normed - (a_normed + u_normed)
    residue_norm = torch.norm(residue_dir, p=2, dim=1).detach().cpu().numpy()
    df = pd.DataFrame({"layer": np.arange(len(cos_sim)), "cosine_sim": cos_sim, "residue_norm": residue_norm})

    fig, ax1 = plt.subplots(figsize=(7, 4.8), dpi=100)
    sns.lineplot(data=df, x="layer", y="cosine_sim", ax=ax1, color="tab:blue", marker="o", label="Cos Sim (AI vs User)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Cosine Similarity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="layer", y="residue_norm", ax=ax2, color="tab:red", marker="o", label="Directional Residue Norm")
    ax2.set_ylabel("Unit Space Residue Norm", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=True)
    plt.title("Interaction Dynamics: Similarity vs Divergence")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    out_path = f"{out_dir}/{model_short}_{base_trait}_interaction_dynamics.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)
    return out_path


"""
ADDITIONS TO plotting_utils.py
────────────────────────────────
Append these functions to the bottom of your existing plotting_utils.py.
Import needs at top of plotting_utils.py (if not already present):
    import math
    from pathlib import Path
"""

# ── EffRank of residue by layer ───────────────────────────────────────────────

def _load_metric_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "layer" in df.columns:
        df = df.sort_values("layer").reset_index(drop=True)
    return df


def save_effrank_plot(
    effrank_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot effective rank of per-prompt residue by layer.
    High EffRank = residue is high-dimensional / distributed (not a single safety direction).
    Low EffRank = residue concentrated in a small subspace.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(effrank_csv_path)
    layers = df["layer"].to_numpy()
    er = df["effective_rank"].to_numpy()

    plt.figure(figsize=(8, 4), dpi=100)
    plt.fill_between(layers, er, color="#D6E4F7", alpha=0.6)
    plt.plot(layers, er, marker="o", color="#2E5FA3", linewidth=2.5, label="Effective Rank")
    plt.axhline(1.0, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="EffRank=1 (single direction)")
    plt.xlabel("Layer")
    plt.ylabel("Effective Rank")
    plt.title("Effective Rank of Per-Prompt Residue by Layer\n"
              "(high = distributed, no single safety direction; low = concentrated subspace)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_effrank.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=er, mode="lines+markers", fill="tozeroy",
                             name="Effective Rank", line=dict(color="#2E5FA3")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5)
    fig.update_layout(title="Effective Rank of Per-Prompt Residue", xaxis_title="Layer",
                      yaxis_title="Effective Rank", template="plotly_white")
    _write_plotly_html(fig, out_path)
    return out_path


# ── Residue decomposition: ||R^||A||, ||R^||U||, ||R^perp|| ──────────────────

def save_projection_norms_plot(
    metrics_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot norms of three residue decomposition components per layer:
      R^||A  — component aligned with AI persona direction
      R^||U  — component aligned with user persona direction
      R^perp — component orthogonal to both (Novel Component; R^perp is (d-2)-dimensional)
    Also plots NCF (Novel Component Fraction = ||R^perp||^2 / ||tau_AU||^2) on a second axis.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(metrics_csv_path)
    layers = df["layer"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # Left: absolute norms
    ax1.plot(layers, df["norm_R_parallel_A"], marker="o", color="#2E5FA3", linewidth=2,
             label=r"$\|R^{\parallel A}\|$: aligned with AI")
    ax1.plot(layers, df["norm_R_parallel_U"], marker="s", color="#C8902A", linewidth=2,
             label=r"$\|R^{\parallel U}\|$: aligned with User")
    ax1.plot(layers, df["norm_R_perp"], marker="^", color="#5B2C8C", linewidth=2,
             label=r"$\|R^{\perp}\|$: novel (orthogonal to both)")
    ax1.plot(layers, df["norm_tau_AU"], marker="x", color="#8B1A1A", linewidth=1.5,
             linestyle="--", label=r"$\|\tau_{A,U}\|$: total residue norm")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("L2 Norm")
    ax1.set_title("Residue Decomposition Norms\n"
                  r"($R = R^{\parallel A} + R^{\parallel U} + R^{\perp}$)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: NCF
    ax2.fill_between(layers, df["ncf"], color="#EDE0FA", alpha=0.7)
    ax2.plot(layers, df["ncf"], marker="o", color="#5B2C8C", linewidth=2.5,
             label="NCF (Novel Component Fraction)")
    ax2.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.6, label="NCF=0.5")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("NCF")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Novel Component Fraction\n"
                  r"(NCF = $\|R^{\perp}\|^2 / \|\tau_{A,U}\|^2$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Residue Decomposition — {model_short} | {base_trait}", fontsize=12)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_projection_norms.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(
        df[["layer", "norm_R_parallel_A", "norm_R_parallel_U", "norm_R_perp", "norm_tau_AU", "ncf"]],
        out_path,
        out_dir,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=df["norm_R_parallel_A"], mode="lines+markers",
                             name="||R^||A||", line=dict(color="#2E5FA3")))
    fig.add_trace(go.Scatter(x=layers, y=df["norm_R_parallel_U"], mode="lines+markers",
                             name="||R^||U||", line=dict(color="#C8902A")))
    fig.add_trace(go.Scatter(x=layers, y=df["norm_R_perp"], mode="lines+markers",
                             name="||R^perp||", line=dict(color="#5B2C8C")))
    fig.add_trace(go.Scatter(x=layers, y=df["norm_tau_AU"], mode="lines+markers",
                             name="||tau_AU||", line=dict(color="#8B1A1A", dash="dash")))
    fig.update_layout(title="Residue Decomposition Norms", xaxis_title="Layer",
                      yaxis_title="L2 Norm", template="plotly_white")
    _write_plotly_html(fig, out_path)
    return out_path


# ── Amplification profile ─────────────────────────────────────────────────────

def save_amplification_plot(
    metrics_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot alpha(A|U) and alpha(U|A) across layers.
    alpha = 1 + (tau_AU · tau_X) / ||tau_X||^2
    baseline=1: U's presence does not change A's effective strength.
    <1: suppression. >1: amplification.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(metrics_csv_path)
    layers = df["layer"].to_numpy()

    plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(layers, df["alpha_A_given_U"], marker="o", color="#1A6B3C", linewidth=2,
             label=r"$\alpha(A|U)$: U's effect on A's direction")
    plt.plot(layers, df["alpha_U_given_A"], marker="s", color="#5B2C8C", linewidth=2,
             label=r"$\alpha(U|A)$: A's effect on U's direction")
    plt.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Baseline (no amplification)")
    plt.fill_between(layers, np.minimum(df["alpha_U_given_A"].to_numpy(), 1.0), 1.0,
                     color="#EDE0FA", alpha=0.4, label="A suppresses U region")
    plt.xlabel("Layer")
    plt.ylabel(r"$\alpha$")
    plt.title("Amplification Scores\n(<1: suppression, >1: amplification)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_amplification.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df[["layer", "alpha_A_given_U", "alpha_U_given_A"]], out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=df["alpha_A_given_U"], mode="lines+markers",
                             name="alpha(A|U)", line=dict(color="#1A6B3C")))
    fig.add_trace(go.Scatter(x=layers, y=df["alpha_U_given_A"], mode="lines+markers",
                             name="alpha(U|A)", line=dict(color="#5B2C8C")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(title="Amplification Scores", xaxis_title="Layer",
                      yaxis_title="alpha", template="plotly_white")
    _write_plotly_html(fig, out_path)
    return out_path


# ── ESVA plot ─────────────────────────────────────────────────────────────────

def save_esva_plot(
    esva_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot Extraction Side Variance Asymmetry (ESVA) by layer.
    ESVA > 0: pos-side activations more scattered (RLHF resistance signature).
    Spikes at late layers with residue norm growth = convergent RLHF contamination evidence.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(esva_csv_path)
    layers = df["layer"].to_numpy()
    esva = df["esva"].to_numpy()

    plt.figure(figsize=(8, 4), dpi=100)
    plt.bar(layers, esva, color=["#C8902A" if v > 0 else "#2E5FA3" for v in esva], alpha=0.8)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel("ESVA")
    plt.ylim(-1.05, 1.05)
    plt.title("Extraction Side Variance Asymmetry (ESVA)\n"
              "(ESVA>0: pos-side more scattered = RLHF asymmetric resistance signature)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_esva.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df[["layer", "esva"]], out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=layers, y=esva, name="ESVA",
                         marker_color=["#C8902A" if v > 0 else "#2E5FA3" for v in esva]))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(title="ESVA", xaxis_title="Layer", yaxis_title="ESVA",
                      yaxis_range=[-1.05, 1.05], template="plotly_white")
    _write_plotly_html(fig, out_path)
    return out_path


def save_projection_on_residue_plot(
    projection_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """Plot mean projection magnitudes on residue direction by layer."""
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(projection_csv_path)
    layers = df["layer"].to_numpy()

    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(layers, df["mean_abs_proj"], marker="o", color="#8B1A1A", linewidth=2,
             label="mean |<h, r_hat>|")
    if "mean_norm_proj" in df.columns:
        plt.plot(layers, df["mean_norm_proj"], marker="s", color="#2E5FA3", linewidth=2,
                 label="mean |<h, r_hat>| / ||h||")
    plt.xlabel("Layer")
    plt.ylabel("Projection")
    plt.title("Projection on Residue Direction")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_projection_on_residue.png"
    plt.savefig(out_path)
    plt.close()
    export_cols = ["layer", "mean_abs_proj"]
    if "mean_norm_proj" in df.columns:
        export_cols.append("mean_norm_proj")
    _write_plot_csv(df[export_cols], out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=df["mean_abs_proj"], mode="lines+markers",
                             name="mean_abs_proj", line=dict(color="#8B1A1A")))
    if "mean_norm_proj" in df.columns:
        fig.add_trace(go.Scatter(x=layers, y=df["mean_norm_proj"], mode="lines+markers",
                                 name="mean_norm_proj", line=dict(color="#2E5FA3")))
    fig.update_layout(
        title="Projection on Residue Direction",
        xaxis_title="Layer",
        yaxis_title="Projection",
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


def save_gbc_plot(
    gbc_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """Plot layerwise GBC scores (defaults to harmfulness if present)."""
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(gbc_csv_path)
    layers = df["layer"].to_numpy()

    gbc_col = "gbc_harmfulness"
    if gbc_col not in df.columns:
        candidate_cols = [c for c in df.columns if c != "layer"]
        if not candidate_cols:
            raise ValueError(f"No GBC metric column found in {gbc_csv_path}")
        gbc_col = candidate_cols[0]

    vals = df[gbc_col].to_numpy()
    colors = ["#C8902A" if v >= 0 else "#2E5FA3" for v in vals]

    plt.figure(figsize=(8, 4), dpi=100)
    plt.bar(layers, vals, color=colors, alpha=0.85)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel(gbc_col)
    plt.title("GBC by Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_gbc.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df[["layer", gbc_col]], out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=layers, y=vals, name=gbc_col, marker_color=colors))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(
        title="GBC by Layer",
        xaxis_title="Layer",
        yaxis_title=gbc_col,
        template="plotly_white",
    )
    _write_plotly_html(fig, out_path)
    return out_path


# ── rho_eff diagnostic plot ───────────────────────────────────────────────────

def save_rho_eff_plot(
    rho_eff_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot rho_eff and rho_rel per layer.
    rho_rel < 0.01: ablation mechanically negligible (explains null result without invoking non-identifiability).
    rho_rel > 0.05 with null behavioral result: genuine non-identifiability.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(rho_eff_csv_path)
    layers = df["layer"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    ax1.plot(layers, df["rho_eff"], marker="o", color="#1A5C6B", linewidth=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel(r"$\rho_{\mathrm{eff}}$")
    ax1.set_title(r"$\rho_\mathrm{eff}^{(\ell)}$: Mean abs projection of hidden states onto residue")
    ax1.grid(True, alpha=0.3)

    ax2.plot(layers, df["rho_rel"], marker="o", color="#8B1A1A", linewidth=2)
    ax2.axhline(0.05, color="orange", linewidth=0.8, linestyle="--", label="0.05 threshold (significant)")
    ax2.axhline(0.01, color="red", linewidth=0.8, linestyle=":", label="0.01 threshold (negligible below)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel(r"$\rho_{\mathrm{rel}}$")
    ax2.set_title(r"$\rho_\mathrm{rel}^{(\ell)}$: Fraction of hidden state energy along residue")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Effective Component Diagnostic — {model_short} | {base_trait}", fontsize=12)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_rho_eff.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df[["layer", "rho_eff", "rho_rel"]], out_path, out_dir)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=df["rho_rel"], mode="lines+markers",
                             name="rho_rel", line=dict(color="#8B1A1A")))
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", annotation_text="significant (0.05)")
    fig.add_hline(y=0.01, line_dash="dot", line_color="red", annotation_text="negligible (0.01)")
    fig.update_layout(title="rho_rel: Fraction of hidden state energy along residue direction",
                      xaxis_title="Layer", yaxis_title="rho_rel", template="plotly_white")
    _write_plotly_html(fig, out_path)
    return out_path


def save_activation_projection_plot(
    activation_projection_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """
    Plot layerwise activation projection on residue direction.
    Tracks the signed component that ablation scales via h' = h - coeff * <h, u> * u.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metric_csv(activation_projection_csv_path)
    layers = df["layer"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=100)

    ax1.plot(layers, df["mean_signed_pos"], marker="o", color="#8B1A1A", label="mean <h_pos, u>")
    ax1.plot(layers, df["mean_signed_neg"], marker="s", color="#1A5C6B", label="mean <h_neg, u>")
    ax1.plot(layers, df["mean_signed_delta"], marker="^", color="#5B2C8C", linestyle="--", label="mean (<h_pos,u>-<h_neg,u>)")
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Signed projection")
    ax1.set_title("Signed Activation Projection on Residue Direction")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(layers, df["mean_norm_abs_pos"], marker="o", color="#C8902A", label="mean |<h_pos,u>| / ||h_pos||")
    ax2.plot(layers, df["mean_norm_abs_neg"], marker="s", color="#2E5FA3", label="mean |<h_neg,u>| / ||h_neg||")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Normalized abs projection")
    ax2.set_title("Projection Energy Along Residue Direction")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.suptitle("Activation-Residue Projection Diagnostic", fontsize=12)
    plt.tight_layout()
    out_path = f"{out_dir}/{model_short}_{base_trait}_activation_projection.png"
    plt.savefig(out_path)
    plt.close()
    _write_plot_csv(df, out_path, out_dir)

    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=layers, y=df["mean_signed_pos"], mode="lines+markers", name="mean <h_pos,u>"))
    fig_int.add_trace(go.Scatter(x=layers, y=df["mean_signed_neg"], mode="lines+markers", name="mean <h_neg,u>"))
    fig_int.add_trace(go.Scatter(x=layers, y=df["mean_signed_delta"], mode="lines+markers", name="delta", line=dict(dash="dash")))
    fig_int.add_hline(y=0, line_color="black")
    fig_int.update_layout(
        title="Signed activation projection on residue direction",
        xaxis_title="Layer",
        yaxis_title="Projection",
        template="plotly_white",
    )
    _write_plotly_html(fig_int, out_path)
    return out_path


def save_plot_grid_dashboard(
    plot_paths: list[str],
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
) -> str:
    """Compile existing plot images into a single dashboard image."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image

        valid = [p for p in plot_paths if p and os.path.exists(p)]
        if not valid:
            return ""

        cols = 3
        rows = math.ceil(len(valid) / cols)
        imgs = [Image.open(p) for p in valid]
        w, h = imgs[0].size
        canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
        for i, img in enumerate(imgs):
            r, c = divmod(i, cols)
            canvas.paste(img.resize((w, h)), (c * w, r * h))
        dash_path = f"{out_dir}/{model_short}_{base_trait}_dashboard.png"
        canvas.save(dash_path)
        return dash_path
    except Exception:
        return plot_paths[0] if plot_paths else ""


# ── Combined interaction metrics dashboard ────────────────────────────────────

def save_dashboard(
    metrics_csv_path: str,
    model_short: str,
    base_trait: str,
    out_dir: str = "output/plots",
    effrank_csv_path: str = "",
    esva_csv_path: str = "",
    gbc_csv_path: str = "",
    projection_on_residue_csv_path: str = "",
    rho_eff_csv_path: str = "",
    activation_projection_csv_path: str = "",
    build_dashboard: bool = True,
) -> tuple[list[str], str]:
    """Generate metric plots and optionally compile a dashboard image."""
    plot_paths = []
    plot_paths.append(save_projection_norms_plot(metrics_csv_path, model_short, base_trait, out_dir))
    plot_paths.append(save_amplification_plot(metrics_csv_path, model_short, base_trait, out_dir))

    if effrank_csv_path and os.path.exists(effrank_csv_path):
        plot_paths.append(save_effrank_plot(effrank_csv_path, model_short, base_trait, out_dir))

    if esva_csv_path and os.path.exists(esva_csv_path):
        plot_paths.append(save_esva_plot(esva_csv_path, model_short, base_trait, out_dir))

    if gbc_csv_path and os.path.exists(gbc_csv_path):
        plot_paths.append(save_gbc_plot(gbc_csv_path, model_short, base_trait, out_dir))

    if projection_on_residue_csv_path and os.path.exists(projection_on_residue_csv_path):
        plot_paths.append(
            save_projection_on_residue_plot(
                projection_on_residue_csv_path,
                model_short,
                base_trait,
                out_dir,
            )
        )

    if rho_eff_csv_path and os.path.exists(rho_eff_csv_path):
        plot_paths.append(save_rho_eff_plot(rho_eff_csv_path, model_short, base_trait, out_dir))

    if activation_projection_csv_path and os.path.exists(activation_projection_csv_path):
        plot_paths.append(save_activation_projection_plot(activation_projection_csv_path, model_short, base_trait, out_dir))

    dashboard_path = ""
    if build_dashboard:
        dashboard_path = save_plot_grid_dashboard(plot_paths, model_short, base_trait, out_dir)
        if dashboard_path:
            print(f"  Dashboard saved: {dashboard_path}")
    return plot_paths, dashboard_path