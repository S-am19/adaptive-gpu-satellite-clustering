"""Visualization and plotting scripts for PCAP benchmark results and satellite cluster maps.

Generates:
1. Quality vs Time convergence curves
2. Throughput scaling (pixels/sec) across algorithms
3. Ablation comparisons (C0, G1, G2, G3, G4, G5)
4. Reconstructed 2D spatial cluster maps
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_quality_vs_time(df: pd.DataFrame, output_path: Path):
    """Plot Final SSE vs Wall-Clock Time across methods."""
    plt.figure(figsize=(10, 6), dpi=150)
    methods = df["method"].unique()

    for method in methods:
        sub = df[df["method"] == method]
        plt.scatter(
            sub["wall_time_s"],
            sub["final_exact_sse"],
            label=method,
            alpha=0.7,
            s=60,
        )

    plt.xlabel("Wall-Clock Time (seconds)", fontsize=12)
    plt.ylabel("Final Exact SSE (lower is better)", fontsize=12)
    plt.title("Clustering Quality vs. Runtime", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Algorithm", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_throughput_bars(df: pd.DataFrame, output_path: Path):
    """Plot average throughput (pixels/sec) per method."""
    plt.figure(figsize=(10, 6), dpi=150)
    summary = df.groupby("method")["throughput_pixels_per_s"].mean().reset_index()

    bars = plt.bar(
        summary["method"],
        summary["throughput_pixels_per_s"],
        color="royalblue",
        alpha=0.85,
        edgecolor="black",
    )
    plt.ylabel("Throughput (pixels / second)", fontsize=12)
    plt.xlabel("Algorithm Method", fontsize=12)
    plt.title("Average Pixel Throughput by Method", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.01 * yval,
            f"{int(yval):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_comparison(df: pd.DataFrame, output_path: Path):
    """Plot multi-metric comparison across ablation variants (RMSE, DBI, Silhouette)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    methods = sorted(df["method"].unique())

    metrics = [
        ("rmse", "RMSE (lower is better)", "forestgreen"),
        ("davies_bouldin_index", "Davies-Bouldin Index (lower is better)", "coral"),
        ("silhouette_score", "Silhouette Score (higher is better)", "mediumpurple"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        grouped = df.groupby("method")[col].mean().reindex(methods)
        ax.bar(methods, grouped, color=color, alpha=0.8, edgecolor="black")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Method", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Algorithmic Ablation Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def render_cluster_map(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    pixel_indices: np.ndarray | None,
    output_path: Path,
    title: str = "Satellite pixel cluster IDs",
):
    """Reconstruct 2D spatial cluster image from 1D labels and valid mask."""
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 2:
        raise ValueError("valid_mask must be a 2D boolean array")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or not len(labels) or np.any(labels < 0):
        raise ValueError("labels must be a nonempty nonnegative integer vector")
    if pixel_indices is not None:
        if (pixel_indices.shape != labels.shape or not np.issubdtype(pixel_indices.dtype, np.integer)
                or np.any(pixel_indices < 0) or np.any(pixel_indices >= valid_mask.size)
                or len(np.unique(pixel_indices)) != len(pixel_indices)
                or not np.all(valid_mask.ravel()[pixel_indices])):
            raise ValueError("pixel_indices must be distinct valid pixel positions matching labels")
    elif len(labels) != valid_mask.sum():
        raise ValueError("labels must match valid pixels")
    h, w = valid_mask.shape
    image_grid = np.full((h, w), -1, dtype=np.int32)

    if pixel_indices is not None:
        flat = image_grid.ravel()
        flat[pixel_indices] = labels
    else:
        image_grid[valid_mask] = labels

    plt.figure(figsize=(8, 8), dpi=200)
    # Masked pixels (-1) displayed in dark gray
    cmap = matplotlib.colormaps["tab20"].resampled(int(labels.max() + 1))
    cmap.set_bad("black")

    im = plt.imshow(np.ma.masked_less(image_grid, 0), cmap=cmap, vmin=-0.5, vmax=int(labels.max()) + 0.5)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Cluster ID", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial cluster map: {output_path}")


def plot_histories(raw_path, output_dir):
    """Plot measured repeat zero for each adaptive case; never average histories."""
    import json
    for line in Path(raw_path).read_text().splitlines():
        row = json.loads(line)
        if row["phase"] != "measured" or row["repeat"] != 0 or row["method"] not in ("G3", "G4", "G5", "G3CUDA", "G4CUDA", "G5CUDA"):
            continue
        h = row["history"]
        fig, axes = plt.subplots(2, 1, sharex=True)
        steps = [v["step"] for v in h]
        axes[0].step(steps, [v["batch_size"] for v in h], where="post")
        axes[0].set_ylabel("Batch observations")
        axes[1].step(steps, [v.get("ann_effort") or 0 for v in h], where="post")
        axes[1].set_ylabel("ANN effort (0=none)")
        axes[1].set_xlabel("Update step")
        fig.suptitle(f'{row["method"]}: {row["implementation"]}, N={row["n"]}, K={row["k"]}, seed={row["seed"]}, repeat=0')
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'history_n{row["n"]}_d{row["d"]}_k{row["k"]}_seed{row["seed"]}_{row["method"]}.png')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PCAP Results Plotter")
    parser.add_argument("--raw-jsonl", type=Path, help="Optional raw log for per-case adaptation histories")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark CSV results file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Choose a new empty figure directory; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment rows from {csv_path}")

    # Never pool sizes, dimensions, K or seeds into one misleading comparison.
    group_columns = [c for c in ("dataset_id", "case_id", "session_id", "n", "d", "k", "seed") if c in df]
    if "implementation" in df:
        df = df.copy()
        df["method"] = df["method"] + " (" + df["implementation"] + ")"
    groups = df.groupby(group_columns) if group_columns else [((), df)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        suffix = "_".join(f"{c}{str(v)[:12]}" for c, v in zip(group_columns, key)) or "results"
        plot_quality_vs_time(group, out_dir / f"quality_vs_time_{suffix}.png")
        plot_throughput_bars(group, out_dir / f"throughput_comparison_{suffix}.png")
        plot_ablation_comparison(group, out_dir / f"ablation_comparison_{suffix}.png")
    if args.raw_jsonl:
        plot_histories(args.raw_jsonl, out_dir)
    print(f"All benchmark figures generated in {out_dir}")


if __name__ == "__main__":
    main()
