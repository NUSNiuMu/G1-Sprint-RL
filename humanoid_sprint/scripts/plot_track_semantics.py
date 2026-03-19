#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def build_track(num_lanes: int, lane_width: float, lane_length: float):
    total_width = num_lanes * lane_width
    left = -0.5 * total_width
    boundaries = [left + i * lane_width for i in range(num_lanes + 1)]
    return boundaries, lane_length


def build_semantic_grid(num_lanes: int, lane_width: float, lane_length: float, boundary_tol: float, nx: int, ny: int):
    boundaries, lane_length = build_track(num_lanes, lane_width, lane_length)
    x = np.linspace(-0.5 * lane_length * 1.1, 0.5 * lane_length * 1.1, nx)
    y = np.linspace(boundaries[0] - lane_width, boundaries[-1] + lane_width, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    sem = np.zeros_like(xx, dtype=np.int32)  # 0 non-track
    on_track = (np.abs(xx) <= 0.5 * lane_length) & (yy >= boundaries[0]) & (yy <= boundaries[-1])
    sem[on_track] = 1  # lane interior

    boundary_mask = np.zeros_like(on_track)
    for yb in boundaries:
        boundary_mask |= np.abs(yy - yb) <= boundary_tol
    boundary_mask &= on_track
    sem[boundary_mask] = 2  # boundary

    return x, y, sem, boundaries


def main():
    parser = argparse.ArgumentParser(description="Plot track semantic map")
    parser.add_argument("--num_lanes", type=int, default=24)
    parser.add_argument("--lane_width", type=float, default=1.25)
    parser.add_argument("--lane_length", type=float, default=12.0)
    parser.add_argument("--boundary_tol", type=float, default=0.04)
    parser.add_argument("--nx", type=int, default=600)
    parser.add_argument("--ny", type=int, default=600)
    parser.add_argument("--out", type=str, default="/home/niumu/unitree_ws/src/unitree_rl_gym/humanoid_sprint/reports/track_semantic_preview.png")
    args = parser.parse_args()

    x, y, sem, boundaries = build_semantic_grid(
        args.num_lanes,
        args.lane_width,
        args.lane_length,
        args.boundary_tol,
        args.nx,
        args.ny,
    )

    cmap = ListedColormap(["#2e2e2e", "#1d4ed8", "#facc15"])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(
        sem,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=2,
    )
    ax.set_title(
        f"Track Semantics Preview: lanes={args.num_lanes}, width={args.lane_width}m, length={args.lane_length}m"
    )
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Lateral Y (m)")
    ax.grid(color="#555555", linestyle="--", linewidth=0.4)

    for i in range(len(boundaries) - 1):
        y_center = 0.5 * (boundaries[i] + boundaries[i + 1])
        ax.text(x.min() + 0.2, y_center, f"L{i+1}", va="center", fontsize=7, color="white")

    legend_txt = "语义ID: 0=非跑道, 1=跑道内部, 2=边界/分道线"
    ax.text(0.01, 0.01, legend_txt, transform=ax.transAxes, fontsize=10, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=4))

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved semantic preview: {args.out}")


if __name__ == "__main__":
    main()
