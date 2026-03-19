#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt


def build_lane_boundaries(num_lanes: int, lane_width: float):
    left = -0.5 * num_lanes * lane_width
    return [left + i * lane_width for i in range(num_lanes + 1)]


def main():
    parser = argparse.ArgumentParser(description="Plot sprint track top-view layout")
    parser.add_argument("--num_lanes", type=int, default=6)
    parser.add_argument("--lane_width", type=float, default=1.25)
    parser.add_argument("--lane_length", type=float, default=18.0)
    parser.add_argument("--out", type=str, default="/home/niumu/unitree_ws/src/unitree_rl_gym/humanoid_sprint/reports/track_layout_preview.png")
    args = parser.parse_args()

    boundaries = build_lane_boundaries(args.num_lanes, args.lane_width)
    half_length = 0.5 * args.lane_length

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, y in enumerate(boundaries):
        color = "white" if i in (0, len(boundaries) - 1) else "gold"
        lw = 3.0 if i in (0, len(boundaries) - 1) else 1.5
        ax.plot([-half_length, half_length], [y, y], color=color, linewidth=lw)

    for lane_id in range(args.num_lanes):
        center_y = 0.5 * (boundaries[lane_id] + boundaries[lane_id + 1])
        ax.text(-half_length + 0.4, center_y, f"Lane {lane_id + 1}", va="center", fontsize=9, color="cyan")

    ax.set_title(f"Sprint Track Layout: {args.num_lanes} lanes, width={args.lane_width}m, length={args.lane_length}m")
    ax.set_xlabel("Forward X (m)")
    ax.set_ylabel("Lateral Y (m)")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#444444", linestyle="--", linewidth=0.5)

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved preview: {args.out}")


if __name__ == "__main__":
    main()
