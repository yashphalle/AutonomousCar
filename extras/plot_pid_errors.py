import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REASON_COLORS = {
    "red_light_stop":    "#FFCCCC",
    "yellow_stop":       "#FFF3CC",
    "yellow_continue":   "#FFFACD",
    "junction_speed_cap":"#E8D5F5",
    "speed_limit":       "#CCE5FF",
    "lead_vehicle":      "#FFE5CC",
    "emergency_brake":   "#FF9999",
    "cruise":            "#D5F5E3",
}

REASON_EDGE = {
    "red_light_stop":    "#E74C3C",
    "yellow_stop":       "#F39C12",
    "yellow_continue":   "#F1C40F",
    "junction_speed_cap":"#9B59B6",
    "speed_limit":       "#2980B9",
    "lead_vehicle":      "#E67E22",
    "emergency_brake":   "#C0392B",
    "cruise":            "#27AE60",
}


def latest_log() -> str:
    logs = [p for p in glob.glob("logs/run_*.csv") if os.path.getsize(p) > 500]
    if not logs:
        sys.exit("No log files found in logs/")
    return max(logs, key=os.path.getmtime)


def load(path: str, start: int | None, end: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame" not in df.columns:
        sys.exit(f"No 'frame' column in {path}")

    if df["planner_reason"].dtype == object:
        pass
    elif "planner_reason.1" in df.columns and df["planner_reason.1"].notna().any():
        df["planner_reason"] = df["planner_reason.1"]
    else:
        df["planner_reason"] = "cruise"

    if start is not None:
        df = df[df["frame"] >= start]
    if end is not None:
        df = df[df["frame"] <= end]

    df = df.reset_index(drop=True)
    if df.empty:
        sys.exit("No rows in the selected frame window.")
    return df


def add_reason_spans(ax, df: pd.DataFrame) -> None:
    frames = df["frame"].values
    reasons = df["planner_reason"].values
    i = 0
    while i < len(reasons):
        r = reasons[i]
        j = i
        while j < len(reasons) and reasons[j] == r:
            j += 1
        color = REASON_COLORS.get(r, "#EEEEEE")
        ax.axvspan(frames[i], frames[j - 1], alpha=0.35, color=color, linewidth=0)
        i = j


def style_ax(ax):
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#DDDDDD", linewidth=0.6, linestyle="--", zorder=0)
    ax.tick_params(colors="#333333", labelsize=9)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")


def plot_longitudinal(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    style_ax(ax)
    add_reason_spans(ax, df)

    t = df["frame"].values
    ax.plot(t, df["target_speed"].to_numpy(), color="#9B59B6", linewidth=1.5,
            linestyle="--", label="Target speed (m/s)", zorder=3)
    ax.plot(t, df["ego_speed"].to_numpy(),    color="#27AE60", linewidth=2.0,
            label="Actual speed (m/s)", zorder=4)

    seen = df["planner_reason"].unique()
    handles, labels = ax.get_legend_handles_labels()
    region_patches = [
        mpatches.Patch(facecolor=REASON_COLORS.get(r, "#EEE"),
                       edgecolor=REASON_EDGE.get(r, "#AAA"),
                       linewidth=0.8, label=r.replace("_", " "))
        for r in seen if r in REASON_COLORS
    ]
    ax.legend(handles=handles + region_patches,
              labels=labels + [p.get_label() for p in region_patches],
              loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_ylabel("Speed (m/s)", fontsize=10)
    ax.set_xlabel("Frame (@ 20 Hz)", fontsize=10)
    ax.set_title("Longitudinal PID — Actual vs Target Speed", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


def plot_lateral(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    style_ax(ax)

    t = df["frame"].values
    heading_err_deg = np.degrees(df["heading_error"].to_numpy())

    ax.axhline(0, color="#E74C3C", linewidth=1.8, linestyle="--",
               label="Target: 0°", zorder=3)
    ax.fill_between(t, heading_err_deg, 0,
                    where=heading_err_deg >= 0, alpha=0.2, color="#3498DB", zorder=2)
    ax.fill_between(t, heading_err_deg, 0,
                    where=heading_err_deg <= 0, alpha=0.2, color="#E67E22", zorder=2)
    ax.plot(t, heading_err_deg, color="#2C3E50", linewidth=1.8,
            label="Heading error (°)", zorder=4)

    ax.set_ylabel("Heading error (°)", fontsize=10)
    ax.set_xlabel("Frame (@ 20 Hz)", fontsize=10)
    ax.set_title("Lateral PID — Heading Error", fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",       default=None)
    ap.add_argument("--start",     type=int, default=110, help="Longitudinal start frame")
    ap.add_argument("--end",       type=int, default=392, help="Longitudinal end frame")
    ap.add_argument("--lat-start", type=int, default=1,   help="Lateral start frame")
    ap.add_argument("--lat-end",   type=int, default=140, help="Lateral end frame")
    ap.add_argument("--out-dir",   default="eval/results/baseline_gt_20260509_184806",
                    help="Directory to save plots")
    args = ap.parse_args()

    log_path = args.log or latest_log()
    print(f"Using log: {log_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    df_lon = load(log_path, args.start, args.end)
    df_lat = load(log_path, args.lat_start, args.lat_end)
    print(f"Longitudinal: frames {df_lon['frame'].iloc[0]}–{df_lon['frame'].iloc[-1]}")
    print(f"Lateral:      frames {df_lat['frame'].iloc[0]}–{df_lat['frame'].iloc[-1]}")

    plot_longitudinal(df_lon, os.path.join(args.out_dir, "pid_longitudinal.png"))
    plot_lateral(df_lat,      os.path.join(args.out_dir, "pid_lateral.png"))


if __name__ == "__main__":
    main()
