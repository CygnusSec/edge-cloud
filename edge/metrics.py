"""
Compute experiment quality metrics and generate a comparison summary.

Aggregates results from all three scenarios and saves to results/metrics_summary.csv.

Usage:
    python metrics.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

SCENARIO_FILES = {
    "cloud_only": "cloud_only.csv",
    "edge_only": "edge_only.csv",
    "edge_cloud": "edge_cloud.csv",
}

DIFFICULTIES = ["easy", "medium", "hard"]


def load_scenario(results_dir: str, scenario: str) -> pd.DataFrame | None:
    """Load a scenario CSV, skipping the SUMMARY row."""
    path = os.path.join(results_dir, SCENARIO_FILES[scenario])
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8")
    df = df[~df.iloc[:, 0].astype(str).str.startswith("SUMMARY")]
    for col in ["latency_ms", "uploaded_bytes", "confidence", "object_count",
                "cpu_percent", "ram_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df if not df.empty else None


def compute_metrics_summary(results_dir: str) -> pd.DataFrame:
    """
    Compute per-scenario, per-difficulty summary metrics.

    Returns a DataFrame with columns:
        scenario, difficulty, n_images, avg_confidence, avg_object_count,
        avg_latency_ms, total_uploaded_bytes, offload_ratio_pct,
        avg_cpu_percent, avg_ram_mb
    """
    rows = []

    for scenario in SCENARIO_FILES:
        df = load_scenario(results_dir, scenario)
        if df is None:
            continue

        rows.append(_compute_row(df, scenario, "all"))

        if "difficulty" in df.columns:
            for diff in DIFFICULTIES:
                sub = df[df["difficulty"] == diff]
                if not sub.empty:
                    rows.append(_compute_row(sub, scenario, diff))

    return pd.DataFrame(rows)


def _compute_row(df: pd.DataFrame, scenario: str, difficulty: str) -> dict:
    """Compute summary metrics for a subset of data."""
    total = len(df)
    offload_ratio = 0.0
    if "offloaded" in df.columns:
        offload_ratio = (
            (df["offloaded"].astype(str).str.lower() == "true").sum() / total * 100
            if total > 0 else 0.0
        )

    return {
        "scenario": scenario,
        "difficulty": difficulty,
        "n_images": total,
        "avg_confidence": round(df["confidence"].mean(), 4) if "confidence" in df.columns else None,
        "avg_object_count": round(df["object_count"].mean(), 2) if "object_count" in df.columns else None,
        "avg_latency_ms": round(df["latency_ms"].mean(), 2) if "latency_ms" in df.columns else None,
        "total_uploaded_bytes": int(df["uploaded_bytes"].sum()) if "uploaded_bytes" in df.columns else 0,
        "offload_ratio_pct": round(offload_ratio, 2),
        "avg_cpu_percent": round(df["cpu_percent"].mean(), 2) if "cpu_percent" in df.columns else None,
        "avg_ram_mb": round(df["ram_mb"].mean(), 2) if "ram_mb" in df.columns else None,
    }


def print_comparison_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted comparison table to console."""
    overall = summary_df[summary_df["difficulty"] == "all"]
    if overall.empty:
        print("No data available to display.")
        return

    print("\n" + "=" * 80)
    print("THREE-MODEL COMPARISON TABLE")
    print("=" * 80)
    header = (
        f"{'Scenario':<15} {'Avg Latency':>12} {'Total Upload':>14} "
        f"{'Offload%':>10} {'Avg Confidence':>15} {'Throughput':>12}"
    )
    print(header)
    print("-" * 80)

    for _, row in overall.iterrows():
        throughput = (
            row["n_images"] / max(row["avg_latency_ms"] * row["n_images"] / 1000, 0.001)
            if row["avg_latency_ms"] else 0
        )
        print(
            f"{row['scenario']:<15} "
            f"{row['avg_latency_ms']:>10.2f}ms "
            f"{row['total_uploaded_bytes']:>12,}B "
            f"{row['offload_ratio_pct']:>9.1f}% "
            f"{row['avg_confidence']:>14.4f} "
            f"{throughput:>10.4f}/s"
        )

    print("=" * 80)

    # Bandwidth savings
    cloud_bw = overall[overall["scenario"] == "cloud_only"]["total_uploaded_bytes"].values
    ec_bw = overall[overall["scenario"] == "edge_cloud"]["total_uploaded_bytes"].values
    if len(cloud_bw) > 0 and len(ec_bw) > 0 and cloud_bw[0] > 0:
        savings = (cloud_bw[0] - ec_bw[0]) / cloud_bw[0] * 100
        print(f"\n✅ Bandwidth savings (edge-cloud vs cloud-only): {savings:.1f}%")

    # Confidence improvement on hard images
    eo_hard = summary_df[
        (summary_df["scenario"] == "edge_only") & (summary_df["difficulty"] == "hard")
    ]["avg_confidence"].values
    ec_hard = summary_df[
        (summary_df["scenario"] == "edge_cloud") & (summary_df["difficulty"] == "hard")
    ]["avg_confidence"].values
    if len(eo_hard) > 0 and len(ec_hard) > 0 and eo_hard[0] > 0:
        improvement = (ec_hard[0] - eo_hard[0]) / eo_hard[0] * 100
        print(f"✅ Confidence improvement on hard images (edge-cloud vs edge-only): {improvement:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute experiment metrics summary")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", "results"),
        help="Path to results directory (default: results/)",
    )
    args = parser.parse_args()

    summary_df = compute_metrics_summary(args.results_dir)

    if summary_df.empty:
        print(
            "No data found. Please run the experiment scenarios first:\n"
            "  python run_cloud_only.py\n"
            "  python run_edge_only.py\n"
            "  python run_edge_cloud.py"
        )
        sys.exit(1)

    output_path = os.path.join(args.results_dir, "metrics_summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Metrics summary saved to: {output_path}")

    print_comparison_table(summary_df)


if __name__ == "__main__":
    main()
