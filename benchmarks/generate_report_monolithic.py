import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name))


def extract_model_name(topology: str) -> str:
    parts = topology.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else topology


def get_gpu_util_from_trace(trace_path: str) -> float:
    if not os.path.exists(trace_path):
        return 0.0

    df = pd.read_csv(trace_path)
    df.columns = df.columns.str.lower().str.strip()

    gpu_cols = [c for c in df.columns if "gpu" in c and "util" in c]
    if not gpu_cols:
        return 0.0

    col = gpu_cols[0]
    series = df[col]

    util = pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    ).fillna(0.0)

    return float(util.quantile(0.9)) if len(util) else 0.0


def detect_knee_point(x, y):
    x = np.array(x)
    y = np.array(y)

    if len(x) < 3:
        return None

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

    curvature = np.abs(np.diff(y_norm, 2))
    if len(curvature) == 0:
        return None

    idx = np.argmax(curvature) + 1
    return int(x[idx])


def classify_regime(concurrency, knee_x):
    if knee_x is None:
        return "Unbounded"
    if concurrency < knee_x:
        return "Linear"
    elif concurrency == knee_x:
        return "Saturation"
    return "Degraded"


def compute_pareto(df):
    pts = df[["Throughput_tps", "TPOT_ms"]].values
    keep = np.ones(len(pts), dtype=bool)

    for i, (t1, l1) in enumerate(pts):
        for j, (t2, l2) in enumerate(pts):
            if (t2 >= t1 and l2 <= l1) and (t2 > t1 or l2 < l1):
                keep[i] = False
                break

    return df[keep]



def generate_report(csv_path, logs_dir, out_dir, topology_filter=None):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Context": "Context_Length",
        "TPOT (ms/tok)": "TPOT_ms",
        "Throughput (tok/s)": "Throughput_tps",
        "Peak KV CACHE(%)": "Peak_KV_pct",
    })

    gpu_utils = []

    for _, row in df.iterrows():
        model = extract_model_name(row["Topology"])
        ctx = int(row["Context_Length"])
        conc = int(row["Concurrency"])

        trace_file = f"{model}_ctx{ctx}_conc{conc}_gpu_trace.csv"
        trace_path = os.path.join(logs_dir, trace_file)

        gpu_utils.append(get_gpu_util_from_trace(trace_path))

    df["GPU_Util_pct"] = gpu_utils

    
    CREAM = "#f4f1ee"
    TITLE = "#2f243a"

    sns.set_theme(style="whitegrid")

    
    for topo in df["Topology"].unique():
        topo_df = df[df["Topology"] == topo].copy()
        topo_df = topo_df.sort_values("Concurrency")

        n_ctx = topo_df["Context_Length"].nunique()
        base_palette = ["#d8c1b8", "#b07aa1", "#3b2f4a", "#8a6d85", "#5c4d66"]
        palette = base_palette[:n_ctx]

        topo_df["Regime"] = topo_df["Concurrency"].apply(
            lambda x: classify_regime(x, None)
        )

        pareto_df = compute_pareto(topo_df)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        
        fig.suptitle(
            f"Hardware Saturation: {topo}",
            fontsize=18,
            fontweight="bold",
            color=TITLE
        )

        
        sns.lineplot(
            data=topo_df,
            x="Concurrency",
            y="Throughput_tps",
            hue="Context_Length",
            marker="o",
            palette=palette,
            ax=axes[0],
            errorbar=None
        )
        axes[0].set_title("Throughput (tokens/s)")

        
        sns.lineplot(
            data=topo_df,
            x="Concurrency",
            y="TPOT_ms",
            hue="Context_Length",
            marker="o",
            palette=palette,
            ax=axes[1],
            errorbar=None
        )
        axes[1].set_title("TPOT (Latency)")

        
        sns.barplot(
            data=topo_df,
            x="Concurrency",
            y="Peak_KV_pct",
            hue="Context_Length",
            palette=palette,
            ax=axes[2],
            errorbar=None
        )
        axes[2].set_title("KV Cache Pressure")

        
        sns.barplot(
            data=topo_df,
            x="Concurrency",
            y="GPU_Util_pct",
            hue="Context_Length",
            palette=palette,
            ax=axes[3],
            errorbar=None
        )
        axes[3].set_title("GPU Utilization (P90)")

        
        for ax in axes:
            ax.set_facecolor(CREAM)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color("#b8b0aa")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out = os.path.join(out_dir, f"report_{sanitize_filename(topo)}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Generated: {out}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    logs_dir = os.path.join(project_root, "logs")
    out_dir = os.path.join(project_root, "plots")
    csv_path = os.path.join(logs_dir, "monolithic_benchmark_results.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=csv_path)
    parser.add_argument("--logs-dir", default=logs_dir)
    parser.add_argument("--out-dir", default=out_dir)
    parser.add_argument("--topology-filter", default=None, type=str) 

    args = parser.parse_args()

    generate_report(args.csv, args.logs_dir, args.out_dir, args.topology_filter)