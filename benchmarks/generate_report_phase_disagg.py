import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name))


def get_temporal_gpu_util(trace_path="logs/gpu_hardware_trace.csv", num_sweeps=0, gpu_id=0):
    if not os.path.exists(trace_path) or num_sweeps <= 0:
        return [0.0] * num_sweeps

    df = pd.read_csv(trace_path)
    df.columns = df.columns.str.strip()

    gpu_cols = [c for c in df.columns if "utilization.gpu" in c.lower()]
    if not gpu_cols or len(gpu_cols) <= gpu_id:
        return [0.0] * num_sweeps

    gpu_series = df[gpu_cols[gpu_id]]
    util = pd.to_numeric(
        gpu_series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

    chunk_size = max(1, len(util) // num_sweeps)
    out = []

    for i in range(num_sweeps):
        start = i * chunk_size
        end = len(util) if i == num_sweeps - 1 else min(start + chunk_size, len(util))
        window = util.iloc[start:end]
        out.append(round(window.mean(), 2) if len(window) else 0.0)

    return out


def generate_report(csv_path="logs/phase_disagg_results.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    CREAM, TITLE = "#f4f1ee", "#2f243a"

    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": CREAM,
        "figure.facecolor": CREAM
    })

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Normalize column names
    if "Throughput" in df.columns:
        df = df.rename(columns={
            "Throughput": "Throughput_tps",
            "TPOT": "TPOT_ms",
            "IPC Transfer": "IPC_Transfer_ms"
        })

    # Ensure required columns exist
    if "Context" not in df.columns:
        df["Context"] = "default"

    topologies = df["Topology"].unique() if "Topology" in df.columns else ["DIP_Prototype"]

    for topo in topologies:
        topo_df = df[df["Topology"] == topo].copy() if "Topology" in df.columns else df.copy()

        n = len(topo_df)

        topo_df["Prefill_Util_pct"] = get_temporal_gpu_util(num_sweeps=n, gpu_id=0)

        dec1 = get_temporal_gpu_util(num_sweeps=n, gpu_id=1)
        dec2 = get_temporal_gpu_util(num_sweeps=n, gpu_id=2)

        topo_df["Decode_Util_pct"] = [
            (a + b) / 2 for a, b in zip(dec1, dec2)
        ] if n > 0 else []

        # Dynamic palette (fix seaborn warning)
        palette = sns.color_palette("magma", n_colors=max(3, topo_df["Context"].nunique()))

        fig, axes = plt.subplots(3, 2, figsize=(16, 15), facecolor=CREAM)
        axes = axes.flatten()

        fig.suptitle(
            f"Phase Disaggregation (DIP): {topo}",
            fontsize=18,
            fontweight="bold",
            color=TITLE,
        )

        for ax in axes:
            ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color("#b8b0aa")

        
        sns.lineplot(
            data=topo_df,
            x="Concurrency",
            y="Throughput_tps",
            hue="Context",
            marker="o",
            palette=palette,
            ax=axes[0],
        )
        axes[0].set_title("System Throughput (tokens/sec)")

        
        sns.lineplot(
            data=topo_df,
            x="Concurrency",
            y="TPOT_ms",
            marker="o",
            ax=axes[1],
        )
        axes[1].set_title("Decode Latency (TPOT)")

        
        sns.barplot(
            data=topo_df,
            x="Concurrency",
            y="Prefill_Util_pct",
            ax=axes[2],
        )
        axes[2].set_title("Prefill GPU Utilization")

        
        sns.barplot(
            data=topo_df,
            x="Concurrency",
            y="Decode_Util_pct",
            ax=axes[3],
        )
        axes[3].set_title("Decode GPU Utilization")

        
        if "IPC_Transfer_ms" in topo_df.columns:
            sns.lineplot(
                data=topo_df,
                x="Concurrency",
                y="IPC_Transfer_ms",
                marker="x",
                ax=axes[4],
            )
            axes[4].set_title("NVLink Transfer Latency")
        else:
            axes[4].axis("off")

        axes[5].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = f"logs/report_{sanitize_filename(topo)}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    generate_report()