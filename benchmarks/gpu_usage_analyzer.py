import sys
import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GPU-Analyzer - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Analyzer")


def analyze_and_update(trace_path, results_path):
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    logger.info(f"Loading trace CSV: {trace_path}")

    trace_df = pd.read_csv(trace_path)

    # Clean column names
    trace_df.columns = [c.strip() for c in trace_df.columns]

    gpu_cols = [
        c for c in trace_df.columns
        if 'utilization.gpu' in c.lower()
    ]

    if not gpu_cols:
        raise ValueError(
            f"Could not find GPU utilization column.\n"
            f"Available columns:\n{trace_df.columns.tolist()}"
        )

    logger.info(f"Using GPU utilization column: {gpu_cols[0]}")

    # Force Series
    gpu_series = trace_df[gpu_cols[0]]

    trace_df['util'] = pd.to_numeric(
        gpu_series.astype(str).str.replace('%', '', regex=False),
        errors='coerce'
    )

    # Drop invalid rows
    trace_df = trace_df.dropna(subset=['util'])

    if trace_df.empty:
        raise ValueError("No valid GPU utilization samples found.")

    logger.info(f"Loaded {len(trace_df)} utilization samples")

    logger.info(f"Loading benchmark results CSV: {results_path}")

    results_df = pd.read_csv(results_path)

    num_sweeps = len(results_df)

    if num_sweeps == 0:
        logger.warning("Results CSV is empty. Nothing to update.")
        return

    logger.info(f"Found {num_sweeps} benchmark sweeps")

    chunk_size = max(1, len(trace_df) // num_sweeps)

    logger.info(
        f"Mapping {len(trace_df)} trace samples "
        f"across {num_sweeps} sweeps "
        f"(chunk size = {chunk_size})"
    )

    # Ensure output column exists
    if 'Peak SM Util (%)' not in results_df.columns:
        results_df['Peak SM Util (%)'] = None

    for i in range(num_sweeps):

        start_idx = i * chunk_size

        # Last sweep gets remainder
        if i == num_sweeps - 1:
            end_idx = len(trace_df)
        else:
            end_idx = min(start_idx + chunk_size, len(trace_df))

        window = trace_df['util'].iloc[start_idx:end_idx]

        if len(window) == 0:
            avg_util = 0.0
        else:
            avg_util = round(window.mean(), 2)

        results_df.at[i, 'Peak SM Util (%)'] = avg_util

        logger.info(
            f"Sweep {i}: samples [{start_idx}:{end_idx}] "
            f"-> avg util = {avg_util}%"
        )

    results_df.to_csv(results_path, index=False)

    logger.info("✅ Successfully updated benchmark results CSV")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python gpu_usage_analyzer.py <trace.csv> <results.csv>"
        )
        sys.exit(1)

    trace_csv = sys.argv[1]
    results_csv = sys.argv[2]

    analyze_and_update(trace_csv, results_csv)