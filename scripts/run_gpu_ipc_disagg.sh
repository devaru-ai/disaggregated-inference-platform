#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,4

# Define the models to benchmark
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen1.5-14B-Chat"

)

mkdir -p "$ROOT_DIR/logs"

for MODEL in "${MODELS[@]}"; do
    echo "🚀 Running Custom IPC Prototype for: $MODEL"
    export MODEL_ID="$MODEL"
    echo "Executing benchmark sweeps..."
    python3 "$ROOT_DIR/benchmarks/load_tester_disagg.py"
    echo "📈 Generating report..."
    python3 "$ROOT_DIR/benchmarks/generate_report_phase_disagg.py"
    echo "✅ Prototype test complete for $MODEL."    
    # Cooldown to ensure VRAM is fully reclaimed by PyTorch
    sleep 10
done
echo "PHASE DISAGGREGATED BENCHMARKS COMPLETED"
