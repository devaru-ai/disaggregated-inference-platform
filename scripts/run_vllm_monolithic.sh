#!/bin/bash
set -euo pipefail

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0


MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen1.5-14B-Chat"
)

CONTEXT_LENGTHS=(512 4096 8192)
CONCURRENCIES=(8 32 64 128 256 512)

PORT=8001
MAX_HEALTH_RETRIES=60
HEALTH_RETRY_INTERVAL=5

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

VLLM_PID=""
TELEMETRY_PID=""

cleanup() {
    echo
    echo "Triggering cleanup..."

    if [[ -n "${TELEMETRY_PID}" ]]; then
        kill "${TELEMETRY_PID}" 2>/dev/null || true
        wait "${TELEMETRY_PID}" 2>/dev/null || true
    fi

    if [[ -n "${VLLM_PID}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi

    echo "✅ Cleanup complete."
}

trap cleanup EXIT INT TERM

echo "Clearing stale vLLM processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

echo "🚀 Starting scaling benchmark sweeps..."

for MODEL in "${MODELS[@]}"; do

    MODEL_BASENAME=$(basename "${MODEL}")

    echo "BOOTING ENGINE FOR MODEL: ${MODEL}"

    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --gpu-memory-utilization 0.95 \
        --max-model-len 16384 \
        --device cuda \
        --dtype half \
        --enforce-eager &
    
    VLLM_PID=$!

    echo "⏳ Waiting for vLLM health check..."

    RETRIES=0

    until curl -sf "http://localhost:${PORT}/health" > /dev/null; do

        sleep "${HEALTH_RETRY_INTERVAL}"
        RETRIES=$((RETRIES + 1))

        # Detect startup crash
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "❌ vLLM crashed during startup."
            exit 1
        fi

        # Timeout protection
        if [[ "${RETRIES}" -ge "${MAX_HEALTH_RETRIES}" ]]; then
            echo "❌ vLLM failed health check timeout."
            exit 1
        fi
    done

    echo "✅ vLLM is online."

    for CONTEXT in "${CONTEXT_LENGTHS[@]}"; do
        for CONCURRENCY in "${CONCURRENCIES[@]}"; do

            # Detect runtime crash before benchmark
            if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                echo "❌ vLLM engine died mid-sweep."
                exit 1
            fi

            TOPOLOGY_TAG="Monolithic_A100_${MODEL_BASENAME}"

            RUN_TAG="${MODEL_BASENAME}_ctx${CONTEXT}_conc${CONCURRENCY}"

            TRACE_FILE="${LOG_DIR}/${RUN_TAG}_gpu_trace.csv"

            echo
            echo "SWEEP START"
            echo "Topology:     ${TOPOLOGY_TAG}"
            echo "Context:      ${CONTEXT}"
            echo "Concurrency:  ${CONCURRENCY}"
            echo "Telemetry:    ${TRACE_FILE}"
            echo

            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,temperature.gpu \
                --format=csv \
                -l 1 \
                > "${TRACE_FILE}" &

            TELEMETRY_PID=$!

            
            python3 benchmarks/load_tester_monolithic.py \
                --url "http://localhost:${PORT}" \
                --topology "${TOPOLOGY_TAG}" \
                --model "${MODEL}" \
                --context "${CONTEXT}" \
                --concurrency "${CONCURRENCY}"

            kill "${TELEMETRY_PID}" 2>/dev/null || true
            wait "${TELEMETRY_PID}" 2>/dev/null || true
            TELEMETRY_PID=""

            if [[ "${CONTEXT}" -ge 8192 || "${CONCURRENCY}" -ge 128 ]]; then
                echo "⏳ Extended cooldown for high-pressure sweep..."
                sleep 10
            else
                sleep 2
            fi
        done
    done

    echo
    echo "Shutting down vLLM engine..."

    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    VLLM_PID=""

    echo "✅ Model sweep complete."

    sleep 10
done

echo
echo "ALL BENCHMARK SWEEPS COMPLETE."

