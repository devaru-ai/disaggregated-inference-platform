#!/bin/bash
set -euo pipefail

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=8001
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

CONTEXT_LENGTHS=(512 4096 8192)
CONCURRENCIES=(8 32 64 128 256 512)

cleanup() {
    trap - EXIT INT TERM
    
    echo -e "\n Cleaning up SGLang engine..."
    
    if [ -n "${SGLANG_PID:-}" ]; then
        kill -9 $SGLANG_PID 2>/dev/null || true
    fi
    
    pkill -9 -f "python3 -m sglang.launch_server" || true
    echo "✅ Cleanup complete."
}

# Ensure cleanup runs on exit or crash
trap cleanup EXIT INT TERM


echo "Clearing zombie processes..."
pkill -9 -f "python3 -m sglang.launch_server" || true
sleep 2

echo "🚀 Starting SGLang: $MODEL"

# Launch server with stability flags
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --mem-fraction-static 0.7 \
    --disable-cuda-graph \
    > sglang_boot.log 2>&1 &

SGLANG_PID=$!
echo "SGLang PID: $SGLANG_PID"


echo "⏳ Waiting for SGLang..."
TIMEOUT=120

while [ $TIMEOUT -gt 0 ]; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo "✅ SGLang is online."
        break
    fi

    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "❌ SGLang crashed at startup. Check sglang_boot.log:"
        tail -n 20 sglang_boot.log
        exit 1
    fi

    sleep 5
    TIMEOUT=$((TIMEOUT - 5))
done

if [ $TIMEOUT -le 0 ]; then
    echo "❌ Timeout waiting for SGLang"
    exit 1
fi


for CONTEXT in "${CONTEXT_LENGTHS[@]}"; do
    for CONCURRENCY in "${CONCURRENCIES[@]}"; do
        echo "SWEEP: context=$CONTEXT concurrency=$CONCURRENCY"
        python3 benchmarks/load_tester_monolithic.py \
            --url "http://localhost:$PORT" \
            --topology "SGLang_A100_Llama3_8B" \
            --model "$MODEL" \
            --context "$CONTEXT" \
            --concurrency "$CONCURRENCY"
        sleep 2
    done
done

echo "PIPELINE COMPLETE."
