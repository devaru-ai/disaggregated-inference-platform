import os
import sys
import csv
import time
import concurrent.futures

# 1. FORCE physical isolation at the Python runtime level BEFORE torch imports
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoConfig

# =====================================================================
# 🔍 DIAGNOSTIC TRAP: Expose PyTorch's actual logical-to-physical map
# =====================================================================
print("\n" + "=" * 50)
print(f"🔥 PyTorch CUDA Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   Logical cuda:{i} -> Hardware: {torch.cuda.get_device_name(i)}")
print("=" * 50 + "\n")

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/gpu_ipc_disagg"))
)

from kv_arena import KVArena
from p2p_engine import P2PEngine
from worker_prefill import PrefillWorker
from worker_decode import DecodeWorker
from router import DisaggregatedRouter


def generate_dummy_prompt(tokenizer, num_tokens):
    token_ids = [tokenizer.eos_token_id] * num_tokens
    return tokenizer.decode(token_ids)


def run_benchmark():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # Logical mapping:
    # cuda:0 = Prefill
    # cuda:1 = Decode 1
    # cuda:2 = Decode 2
    arena_1 = KVArena("cuda:1", config)
    arena_2 = KVArena("cuda:2", config)

    p2p_1 = P2PEngine("cuda:0", "cuda:1", arena_1)
    p2p_2 = P2PEngine("cuda:0", "cuda:2", arena_2)

    prefill = PrefillWorker(model_id, tokenizer, "cuda:0", p2p_1)

    decode_workers = [
        DecodeWorker(model_id, tokenizer, "cuda:1", arena_1),
        DecodeWorker(model_id, tokenizer, "cuda:2", arena_2),
    ]

    router = DisaggregatedRouter(prefill, decode_workers, [p2p_1, p2p_2])

    # ============================================================
    # Benchmark sweep configuration (FIXED)
    # ============================================================
    concurrencies = [1, 2, 4, 8]
    context_lengths = [128, 256, 512, 1024, 2048]
    num_requests_per_sweep = 8

    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/prototype_results.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Concurrency", "Context", "Throughput", "TPOT", "IPC Transfer"])

    def send_request(prompt):
        start = time.perf_counter()
        result = router.process_request(prompt, 32)
        total = time.perf_counter() - start

        return (
            result.get("ipc_transfer_ms", 0.0),
            result.get("generated_tokens", 1),
            total,
        )

    for ctx in context_lengths:
        prompt = generate_dummy_prompt(tokenizer, ctx)

        for conc in concurrencies:
            transfer_times = []
            tpot_times = []
            total_tokens = 0

            start_sweep = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
                futures = [
                    ex.submit(send_request, prompt)
                    for _ in range(num_requests_per_sweep)
                ]

                for f in concurrent.futures.as_completed(futures):
                    transfer, tokens, runtime = f.result()

                    transfer_times.append(transfer)
                    total_tokens += tokens
                    tpot_times.append((runtime * 1000) / max(1, tokens))

            sweep_time = time.perf_counter() - start_sweep

            throughput = total_tokens / sweep_time if sweep_time > 0 else 0.0
            avg_tpot = sum(tpot_times) / len(tpot_times)
            avg_transfer = sum(transfer_times) / len(transfer_times)

            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow(
                    [conc, ctx, throughput, avg_tpot, avg_transfer]
                )

            print(
                f"[ctx={ctx} conc={conc}] "
                f"tok/s={throughput:.2f} "
                f"tpot={avg_tpot:.2f}ms "
                f"transfer={avg_transfer:.2f}ms"
            )


if __name__ == "__main__":
    run_benchmark()