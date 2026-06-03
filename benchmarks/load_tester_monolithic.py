import asyncio
import aiohttp
import time
import numpy as np
import logging
import json
import csv
import os
import argparse

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Load-Tester")


class BenchmarkSuite:
    def __init__(
        self,
        target_url="http://localhost:8001",
        engine="vllm",
        topology="Monolithic_A100",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        context_len=512
    ):
        self.target_url = target_url
        self.engine = engine.lower()
        self.topology = topology
        self.model_name = model

        self.api_url = f"{target_url}/v1/completions"
        self.metrics_url = f"{target_url}/metrics"

        self.target_input_tokens = context_len
        self.max_output_tokens = 128

        self.prompt_base = (
            "Explain the architectural differences between monolithic and disaggregated LLM inference."
        )

        try:
            logger.info(f"Loading tokenizer: {model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Tokenizer load failed, falling back to heuristic mode: {e}")
            self.tokenizer = None

        self.prompt = self._build_prompt()

        self.peak_kv_cache_recorded = 0.0

    def _build_prompt(self):
        safe_budget = self.target_input_tokens - self.max_output_tokens - 256
        safe_budget = max(256, safe_budget)

        # fallback mode
        if self.tokenizer is None:
            repeat = max(1, safe_budget // 20)
            return (self.prompt_base + " ") * repeat

        base_ids = self.tokenizer.encode(
            self.prompt_base,
            add_special_tokens=False
        )

        if len(base_ids) == 0:
            raise ValueError("Tokenizer returned empty encoding")

        repeats = safe_budget // len(base_ids)
        remainder = safe_budget % len(base_ids)

        token_ids = (base_ids * repeats) + base_ids[:remainder]

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    async def monitor_vram(self):
        self.peak_kv_cache_recorded = 0.0

        metric_target = {
            "vllm": "gpu_cache_usage_perc",
            "sglang": "kv_cache",
            "trt": "kv_cache"
        }.get(self.engine, "gpu_cache_usage_perc")

        try:
            async with aiohttp.ClientSession() as session:
                while True:
                    try:
                        async with session.get(self.metrics_url) as response:
                            if response.status != 200:
                                await asyncio.sleep(0.5)
                                continue

                            text = await response.text()

                            for line in text.splitlines():
                                if metric_target in line:
                                    try:
                                        usage = float(line.split()[-1])
                                        usage = usage * 100 if usage <= 1 else usage
                                        self.peak_kv_cache_recorded = max(
                                            self.peak_kv_cache_recorded,
                                            usage
                                        )
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            return

    async def simulate_user(self, session, request_id):
        payload = {
            "model": self.model_name,
            "prompt": self.prompt,
            "max_tokens": self.max_output_tokens,
            "temperature": 0.0,
            "stream": True
        }

        start_time = time.perf_counter()
        first_token_time = None
        generated_text = ""

        try:
            async with session.post(
                self.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:

                if response.status != 200:
                    return None

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()

                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    chunk = choices[0].get("text", "")
                    if not chunk:
                        continue

                    generated_text += chunk

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

        except Exception:
            return None

        end_time = time.perf_counter()

        ttft_ms = (
            (first_token_time - start_time) * 1000
            if first_token_time else None
        )

        total_time = end_time - start_time

        if self.tokenizer:
            try:
                out_tokens = len(self.tokenizer.encode(
                    generated_text,
                    add_special_tokens=False
                ))
            except Exception:
                out_tokens = max(1, len(generated_text) // 4)
        else:
            out_tokens = max(1, len(generated_text) // 4)

        tpot_ms = (
            ((total_time - (ttft_ms / 1000)) / (out_tokens - 1)) * 1000
            if ttft_ms and out_tokens > 1
            else 0
        )

        return ttft_ms, tpot_ms, out_tokens

    async def run_warmup(self):
        logger.info("--- Warmup ---")

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[self.simulate_user(session, i) for i in range(4)]
            )

        await asyncio.sleep(2)

    async def run_concurrency_sweep(self, concurrency):
        logger.info(f"--- Sweep {concurrency} ---")

        vram_task = asyncio.create_task(self.monitor_vram())
        start = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self.simulate_user(session, i) for i in range(concurrency)]
            )

        end = time.perf_counter()
        vram_task.cancel()

        successful = [r for r in results if r is not None]

        if not successful:
            logger.error("All requests failed during sweep.")
            return

        ttfts = [r[0] for r in successful if r[0] is not None]
        tpots = [r[1] for r in successful]
        tokens = sum(r[2] for r in successful)

        p95_ttft = np.percentile(ttfts, 95) if ttfts else 0
        median_tpot = np.median(tpots) if tpots else 0
        throughput = tokens / (end - start)

        logger.info(f"P95 TTFT: {p95_ttft:.2f} ms")
        logger.info(f"Median TPOT: {median_tpot:.2f} ms/token")
        logger.info(f"Throughput: {throughput:.2f} tok/s")
        logger.info(f"KV Peak: {self.peak_kv_cache_recorded:.1f}%")

        os.makedirs("logs", exist_ok=True)
        csv_file = "logs/monolithic_benchmark_results.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "Context", "Topology", "Parallelism", "Concurrency",
                    "TTFT (ms)", "TPOT (ms/tok)", "Throughput (tok/s)",
                    "P95 Latency (ms)", "Peak KV CACHE(%)",
                    "Peak SM Util (%)", "Bottleneck"
                ])

            writer.writerow([
                self.target_input_tokens,
                self.topology,
                "TP=1",
                concurrency,
                round(np.median(ttfts), 2) if ttfts else 0,
                round(median_tpot, 2),
                round(throughput, 2),
                round(p95_ttft, 2),
                round(self.peak_kv_cache_recorded, 2),
                "TBD",
                "TBD"
            ])

        logger.info("Saved sweep results.")

    async def execute_single_sweep(self, concurrency):
        await self.run_warmup()
        await self.run_concurrency_sweep(concurrency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)

    args = parser.parse_args()

    suite = BenchmarkSuite(
        target_url=args.url,
        engine=args.engine,
        topology=args.topology,
        model=args.model,
        context_len=args.context
    )

    asyncio.run(suite.execute_single_sweep(args.concurrency))