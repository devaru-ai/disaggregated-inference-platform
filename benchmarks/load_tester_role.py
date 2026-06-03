import asyncio
import httpx
import time
from collections import Counter

ROUTER_URL = "http://127.0.0.1:8090/v1/completions"
TARGET_CONCURRENCY = 150 
TEST_DURATION_SEC = 30

PAYLOAD = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Explain the mechanics of phase disaggregation in distributed systems.",
    "max_tokens": 50,
    "temperature": 0.0
}

async def fire_request(client):
    try:
        response = await client.post(ROUTER_URL, json=PAYLOAD)
        # Drain the stream to ensure the connection completes
        async for _ in response.aiter_bytes():
            pass
        return response.status_code
    except httpx.ReadTimeout:
        return "TIMEOUT"
    except Exception as e:
        print(f"DEBUG Connection Error: {type(e).__name__} - {str(e)}")
        return f"ERROR_{type(e).__name__}"

async def run_stress_test():
    print(f"Launching Stress Test | Target Concurrency: {TARGET_CONCURRENCY}")
    print(f"Target URL: {ROUTER_URL}")
    print(f"Payload: {PAYLOAD}")
    
    # Custom limits to allow the test script to open enough sockets
    limits = httpx.Limits(max_connections=TARGET_CONCURRENCY + 50, max_keepalive_connections=TARGET_CONCURRENCY)
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0)
    
    start_time = time.time()
    
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Fire all requests concurrently
        tasks = [fire_request(client) for _ in range(TARGET_CONCURRENCY)]
        results = await asyncio.gather(*tasks)
        
    duration = time.time() - start_time
    tally = Counter(results)
    
    print("\n" + "="*50)
    print("ADMISSION CONTROL BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Duration:      {duration:.2f} seconds")
    print(f"Total Requests:      {TARGET_CONCURRENCY}")
    print(f"Overall Throughput:  {TARGET_CONCURRENCY / duration:.2f} req/sec")
    print("-" * 50)
    print(f"✅ 200 OK (Processed):      {tally.get(200, 0)}")
    print(f"503 Rejected (Saturated): {tally.get(503, 0)}")
    print(f"❌ Timeouts:                 {tally.get('TIMEOUT', 0)}")
    
    total_errors = sum(count for key, count in tally.items() if isinstance(key, str) and key.startswith('ERROR'))
    print(f"⚠️ Network Errors:           {total_errors}")
    
    print("-" * 50)
    print(f"RAW STATUS CODES / RESULTS: {dict(tally)}")
    print("="*50)
    
    if tally.get(503, 0) > 0:
        print("\n SUCCESS: Circuit breaker successfully protected the engine from overload.")
    else:
        print("\n WARNING: No 503s detected. System was not pushed to saturation.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())