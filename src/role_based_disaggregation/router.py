import asyncio
import json
import logging
import os
import time

import httpx
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.infrastructure.scheduler.backpressure import BackpressureController

WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "100"))

SEMAPHORE_TIMEOUT = float(os.getenv("SEMAPHORE_TIMEOUT", "0.05"))
REQUEST_MAX_DURATION_SEC = float(os.getenv("REQUEST_MAX_DURATION_SEC", "300"))

HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
HTTP_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)

DISCONNECT_CHECK_INTERVAL = float(os.getenv("DISCONNECT_CHECK_INTERVAL", "1.0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Router")

app = FastAPI()

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

circuit_breaker = BackpressureController(
    max_concurrent_seqs=64,
    heartbeat_ttl_sec=5.0,
)

client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS)

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()

@app.post("/heartbeat")
async def heartbeat(request: Request):
    data = await request.json()
    circuit_breaker.update_remote_state(data.get("active_sequences", -1))
    return {"ok": True}

@app.post("/v1/completions")
async def completions(request: Request):

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=SEMAPHORE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(503, "network saturated")

    start_time = time.monotonic()

    try:
        body = await request.json()
        body["model"] = body.get("model", MODEL_NAME)

        if not circuit_breaker.try_acquire():
            raise HTTPException(503, "gpu saturated")

        async def stream():

            try:
                async with client.stream(
                    "POST",
                    f"{WORKER_URL}/v1/completions",
                    json=body,
                ) as r:

                    if r.status_code != 200:
                        err = await r.aread()
                        yield b"data: " + json.dumps({
                            "error": "upstream failure",
                            "status": r.status_code,
                            "body": err.decode(errors="ignore")
                        }).encode() + b"\n\n"
                        return

                    last_check = time.monotonic()

                    async for chunk in r.aiter_bytes():

                        # time-based disconnect check
                        if time.monotonic() - last_check > DISCONNECT_CHECK_INTERVAL:
                            last_check = time.monotonic()
                            if await request.is_disconnected():
                                await r.aclose()
                                break

                        # global request timeout guard
                        if time.monotonic() - start_time > REQUEST_MAX_DURATION_SEC:
                            await r.aclose()
                            yield b"data: {\"error\":\"timeout\"}\n\n"
                            break

                        yield chunk

            except Exception as e:
                logger.exception(e)
                yield b"data: " + json.dumps({
                    "error": "router failure"
                }).encode() + b"\n\n"

            finally:
                circuit_breaker.release()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    finally:
        semaphore.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)