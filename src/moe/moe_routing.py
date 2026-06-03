import uvicorn
import logging
import httpx
import asyncio
import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

VLLM_CLUSTER_URL = os.getenv("VLLM_CLUSTER_URL", "http://localhost:8000")
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MoE-Router")

MAX_CONCURRENCY = 128
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@app.post("/v1/completions")
async def proxy_mixtral(request: Request):

    await semaphore.acquire()

    body = await request.json()
    body["model"] = MODEL_NAME

    async def stream():

        try:
            if await request.is_disconnected():
                return

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_CLUSTER_URL}/v1/completions",
                    json=body,
                ) as r:

                    if r.status_code != 200:
                        err = await r.aread()
                        yield f"data: {err.decode(errors='ignore')}\n\n"
                        return

                    async for line in r.aiter_lines():

                        if await request.is_disconnected():
                            break

                        if not line:
                            continue

                        if line.startswith("data:"):
                            yield f"{line}\n"
                        else:
                            yield f"data: {line}\n"

        except Exception as e:
            logger.error(f"Cluster failure: {e}")
            yield f"data: {json.dumps({'error': 'MoE Cluster unreachable'})}\n\n"

        finally:
            semaphore.release()

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    logger.info("Starting MoE Routing Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8080)