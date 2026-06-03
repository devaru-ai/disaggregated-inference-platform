import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

LOCAL_VLLM = os.getenv("LOCAL_VLLM_PORT", "http://localhost:8000")

HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "5.0"))
HTTP_WRITE_TIMEOUT = float(os.getenv("HTTP_WRITE_TIMEOUT", "30.0"))
HTTP_POOL_TIMEOUT = float(os.getenv("HTTP_POOL_TIMEOUT", "5.0"))
MAX_HTTP_CONNECTIONS = int(os.getenv("MAX_HTTP_CONNECTIONS", "1000"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "100"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Worker - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Worker")

timeout = httpx.Timeout(
    connect=HTTP_CONNECT_TIMEOUT,
    read=None,
    write=HTTP_WRITE_TIMEOUT,
    pool=HTTP_POOL_TIMEOUT,
)

limits = httpx.Limits(
    max_connections=MAX_HTTP_CONNECTIONS,
    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
)

shared_client = httpx.AsyncClient(timeout=timeout, limits=limits)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield # App is running
    await shared_client.aclose() # App is shutting down

app = FastAPI(lifespan=lifespan)

def sse_error_bytes(
    message: str,
    status_code: int = 500,
    upstream_body: str | None = None,
) -> bytes:

    payload = {
        "error": message,
        "status_code": status_code,
    }

    if upstream_body:
        payload["upstream_body"] = upstream_body

    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

@app.post("/v1/completions")
async def completions(request: Request):

    body = await request.json()

    async def stream():
        try:
            async with shared_client.stream(
                "POST",
                f"{LOCAL_VLLM}/v1/completions",
                json=body,
            ) as upstream_response:

                if upstream_response.status_code != 200:
                    err = await upstream_response.aread()
                    logger.error(f"vLLM upstream failure: HTTP {upstream_response.status_code}")
                    yield sse_error_bytes(
                        message="vLLM upstream failure",
                        status_code=upstream_response.status_code,
                        upstream_body=err.decode(errors="ignore"),
                    )
                    return

                async for chunk in upstream_response.aiter_bytes():
                    yield chunk

        except Exception as e:
            logger.exception(f"Worker-to-vLLM proxy failure: {e}")
            yield sse_error_bytes(message="Worker proxy failure", status_code=500)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "target": LOCAL_VLLM,
    }

if __name__ == "__main__":
    logger.info(f"Starting GPU Worker Proxy | Target={LOCAL_VLLM}")
    uvicorn.run(app, host="0.0.0.0", port=8011)