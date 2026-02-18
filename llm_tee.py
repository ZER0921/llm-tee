# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    app.state.clients = []

    for i, endpoint in enumerate(global_args.llm_endpoints):
        base_url = f"{endpoint}/v1"
        app.state.clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "id": i,
            }
        )

    app.state.llm_iterator = itertools.cycle(range(len(app.state.clients)))

    yield

    for client_info in app.state.clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--llm-endpoints",
        "--llm-endpoint",
        type=str,
        nargs="+",
        default=["http://localhost:8100"],
    )

    parser.add_argument("--force-block", action='store_true', default=False)

    args = parser.parse_args()

    return args


def get_next_client(app):
    """Get the next client in round-robin fashion."""
    client_idx = next(app.state.llm_iterator)
    return app.state.clients[client_idx]


async def block_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, req_headers: dict
):
    """
    Send a request to a service using a client from the pool.
    """

    response = await client.post(
        endpoint, json=req_data, headers=req_headers
    )
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    response_json = response.json()
    await response.aclose()  # CRITICAL: Release connection back to pool

    return response_json


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, req_headers: dict
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """

    async with client.stream(
        "POST", endpoint, json=req_data, headers=req_headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_headers = {
            "Authorization": request.headers.get('authorization', '')
        }

        client_info = get_next_client(request.app)

        if global_args.force_block:
            req_data['stream'] = False

        if req_data.get('stream', False):
            async def generate_stream():
                async for chunk in stream_service_response(
                    client_info["client"], api, req_data, req_headers
                ):
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            return await block_service_response(client_info["client"], api, req_data, req_headers)

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
