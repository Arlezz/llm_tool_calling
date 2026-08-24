"""Thin client for the natively-running `mlx_lm.server` (OpenAI-compatible HTTP API).

The model itself needs Apple Silicon/Metal, so it always runs natively on the host, never inside
a container - this module is how the rest of the app (which can run anywhere, containerized or
not) talks to it. Point MODEL_SERVER_URL at wherever that server is reachable from.
"""

import os

from openai import OpenAI

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:8080/v1")

# Qwen3 thinking-mode recommended sampling: temp=0.6, top_p=0.95, top_k=20, min_p=0.
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0

client = OpenAI(base_url=MODEL_SERVER_URL, api_key="not-needed")


def chat(messages, tools=None, stream=False, enable_thinking=True, max_tokens=2048):
    """Calls the model server's /chat/completions. mlx_lm.server maps the "default_model" sentinel
    to whichever model it was started with (--model); any other value is treated as a HF repo id
    to load on demand, which isn't what we want here since it's a single fixed-model server."""
    return client.chat.completions.create(
        model="default_model",
        messages=messages,
        tools=tools,
        stream=stream,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        extra_body={
            "top_k": TOP_K,
            "min_p": MIN_P,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )
