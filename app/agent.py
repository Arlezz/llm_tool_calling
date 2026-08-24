"""The tool-calling agent: understands the question, decides whether/how to search, evaluates
results, refines and iterates via the `web_search` tool, and eventually concludes.

Two entry points share this loop:
- main.py's REPL, which keeps one `cache` across turns (conversation memory).
- run_agent_turn(), a stateless one-shot turn (fresh cache) used by the HTTP API.
"""

import contextvars
import json
import re
import traceback
from datetime import datetime
from urllib.parse import urlparse

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from prompt import AGENT_SYSTEM_PROMTP
from search import (
    MAX_SOURCES,
    model,
    model_lock,
    normalize_url,
    run_search,
    sampler,
    tokenizer,
)

MAX_KV_SIZE = None
MAX_TOKENS = 2048  # enable_thinking burns tokens on <think> before it ever reaches a tool_call/answer
MAX_TOOL_ROUNDS = 8  # safety cap on consecutive tool calls within a single user turn

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Populated only during run_agent_turn(): the raw {"status","sources"} dict from every
# web_search call made in this turn, so the API can build its own {"status","sources"} response
# without depending on the free-text answer. None (the default) in the REPL, which doesn't use it.
_collected_sources = contextvars.ContextVar("collected_sources", default=None)


def tool_web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "Error: empty search query."

    result = run_search(query)

    sink = _collected_sources.get()
    if sink is not None:
        sink.append(result)

    if result["status"] == "error":
        return "Error: web search failed for all generated queries. Try a different or simpler query."
    if result["status"] == "not_found":
        return f"No search results found for query: {query!r}."

    lines = [f"Search results for {query!r}:"]
    for i, s in enumerate(result["sources"], start=1):
        domain = urlparse(s["url"]).netloc.removeprefix("www.")
        lines.append(f"{i}. {s['title']} ({domain})\n   URL: {s['url']}\n   {s['summary']}")
    return "\n".join(lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about a topic or question. Internally "
                "expands it into several search queries covering different angles, runs them, "
                f"and returns up to {MAX_SOURCES} deduplicated results, each with a title, URL, "
                "and snippet. Use the URLs to cite sources in your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to research.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_REGISTRY = {"web_search": tool_web_search}
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

_JSON_SCHEMA_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema, arguments):
    """Validate `arguments` against a tool's JSON Schema `parameters`. Returns an error string, or None if valid."""
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"

    properties = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"

    unknown = [k for k in arguments if k not in properties]
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"

    for key, value in arguments.items():
        expected_type = properties.get(key, {}).get("type")
        py_type = _JSON_SCHEMA_TYPES.get(expected_type)
        if py_type is not None and not isinstance(value, py_type):
            return f"argument '{key}' must be of type '{expected_type}'"

    return None


def new_cache():
    """Fresh KV cache primed with the system prompt + tool defs (encoded once)."""
    cache = make_prompt_cache(model, MAX_KV_SIZE)
    current_date = datetime.now().astimezone().date().isoformat()
    system_content = AGENT_SYSTEM_PROMTP.format(current_date=current_date)
    system_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}],
        tools=TOOLS,
        add_generation_prompt=False,
    )
    mx.eval(model(mx.array(system_prompt)[None], cache=cache))
    return cache


def generate_from_model(prompt, cache):
    """Stream a response and return (text, last_response) so callers can inspect finish_reason.

    Holds `model_lock` for just this one generate call (not the whole turn): the shared MLX
    `model` must not be invoked from two threads at once, but each call here uses its own
    `cache`, so calls don't need to stay serialized beyond that. A turn can itself trigger a
    nested generate call (search.generate_queries, inside a web_search tool execution) which
    takes the same lock again *after* this one has already been released - locking the whole
    turn instead would deadlock on that reentry, since threading.Lock isn't reentrant.
    """
    text = ""
    last_response = None
    with model_lock:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            sampler=sampler,
            prompt_cache=cache,
            max_tokens=MAX_TOKENS,
        ):
            print(response.text, end="", flush=True)
            text += response.text
            last_response = response
    print()
    return text, last_response


def strip_think(text):
    """Drop <think>...</think> blocks so we never mistake reasoning-about-tools for a real call."""
    return THINK_RE.sub("", text)


def response_is_tool_call(response):
    return bool(TOOL_CALL_RE.search(response))


def has_incomplete_tool_call(response):
    """True if there's a dangling <tool_call> the model never closed (not caught by finish_reason=="length")."""
    return response.count("<tool_call>") > response.count("</tool_call>")


def extract_tool_calls(response):
    calls = []
    for match in TOOL_CALL_RE.finditer(response):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            print(f"[tool] malformed tool_call JSON: {match.group(1)!r}")
    return calls


def normalize_call(call):
    """Signature used to detect repeated tool calls (same tool, same args, case/whitespace-insensitive)."""
    arguments = call.get("arguments") or {}
    normalized_args = {
        k: (v.strip().lower() if isinstance(v, str) else v) for k, v in arguments.items()
    }
    return call.get("name"), json.dumps(normalized_args, sort_keys=True)


def execute_tool(call):
    name = call.get("name")
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."

    arguments = call.get("arguments") or {}
    error = validate_arguments(TOOL_SCHEMAS[name], arguments)
    if error:
        return f"Error: invalid arguments for '{name}': {error}."

    try:
        return str(fn(**arguments))
    except Exception:  # noqa: BLE001 - tool boundary: must not crash the loop on arbitrary tool failures
        traceback.print_exc()
        return f"Error: tool '{name}' failed unexpectedly. Try a different query or approach."


def run_tool_calling_loop(prompt, cache, question):
    """Runs the tool-calling loop for one turn until a final answer or the usual limits.

    `question` is the original user question (plain text, not yet templated) for this turn. It is
    only used for the forced-search safety net below - the loop otherwise just keeps re-templating
    tool results into the next `prompt` as usual.

    Returns (final_text | None, truncated: bool). final_text is None if truncated or if the
    round limit was hit without a final answer.
    """
    tool_rounds = 0
    seen_calls = set()
    while True:
        response, last_response = generate_from_model(prompt, cache)

        if last_response is not None and last_response.finish_reason == "length":
            trimmed = trim_prompt_cache(cache, last_response.generation_tokens)
            print(
                f"[warn] Respuesta truncada por max_tokens; se recortaron {trimmed} "
                "tokens del cache. Intenta reformular la pregunta.\n"
            )
            return None, True

        clean_response = strip_think(response)
        if not response_is_tool_call(clean_response):
            if tool_rounds > 0:
                return clean_response, False

            # The model never called web_search this turn despite being instructed to
            # prioritize it - an instruction is never a hard guarantee, and it can (and did,
            # during testing) answer confidently while fabricating citations. Roll back this
            # ungrounded answer and force a real search before letting it conclude.
            print(
                "[warn] El modelo respondió sin buscar; se fuerza una búsqueda real antes de "
                "confirmar la respuesta.\n"
            )
            trim_prompt_cache(cache, last_response.generation_tokens)
            tool_rounds += 1

            call = {"name": "web_search", "arguments": {"query": question}}
            print(f"[tool] {call['name']}({call['arguments']}) [forced]")
            result = execute_tool(call)
            print(f"[tool result] {result}\n")

            prompt = tokenizer.apply_chat_template(
                [{"role": "tool", "content": result}], add_generation_prompt=True, enable_thinking=True
            )
            continue

        if has_incomplete_tool_call(clean_response):
            print("[warn] Se detectó un <tool_call> sin cerrar; se ignora esa parte.\n")

        tool_rounds += 1
        if tool_rounds > MAX_TOOL_ROUNDS:
            print(
                f"[warn] Se alcanzó el límite de {MAX_TOOL_ROUNDS} llamadas a "
                "herramientas seguidas; deteniendo aquí.\n"
            )
            return None, False

        tool_messages = []
        for call in extract_tool_calls(clean_response):
            print(f"[tool] {call.get('name')}({call.get('arguments')})")

            call_key = normalize_call(call)
            if call_key in seen_calls:
                result = (
                    "Error: this exact tool call was already made earlier in this turn "
                    "and returned no new information. Try a different, more specific query "
                    "or move on to answering with what you already have."
                )
            else:
                seen_calls.add(call_key)
                result = execute_tool(call)

            print(f"[tool result] {result}\n")
            tool_messages.append({"role": "tool", "content": result})

        prompt = tokenizer.apply_chat_template(
            tool_messages, add_generation_prompt=True, enable_thinking=True
        )


def _aggregate_sources(raw_results):
    """Merges the {"status","sources"} dicts from every web_search call made in a turn."""
    if not raw_results:
        return {"status": "not_found", "sources": []}
    if all(r["status"] == "error" for r in raw_results):
        return {"status": "error", "sources": []}

    seen_urls = set()
    merged = []
    for r in raw_results:
        if r["status"] != "found":
            continue
        for s in r["sources"]:
            key = normalize_url(s["url"])
            if key in seen_urls:
                continue
            seen_urls.add(key)
            merged.append(s)
            if len(merged) >= MAX_SOURCES:
                break
        if len(merged) >= MAX_SOURCES:
            break

    if not merged:
        return {"status": "not_found", "sources": []}
    return {"status": "found", "sources": merged}


def run_agent_turn(question):
    """Stateless one-shot agent turn: fresh cache, no memory across calls. Used by the HTTP API.

    Runs the same understand -> decide -> search -> evaluate -> iterate -> conclude loop as the
    REPL, and returns the sources gathered along the way instead of the free-text final answer.
    run_tool_calling_loop guarantees at least one real web_search call per turn (forcing one if
    the model concludes without searching), so not_found here always means "searched, found
    nothing", never "didn't bother searching".
    """
    cache = new_cache()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, enable_thinking=True,
    )

    token = _collected_sources.set([])
    try:
        run_tool_calling_loop(prompt, cache, question)
        raw_results = _collected_sources.get()
    finally:
        _collected_sources.reset(token)

    return _aggregate_sources(raw_results)
