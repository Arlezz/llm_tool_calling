"""The tool-calling agent: understands the question, decides whether/how to search, evaluates
results, refines and iterates via the `web_search` tool, and eventually concludes.

Talks to the model through llm_client (a natively-running mlx_lm.server), not in-process - so this
module itself has no Apple Silicon/Metal dependency and can run anywhere, containerized or not.

Two entry points share this loop:
- main.py's REPL, which keeps one `messages` list across turns (conversation memory).
- run_agent_turn(), a stateless one-shot turn (fresh `messages`) used by the HTTP API.
"""

import contextvars
import json
import re
import traceback
import uuid
from datetime import datetime
from urllib.parse import urlparse

from app import llm_client
from app.prompt import AGENT_SYSTEM_PROMTP
from app.search import MAX_SOURCES, normalize_url, run_search

MAX_TOKENS = 2048
MAX_TOOL_ROUNDS = 8  # safety cap on consecutive tool calls within a single user turn

# Matches the shape of the fabricated-citation failures caught during testing (fake URLs, "según
# X", "according to Y") - not any factual-sounding text in general. A plain conversational reply
# (greeting, thanks, small talk) never matches this, so it's never forced into a pointless search.
_UNVERIFIED_CLAIM_RE = re.compile(
    r"https?://|www\.\w|seg[uú]n\b|fuentes?\s*[:\-]|confirmad[oa]s?\s+por|de acuerdo (a|con)\b|"
    r"according to|sources?\s*[:\-]|confirmed by|studies show|data from",
    re.IGNORECASE,
)


def _looks_like_unverified_claim(content):
    return bool(content) and bool(_UNVERIFIED_CLAIM_RE.search(content))

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


def new_conversation():
    """Fresh conversation: just the system prompt. Tools are passed per-request (see
    generate_from_model), not baked into the prompt text."""
    current_date = datetime.now().astimezone().date().isoformat()
    return [{"role": "system", "content": AGENT_SYSTEM_PROMTP.format(current_date=current_date)}]


def generate_from_model(messages):
    """Streams one assistant turn from the model server.

    Returns (assistant_message, finish_reason). assistant_message is an OpenAI-style dict
    ({"role": "assistant", "content": str | None, "tool_calls": [...]}) ready to append to
    `messages` or to hand to a caller inspecting tool_calls. Prints content and reasoning
    (<think>) live as they stream, same as the raw text used to look before the model server.
    """
    stream = llm_client.chat(messages, tools=TOOLS, stream=True, enable_thinking=True, max_tokens=MAX_TOKENS)

    content = ""
    tool_calls = {}
    finish_reason = None

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        finish_reason = choice.finish_reason or finish_reason

        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            print(reasoning, end="", flush=True)
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content

        for tc_delta in delta.tool_calls or []:
            entry = tool_calls.setdefault(
                tc_delta.index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    entry["function"]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    entry["function"]["arguments"] += tc_delta.function.arguments

    print()

    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        ordered = [tool_calls[i] for i in sorted(tool_calls)]
        for tc in ordered:
            if not tc["id"]:
                tc["id"] = f"call_{uuid.uuid4().hex[:8]}"
        message["tool_calls"] = ordered

    return message, finish_reason


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


def run_tool_calling_loop(messages, question):
    """Runs the tool-calling loop for one turn until a final answer or the usual limits.

    Mutates `messages` in place, appending the assistant/tool messages generated this turn - the
    REPL's persistent conversation naturally accumulates history across turns this way.

    `question` is the original user question for this turn, used only for the forced-search
    safety net below.

    Returns (final_text | None, truncated: bool). final_text is None if truncated or if the
    round limit was hit without a final answer.
    """
    tool_rounds = 0
    seen_calls = set()
    while True:
        assistant_message, finish_reason = generate_from_model(messages)

        if finish_reason == "length":
            print(
                "[warn] Respuesta truncada por max_tokens. Intenta reformular la pregunta.\n"
            )
            return None, True

        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            content = assistant_message.get("content")

            if tool_rounds > 0 or not _looks_like_unverified_claim(content):
                # Either already searched this turn, or this reply has nothing citation-like to
                # verify (greeting, thanks, small talk, a question about the assistant itself) -
                # trust the model's judgment that no search was needed.
                messages.append(assistant_message)
                return content, False

            # The model asserted facts/citations (URLs, "según X") without ever calling
            # web_search this turn - an instruction to prioritize search is never a hard
            # guarantee, and it can (and did, during testing) answer confidently while
            # fabricating citations. Don't keep this ungrounded answer; force a real search
            # before letting it conclude.
            print(
                "[warn] El modelo hizo afirmaciones sin respaldo real; se fuerza una búsqueda "
                "antes de confirmar la respuesta.\n"
            )
            tool_rounds += 1
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            print(f"[tool] web_search({{'query': {question!r}}}) [forced]")
            result = execute_tool({"name": "web_search", "arguments": {"query": question}})
            print(f"[tool result] {result}\n")
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": question})},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
            continue

        tool_rounds += 1
        if tool_rounds > MAX_TOOL_ROUNDS:
            print(
                f"[warn] Se alcanzó el límite de {MAX_TOOL_ROUNDS} llamadas a "
                "herramientas seguidas; deteniendo aquí.\n"
            )
            return None, False

        messages.append(assistant_message)
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                arguments = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                print(f"[tool] {name}({tc['function']['arguments']!r}) [malformed JSON]")
                result = f"Error: invalid arguments for '{name}': not valid JSON."
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue

            print(f"[tool] {name}({arguments})")
            call = {"name": name, "arguments": arguments}
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
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})


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
    """Stateless one-shot agent turn: fresh conversation, no memory across calls. Used by the
    HTTP API.

    Runs the same understand -> decide -> search -> evaluate -> iterate -> conclude loop as the
    REPL, and returns the sources gathered along the way instead of the free-text final answer.
    run_tool_calling_loop forces a real web_search if the model asserts unverified facts/citations
    without ever searching, but trusts it when it has nothing citation-like to verify (e.g. the
    query itself is conversational, not a real research question) - so not_found here can mean
    either "searched, found nothing" or "didn't need to search at all".
    """
    messages = new_conversation()
    messages.append({"role": "user", "content": question})

    token = _collected_sources.set([])
    try:
        run_tool_calling_loop(messages, question)
        raw_results = _collected_sources.get()
    finally:
        _collected_sources.reset(token)

    return _aggregate_sources(raw_results)
