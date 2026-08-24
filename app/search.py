"""Search phase: question -> generated queries -> search engine -> deduplicated, ranked sources.

Deliberately does not fetch or scrape the URLs it returns; that is a separate, later phase.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ddgs import DDGS

from app import llm_client
from app.prompt import QUERY_GEN_SYSTEM_PROMPT

WEB_SEARCH_TIMEOUT = 10  # seconds, per individual query
MAX_QUERIES = 3  # queries generated per search
MAX_RESULTS_PER_QUERY = 5
MAX_SOURCES = 10  # final sources returned per search
MAX_RESULTS_PER_DOMAIN = 2  # domain diversity cap on the final sources
QUERY_GEN_MAX_TOKENS = 256  # short JSON output, no need for a large budget

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid"}


@dataclass
class SearchResult:
    title: str
    url: str
    summary: str
    query: str  # which query produced this; internal use only, not part of the final output
    rank: int  # position within this query's own results (0 = top)


@dataclass
class Source:
    title: str
    url: str
    summary: str
    matched_queries: list = field(default_factory=list)
    best_rank: int = 0


class SearchEngine(Protocol):
    def search(self, query: str, max_results: int) -> list:
        ...


class DDGSSearchEngine:
    def __init__(self, timeout):
        self.timeout = timeout

    def search(self, query, max_results):
        results = DDGS(timeout=self.timeout).text(
            query, max_results=max_results, safesearch="moderate"
        )
        parsed = []
        for rank, r in enumerate(results):
            url = r.get("href", "").strip()
            if not url:
                continue
            title = " ".join(r.get("title", "").split())
            summary = " ".join(r.get("body", "").split())
            parsed.append(SearchResult(title=title, url=url, summary=summary, query=query, rank=rank))
        return parsed


def _parse_queries(text, max_queries):
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        return None

    seen = set()
    queries = []
    for q in raw_queries:
        if not isinstance(q, str):
            continue
        normalized = " ".join(q.split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        queries.append(normalized)
        if len(queries) >= max_queries:
            break

    return queries or None


def generate_queries(question, max_queries):
    """One-shot call to the model server: expands `question` into up to `max_queries` search
    queries. Non-streaming, thinking disabled - this just needs a short JSON reply."""
    response = llm_client.chat(
        [
            {"role": "system", "content": QUERY_GEN_SYSTEM_PROMPT.format(max_queries=max_queries)},
            {"role": "user", "content": question},
        ],
        enable_thinking=False,
        max_tokens=QUERY_GEN_MAX_TOKENS,
    )
    text = response.choices[0].message.content or ""

    queries = _parse_queries(text, max_queries)
    if queries is None:
        print(f"[search] query generation failed to parse; falling back to original question. Raw output: {text!r}")
        return [question]

    print(f"[search] generated {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} for {question!r}: {queries}")
    return queries


def normalize_url(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    kept_params = sorted(
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PARAM_PREFIXES) and k.lower() not in _TRACKING_PARAMS
    )
    return urlunparse((parsed.scheme.lower(), netloc, path, "", urlencode(kept_params), ""))


def _timed_search(engine, query, max_results):
    start = time.monotonic()
    try:
        results = engine.search(query, max_results)
    except Exception as e:
        print(f"[search] query {query!r} failed after {time.monotonic() - start:.2f}s: {e}")
        raise
    print(f"[search] query {query!r} -> {len(results)} results ({time.monotonic() - start:.2f}s)")
    return results


def run_queries(queries, engine, max_results_per_query):
    """Runs all queries concurrently. A failing query is logged and excluded, not fatal."""
    all_results = []
    failures = 0
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [executor.submit(_timed_search, engine, q, max_results_per_query) for q in queries]
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:  # noqa: BLE001 - already logged in _timed_search
                failures += 1
    return all_results, failures


def dedupe(results):
    """Merges results that share a normalized URL, keeping every query that matched it."""
    sources = {}
    for r in results:
        key = normalize_url(r.url)
        existing = sources.get(key)
        if existing is None:
            sources[key] = Source(
                title=r.title, url=r.url, summary=r.summary,
                matched_queries=[r.query], best_rank=r.rank,
            )
            continue
        if r.query not in existing.matched_queries:
            existing.matched_queries.append(r.query)
        existing.best_rank = min(existing.best_rank, r.rank)
        if not existing.title and r.title:
            existing.title = r.title
        if not existing.summary and r.summary:
            existing.summary = r.summary
    return list(sources.values())


def rank_and_filter(sources, max_sources, max_results_per_domain):
    """Ranks by (queries that agreed on it, then best engine rank) and caps per-domain results."""
    ranked = sorted(sources, key=lambda s: (-len(s.matched_queries), s.best_rank))

    selected = []
    domain_counts = {}
    for s in ranked:
        domain = urlparse(s.url).netloc.lower().removeprefix("www.")
        if domain_counts.get(domain, 0) >= max_results_per_domain:
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        selected.append(s)
        if len(selected) >= max_sources:
            break
    return selected


def search_pipeline(question, engine, max_queries, max_results_per_query, max_sources, max_results_per_domain):
    """Returns {"status": "found" | "not_found" | "error", "sources": [{"title", "summary", "url"}]}."""
    queries = generate_queries(question, max_queries)

    raw_results, failures = run_queries(queries, engine, max_results_per_query)
    print(f"[search] {len(raw_results)} raw candidates from {len(queries)} queries ({failures} query failures)")

    if failures == len(queries):
        return {"status": "error", "sources": []}

    sources = dedupe(raw_results)
    print(f"[search] {len(sources)} candidates after dedup")

    final = rank_and_filter(sources, max_sources, max_results_per_domain)
    print(f"[search] {len(final)} final sources after ranking/filtering")

    if not final:
        return {"status": "not_found", "sources": []}

    return {
        "status": "found",
        "sources": [{"title": s.title, "summary": s.summary, "url": s.url} for s in final],
    }


def run_search(
    query,
    max_queries=MAX_QUERIES,
    max_results_per_query=MAX_RESULTS_PER_QUERY,
    max_sources=MAX_SOURCES,
    max_results_per_domain=MAX_RESULTS_PER_DOMAIN,
):
    """Convenience entry point used by both the REPL tool and the HTTP API."""
    engine = DDGSSearchEngine(WEB_SEARCH_TIMEOUT)
    return search_pipeline(query, engine, max_queries, max_results_per_query, max_sources, max_results_per_domain)
