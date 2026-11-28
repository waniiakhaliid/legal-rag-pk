"""
Query-side RAG pipeline for LegalRAG.

Flow:  user message -> prepare_query (language + English search query + doc routing)
                    -> retrieve (dense search + exact section-number lookup)
                    -> stream_answer (Gemini, grounded in retrieved chunks)
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Iterator

from google import genai
from google.genai import types
from pinecone import Pinecone

import config
from config import (
    GEMINI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
    GEMINI_PREPARE_MODEL,
    GEMINI_FALLBACK_MODEL,
)

CHAT_MODEL = GEMINI_GENERATION_MODEL
PREPARE_MODEL = GEMINI_PREPARE_MODEL or GEMINI_GENERATION_MODEL
FALLBACK_MODEL = GEMINI_FALLBACK_MODEL

# Optional settings: add these to config.py if you want to change them.
TOP_K = getattr(config, "TOP_K_RETRIEVAL", 10)          # semantic candidates
SECTION_TOP_K = getattr(config, "SECTION_TOP_K", 4)     # exact section matches
MAX_CONTEXT = getattr(config, "MAX_CONTEXT_CHUNKS", 8)  # chunks sent to the LLM
MAX_HISTORY = 6  # messages kept for context

# Timeout is in milliseconds. Fail fast so we can move to the fallback model.
# If long answers get cut off mid-stream, raise this value.
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=30_000),
)
_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

DOC_LABELS = {
    "penal_code": "Pakistan Penal Code",
    "criminal_procedure": "Code of Criminal Procedure",
    "civil_procedure": "Code of Civil Procedure",
    "constitution": "Constitution of Pakistan",
}

# Urdu (۰-۹) and Arabic-Indic (٠-٩) digits -> ASCII, so "دفعہ ۳۰۲" works
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_SECTION_RE = re.compile(
    r"(?:section|sec\.?|s\.|article|art\.?|دفعہ|دفع|آرٹیکل|شق)\s*(\d{1,3}[A-Za-z]{0,3})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RETRY_CODES = {500, 502, 503, 504}          # retry the SAME model
FAILOVER_CODES = RETRY_CODES | {429, 404}   # move on to ANOTHER model


def _is_transient(e: Exception) -> bool:
    """Server-side hiccup worth retrying on the same model."""
    return getattr(e, "code", None) in RETRY_CODES


def _is_failover(e: Exception) -> bool:
    """Quota / missing model / timeout / server error: try the fallback model."""
    return (
        getattr(e, "code", None) in FAILOVER_CODES
        or "timeout" in type(e).__name__.lower()
    )


def _retry(fn, attempts: int = 2):
    """Short retry for transient server errors only (never for 429)."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not _is_transient(e):
                raise
            time.sleep(1)


def _models(primary: str) -> list[str]:
    models = [primary]
    if FALLBACK_MODEL and FALLBACK_MODEL != primary:
        models.append(FALLBACK_MODEL)
    return models


def _generate(contents, cfg, primary: str):
    """Non-streaming generation: primary model, then the fallback model."""
    last = None
    for model in _models(primary):
        try:
            return _retry(lambda m=model: client.models.generate_content(
                model=m, contents=contents, config=cfg,
            ))
        except Exception as e:
            last = e
            print(f"[generate] {model} failed: {getattr(e, 'code', type(e).__name__)}")
            if not _is_failover(e):
                raise
    raise last


def _has_urdu(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


@lru_cache(maxsize=256)
def embed_query(text: str) -> list[float]:
    # Same call style as ingest.py (no task_type) so query and document
    # embeddings live in the same space. Cached so repeated queries are free.
    result = _retry(lambda: client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL, contents=text,
    ))
    return result.embeddings[0].values


def extract_section_ref(*texts: str) -> str | None:
    for t in texts:
        m = _SECTION_RE.search(t.translate(_DIGITS))
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Step 1: understand the question
# ---------------------------------------------------------------------------

PREPARE_PROMPT = """You prepare search queries for a retrieval system over Pakistani law
(documents: penal_code, criminal_procedure, civil_procedure, constitution).
The documents are in English. The user may write in English or Roman Urdu.

Given the conversation and the latest user message, return JSON with:
- "language": "en" or "ur" (language the user wrote the latest message in;
  Roman Urdu counts as "ur")
- "search_query": a standalone English search query capturing what the user wants.
  Resolve follow-ups using the conversation (e.g. "what about the second offence?").
  Keep any section/article numbers. Use precise legal terms where you can.
  If the user describes a situation in everyday words, translate it into the legal
  concepts and offences that may apply (for example: public nuisance, criminal
  intimidation, hurt, assault, domestic violence, defamation, trespass, cheating,
  property dispute) instead of echoing their words. Name several relevant concepts
  if more than one could apply.
- "doc_type": one of "penal_code", "criminal_procedure", "civil_procedure",
  "constitution", or null if unclear or if several documents may apply.

Conversation so far:
{history}

Latest user message:
{message}
"""


def _format_history(history: list[dict], n: int = 4) -> str:
    recent = history[-n:]
    if not recent:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)


def prepare_query(message: str, history: list[dict]) -> dict:
    fallback = {
        "language": "ur" if _has_urdu(message) else "en",
        "search_query": message,
        "doc_type": None,
    }
    try:
        resp = _generate(
            PREPARE_PROMPT.format(history=_format_history(history), message=message),
            types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
            primary=PREPARE_MODEL,
        )
        data = json.loads(resp.text)
        return {
            "language": data.get("language") if data.get("language") in ("en", "ur") else fallback["language"],
            "search_query": data.get("search_query") or message,
            "doc_type": data.get("doc_type") if data.get("doc_type") in DOC_LABELS else None,
        }
    except Exception as e:
        print(f"[prepare_query] falling back: {e}")
        return fallback


# ---------------------------------------------------------------------------
# Step 2: retrieve
# ---------------------------------------------------------------------------

def _to_hit(match) -> dict:
    md = match.metadata or {}
    return {
        "id": match.id,
        "score": float(match.score),
        "text": md.get("text", ""),
        "doc_type": md.get("doc_type", ""),
        "doc_label": DOC_LABELS.get(md.get("doc_type", ""), md.get("source", "")),
        "section_number": md.get("section_number", ""),
        "section_title": md.get("section_title", ""),
        "content_type": md.get("content_type", "section"),
    }


def retrieve(search_query: str, original_message: str = "", doc_type: str | None = None) -> list[dict]:
    vector = embed_query(search_query)
    ref = extract_section_ref(search_query, original_message)

    def _query(top_k: int, flt: dict):
        return _retry(lambda: _index.query(
            vector=vector, top_k=top_k, include_metadata=True, filter=flt,
        ))

    # Sections only: footnote chunks are short amendment notes that match
    # loosely on almost any query and crowd out the real provisions.
    semantic_filter = {"content_type": {"$eq": "section"}}

    # Run the exact-section lookup and the semantic search in parallel.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_sem = ex.submit(_query, TOP_K, semantic_filter)
        f_ref = None
        if ref:
            ref_filter = {"section_number": {"$eq": ref}, "content_type": {"$eq": "section"}}
            if doc_type:
                ref_filter["doc_type"] = {"$eq": doc_type}
            f_ref = ex.submit(_query, SECTION_TOP_K, ref_filter)

        hits: dict[str, dict] = {}
        if f_ref is not None:
            for m in f_ref.result().matches:       # exact matches take priority
                hits[m.id] = _to_hit(m)
        for m in f_sem.result().matches:
            hits.setdefault(m.id, _to_hit(m))

    # exact-section hits first, then by score
    ordered = list(hits.values())
    if ref:
        ordered.sort(key=lambda h: (h["section_number"] != ref, -h["score"]))
    else:
        ordered.sort(key=lambda h: -h["score"])

    # One chunk per section, so a single section can't fill the whole context.
    seen: set = set()
    unique: list[dict] = []
    for h in ordered:
        key = (h["doc_type"], h["section_number"]) if h["section_number"] else h["id"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique[:MAX_CONTEXT]


# ---------------------------------------------------------------------------
# Step 3: generate
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are a legal information assistant for Pakistani law. Answer ONLY from the numbered context passages (Pakistan Penal Code, CrPC, CPC, Constitution). Never use outside knowledge or invent sections, wording, or punishments.

When the context contains a provision that fits the situation, reply in exactly this order:

1. The Law: for every applicable section, give the section number and document, then its exact wording in a quote block, copied word for word (use [...] for omissions). Include all applicable sections, each with its own quote, and none that don't apply. Cite as (Section X, Document [n]).
2. What It Means: explain each quoted section in plain language, defining legal terms. Add nothing the text doesn't say.
3. How It May Apply to You: link the sections to the person's situation using cautious wording ("may apply"), staying within the quoted text.

If no provision clearly matches, say: "I could not find a section addressing this in the available laws." Do not substitute a similar section. If coverage is partial, explain what exists and state what is not covered. 
If a passage looks cut off, say so and do not complete it.
If the question is too vague to identify the provision, ask ONE short clarifying question instead of guessing.
If the person is in danger or being abused, first tell them to contact police or emergency services, then continue.
Mention "Amendment/Footnote Reference" passages only when they affect a quoted section.
Reply in the requested language. For Urdu, keep the quoted law in its original wording, write parts 2 and 3 in Urdu script, and keep section numbers, 
act names, and [n] markers unchanged. End with one line: this is general legal information, not legal advice; consult a qualified lawyer."""


def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "(no relevant passages were found)"
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['text']}")
    return "\n\n".join(blocks)


def stream_answer(message: str, hits: list[dict], language: str, history: list[dict]) -> Iterator[str]:
    lang_name = "Urdu" if language == "ur" else "English"
    user_turn = (
        f"CONTEXT PASSAGES:\n{_format_context(hits)}\n\n"
        f"QUESTION: {message}\n\n"
        f"Reply in {lang_name}."
    )

    contents = []
    for m in history[-MAX_HISTORY:]:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_turn)]))

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # The streaming request only fires when we start iterating, so retries must
    # wrap the iteration. We only retry/fail over if no text has been sent yet;
    # once tokens have reached the user, a mid-stream failure is raised instead.
    last = None
    for model in _models(CHAT_MODEL):
        for attempt in range(2):
            started = False
            try:
                stream = client.models.generate_content_stream(
                    model=model, contents=contents, config=cfg,
                )
                for chunk in stream:
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                last = e
                print(f"\n[stream] {model} failed: {getattr(e, 'code', type(e).__name__)}")
                if started or not _is_failover(e):
                    raise
                if not _is_transient(e):
                    break          # 429 / 404 / timeout: go straight to the next model
                time.sleep(1)      # 5xx: one quick retry on the same model
    raise last


def public_sources(hits: list[dict]) -> list[dict]:
    """What the UI shows as citations; index i here = citation marker [i+1]."""
    return [
        {
            "n": i,
            "document": h["doc_label"],
            "section_number": h["section_number"],
            "section_title": h["section_title"],
            "content_type": h["content_type"],
            "snippet": h["text"][:600],
            "score": round(h["score"], 3),
        }
        for i, h in enumerate(hits, 1)
    ]