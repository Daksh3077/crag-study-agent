"""Scoped hybrid retrieval with pagination, version checks and bounded cache."""
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from app.core.config import get_settings
from app.core.database import get_supabase
from app.services.embeddings import EMBEDDING_VERSION, embed_query

logger = logging.getLogger(__name__)
_cache = OrderedDict()
_cache_lock = threading.Lock()
_search_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")

# Question scaffolding should not outrank the actual subject in handwritten notes.
_STOP_WORDS = set("a an the and or of in on at to for from with by as is are was were be been being what which who when where how do does did can could would should please me my this these those it its".split())
_TASK_WORDS = set("explain define describe tell give show list discuss summarize summarise definition meaning type kind example overview detail note document pdf".split())

def _normalize_word(word: str) -> str:
    # Conservative English plural normalization; no downloads or fuzzy topic substitution.
    if not word.isascii() or not word.isalpha() or len(word) < 4:
        return word
    if word.endswith("ies") and word not in {"series", "species"}:
        return word[:-3] + "y"
    if word.endswith(("sses", "ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "ics", "ies")):
        return word[:-1]
    return word

def _tokenize(text: str) -> list[str]:
    return [_normalize_word(w) for w in re.findall(r"\w+", text.casefold(), re.UNICODE) if w not in _STOP_WORDS]

def topic_terms(question: str) -> set[str]:
    return set(_tokenize(question)) - _TASK_WORDS

def has_topic_match(question: str, content: str) -> bool:
    """An exact lexical anchor allows a checked attempt, not an assurance of support."""
    terms = topic_terms(question)
    # A count alone (e.g. "give 5 examples") is not a topic match.
    return bool(terms) and any(any(ch.isalpha() for ch in term) for term in terms) and terms <= set(_tokenize(content))

def has_topic_heading(question: str, content: str) -> bool:
    terms = topic_terms(question)
    if not terms or not any(any(ch.isalpha() for ch in term) for term in terms):
        return False
    for line in content.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]\s*|[-*#]+\s*)", "", line)
        if set(_tokenize(line)) == terms:
            return True
    return False

def _all_candidate_chunks(document_ids: list[str]) -> list[dict]:
    if not document_ids:
        return []
    settings = get_settings()
    rows, offset = [], 0
    while True:
        batch = (get_supabase().table("document_chunks")
                 .select("id,document_id,content,page_number,chunk_index,metadata")
                 .in_("document_id", document_ids)
                 .eq("metadata->>embedding_model", settings.embedding_model)
                 .eq("metadata->>embedding_version", EMBEDDING_VERSION)
                 .order("id").range(offset, offset + 499).execute().data or [])
        rows.extend(batch)
        if len(batch) < 500:
            return rows
        offset += len(batch)

def invalidate_bm25_cache(document_ids=None):
    with _cache_lock:
        _cache.clear()

def _get_bm25_index(document_ids: list[str]):
    key = tuple(sorted(set(document_ids)))
    settings = get_settings()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < settings.bm25_cache_ttl_seconds:
            _cache.move_to_end(key)
            return cached[1:]
    rows = [r for r in _all_candidate_chunks(document_ids) if r.get("content", "").strip()]
    corpus = [_tokenize(r["content"]) for r in rows]
    valid = [(r, tokens) for r, tokens in zip(rows, corpus) if tokens]
    rows, corpus = ([r for r, _ in valid], [t for _, t in valid])
    index = BM25Okapi(corpus) if corpus else None
    if index:
        # Vanilla BM25Okapi can give negative/zero IDF on a one-page corpus.
        index.idf = {word: max(score, .1) for word, score in index.idf.items()}
    with _cache_lock:
        _cache[key] = (time.monotonic(), rows, index)
        _cache.move_to_end(key)
        while len(_cache) > settings.bm25_cache_entries:
            _cache.popitem(last=False)
    return rows, index

def keyword_search(query: str, document_ids: list[str], top_k: int) -> list[dict]:
    if not document_ids or not query.strip():
        return []
    started = time.perf_counter()
    rows, index = _get_bm25_index(document_ids)
    logger.info('Retrieval keyword index ready in %.3fs (%d passages)', time.perf_counter()-started, len(rows))
    if not index:
        return []
    tokens = _tokenize(query)
    scored = [dict(row, bm25_score=float(score), topic_heading_match=has_topic_heading(query, row["content"]))
              for row, score in zip(rows, index.get_scores(tokens)) if score > 0]
    return sorted(scored, key=lambda row: (row["topic_heading_match"], row["bm25_score"]), reverse=True)[:top_k]

def vector_search(query: str, document_ids: list[str], top_k: int) -> list[dict]:
    if not document_ids or not query.strip():
        return []
    settings = get_settings()
    started = time.perf_counter()
    query_embedding = embed_query(query)
    logger.info('Retrieval query embedding ready in %.3fs', time.perf_counter()-started)
    started = time.perf_counter()
    result = get_supabase().rpc("match_study_chunks", {
        "query_embedding": query_embedding, "match_count": top_k,
        "filter_document_ids": document_ids, "query_model": settings.embedding_model,
        "query_version": EMBEDDING_VERSION}).execute()
    logger.info('Retrieval vector database request completed in %.3fs', time.perf_counter()-started)
    return [row for row in result.data or [] if row.get("similarity", 0) >= settings.vector_similarity_threshold]

def _rrf(*ranked_lists: list[dict]) -> list[dict]:
    scores, items = {}, {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = str(item["id"])
            items[key] = {**items.get(key, {}), **item}
            scores[key] = scores.get(key, 0) + 1 / (get_settings().rrf_k + rank)
    return [dict(items[key], fusion_score=scores[key]) for key in sorted(scores, key=scores.get, reverse=True)]

def _expand_neighbors(chunks: list[dict], document_ids: list[str] | None = None) -> list[dict]:
    if not chunks or not get_settings().neighbor_window:
        return chunks
    rows, _ = _get_bm25_index(document_ids or list({c["document_id"] for c in chunks}))
    lookup = {(r["document_id"], r["chunk_index"]): r for r in rows}
    expanded, seen = [], set()
    for seed in chunks:
        for index in range(seed["chunk_index"] - get_settings().neighbor_window,
                           seed["chunk_index"] + get_settings().neighbor_window + 1):
            row = seed if index == seed["chunk_index"] else lookup.get((seed["document_id"], index))
            if row and row["id"] not in seen:
                seen.add(row["id"])
                expanded.append(row)
    return expanded

def hybrid_retrieve(query: str, document_ids: list[str], top_k: int, warnings: list[str] | None = None) -> list[dict]:
    if not document_ids:
        return []
    warnings = warnings if warnings is not None else []
    results = []
    searches = [(name, _search_pool.submit(search, query, document_ids, top_k * 2))
                for name, search in (("semantic", vector_search), ("keyword", keyword_search))]
    for name, future in searches:
        try:
            results.append(future.result())
        except Exception:
            logger.exception("%s retrieval failed", name)
            warnings.append(f"{name.title()} retrieval was unavailable; results may be incomplete.")
    if not results:
        raise RuntimeError("Both retrieval methods failed")
    fused = _rrf(*results)
    # A definition at the end of an OCR page should survive a longer unrelated
    # paragraph on that page and repeated mentions elsewhere in the chapter.
    return sorted(fused, key=lambda row: bool(row.get("topic_heading_match")), reverse=True)[:top_k]
