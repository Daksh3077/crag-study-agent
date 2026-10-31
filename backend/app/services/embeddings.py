"""Versioned normalized passage embeddings and asymmetric BGE queries."""
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from app.core.config import get_settings
from app.services.local_models import cached_model, prefer_local
settings = get_settings()
EMBEDDING_VERSION = "passages-v2"
_warm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-warmup")
_warm_lock = threading.Lock()
_warm_future = None

def warm_models():
    """Load local weights while remote OCR runs, once per backend process."""
    global _warm_future
    def load():
        get_tokenizer()
        _get_model()
        # Keep the first question from paying for a cold cross-encoder load.
        # OCR/indexing remains usable if this optional ranking model cannot load.
        try:
            from app.agents.reranker import _get_reranker
            _get_reranker()
        except Exception:
            logging.getLogger(__name__).exception("Reranker warmup unavailable; retrieval can retry it later")
    with _warm_lock:
        if _warm_future is None or (_warm_future.done() and _warm_future.exception() is not None):
            _warm_future = _warm_executor.submit(load)
        return _warm_future

@cached_model
def _get_model():
    from sentence_transformers import SentenceTransformer
    model = prefer_local(SentenceTransformer, settings.embedding_model)
    if model.get_sentence_embedding_dimension() != settings.embedding_dim:
        raise ValueError("Embedding dimension does not match the database")
    return model

@cached_model
def get_tokenizer():
    from transformers import AutoTokenizer
    return prefer_local(AutoTokenizer.from_pretrained, settings.embedding_model)

def token_count(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False, truncation=False))

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Empty passages cannot be embedded")
    model = _get_model()
    if any(token_count(text) + 2 > model.max_seq_length for text in texts):
        raise ValueError("Passage exceeds embedding model token limit")
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=settings.embedding_batch_size).tolist()

def embed_query(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Cannot embed an empty query")
    if "bge-" in settings.embedding_model.lower():
        text = "Represent this sentence for searching relevant passages: " + text
    return _get_model().encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()
