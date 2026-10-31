"""Token-bounded chunks retaining source text and page structure."""
import re
from app.core.config import get_settings
from app.services.embeddings import get_tokenizer

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()

def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    settings = get_settings()
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = clean_text(text)
    if not text:
        return []
    offsets = get_tokenizer()(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)["offset_mapping"]
    chunks, start = [], 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        if end < len(offsets):
            for candidate in range(end, start + max(overlap + 1, int(size * .75)), -1):
                gap = text[offsets[candidate - 1][1]:offsets[candidate][0]]
                previous = text[offsets[candidate - 1][0]:offsets[candidate - 1][1]]
                if "\n" in gap or previous.endswith((".", "!", "?", ";")):
                    end = candidate
                    break
        piece = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks
