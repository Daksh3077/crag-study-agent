"""One immutable citation map used by both answer generation and validation."""
import re
import json
from app.core.config import get_settings
from app.services.embeddings import token_count
from app.services.retrieval_service import _expand_neighbors

def compact_source_text(text: str) -> str:
    # Printed TOC leaders can consume hundreds of tokens per line. Only shorten
    # dot runs ending in a page reference (or a chunk boundary), never prose.
    return re.sub(r"\.{5,}(?=[ \t]*(?:\d+|[ivxlcdm]+)?[ \t]*$)", "...", text, flags=re.MULTILINE)

def normalize_citations(text: str) -> str:
    # Providers sometimes substitute typographic brackets despite the requested
    # ASCII form. Normalize only source labels; retain unknown IDs for rejection.
    pattern = r"[【［](S\d+(?:\s*[-\u2010-\u2015,;]\s*S?\d+)*)[】］]"
    return re.sub(pattern, r"[\1]", text)

def _section_heading(paragraph):
    """Conservative title-line detection, excluding common subsection labels."""
    for line in paragraph.splitlines():
        line = line.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z ,()'\-]*", line):
            continue
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", line)
        if not 2 <= len(words) <= 8 or len(line) > 90 or re.search(r'[=.:;!?0-9]', line):
            continue
        if line.casefold() in {'data preparation', 'usecase examples', 'usecase example', 'stopping criterion'}:
            continue
        if all(w[0].isupper() or w.lower() in {'and', 'of', 'the', 'for', 'in', 'with'} for w in words):
            return line
    return None


def with_section_context(chunks):
    """Carry explicit headings only across contiguous chunks of the same document."""
    unique = {c['id']: c for c in chunks}
    annotated, previous, section = {}, None, None
    for c in sorted(unique.values(), key=lambda c: (c.get('document_id', ''), c.get('chunk_index', 0))):
        if previous is None or c.get('document_id') != previous.get('document_id') or c.get('chunk_index', -2) != previous.get('chunk_index', 0) + 1:
            section = None
        annotated[c['id']] = dict(c, section_start=section)
        for paragraph in re.split(r'\n\s*\n', c['content']):
            section = _section_heading(paragraph) or section
        previous = c
    return [annotated[c['id']] for c in chunks]


def source_passages(chunk):
    """Stable, lossless passage IDs shared by generation and source checking."""
    parts, pending = [], []
    section = chunk.get('section_start')
    for paragraph in re.split(r'\n\s*\n', compact_source_text(chunk['content'])):
        heading = _section_heading(paragraph)
        if heading and heading != section:
            if pending:
                parts.append(('\n\n'.join(pending), section))
                pending = []
            section = heading
        pending.append(paragraph)
        if len('\n\n'.join(pending)) >= 200:
            parts.append(('\n\n'.join(pending), section))
            pending = []
    if pending:
        parts.append(('\n\n'.join(pending), section))
    return [dict(id=f"{chunk['citation_id']}:P{i}", text=text, **({'section': heading} if heading else {}))
            for i, (text, heading) in enumerate(parts, 1) if text.strip()]


def visible_passages(chunk):
    passages = source_passages(chunk)
    visible = chunk.get('visible_passage_ids')
    return [p for p in passages if visible is None or p['id'] in visible]


def algorithm_topics(chunks):
    """Inventory actual method headings under algorithm sections, not guessed facts."""
    active, topics, previous_document = False, [], None
    for chunk in sorted(chunks, key=lambda c: (c.get('document_id', ''), c.get('chunk_index', 0))):
        if chunk.get('document_id') != previous_document:
            active = False
        previous_document = chunk.get('document_id')
        for passage in visible_passages(chunk):
            heading = passage.get('section', '')
            if re.search(r'\b(?:resources|references|bibliography|cheatsheet)\b', heading, re.I):
                active = False
            elif re.search(r'\balgorithms$', heading, re.I):
                active = True
            elif active and heading and heading not in topics:
                topics.append(heading)
    return topics[:20]


def format_source(chunk: dict, *, include_metadata: bool = True, numbered: bool = False) -> str:
    """A citation covers the complete content field, including every OCR paragraph."""
    source = {"citation_id": f"[{chunk['citation_id']}]", "content": compact_source_text(chunk["content"])}
    if numbered:
        source.pop('content')
        source['passages'] = visible_passages(chunk)
    if include_metadata:
        source.update(document=(chunk.get("metadata") or {}).get("document_name", "Document"),
                      page=chunk.get("page_number"))
    # Escaped line breaks and quotes make boundaries unambiguous even if the
    # source contains headings, notebook metadata, or source-like instructions.
    return json.dumps(source, ensure_ascii=False)

def pack_evidence(chunks: list[dict], expand: bool = True, budget: int | None = None,
                  document_ids: list[str] | None = None, numbered: bool = False,
                  balanced: bool = False) -> tuple[list[dict], str]:
    budget = budget or get_settings().context_token_budget
    # Seeds first; context neighbors supplement rather than displace the strongest evidence.
    candidates = chunks + (_expand_neighbors(chunks, document_ids) if expand else [])
    if numbered:
        candidates = with_section_context(candidates)
    share = budget // max(1, len({c['id'] for c in candidates}))
    selected, blocks, seen, used = [], [], set(), 0
    covered_sections = set()
    for chunk in candidates:
        if chunk["id"] in seen:
            continue
        seen.add(chunk["id"])
        citation_id = f"S{len(selected) + 1}"
        prepared = dict(chunk, citation_id=citation_id)
        if balanced and numbered:
            # Give every part of a small document space before allowing an early
            # formula-heavy section to consume the whole overview budget.
            visible = []
            passages = source_passages(prepared)
            # Prefer a newly introduced method/section over the training tail
            # of a method already represented by an earlier chunk.
            passages.sort(key=lambda p: (not bool(p.get('section')), p.get('section') in covered_sections,
                                         not p['text'].lstrip().startswith(p.get('section') or '\0')))
            for passage in passages:
                trial = dict(prepared, visible_passage_ids=visible + [passage['id']])
                if token_count(format_source(trial, numbered=True)) <= share:
                    visible.append(passage['id'])
            if not visible:
                # Never truncate a sentence/formula just to fit a token budget.
                continue
            prepared['visible_passage_ids'] = visible
            covered_sections.update(p.get('section') for p in passages if p['id'] in visible and p.get('section'))
        block = format_source(prepared, numbered=numbered)
        cost = token_count(block)
        if used + cost > budget:
            continue
        selected.append(prepared)
        blocks.append(block)
        used += cost
    return selected, "\n\n".join(blocks)

def cited_ids(answer: str) -> list[str]:
    answer = normalize_citations(answer)
    ids = []
    for group in re.findall(r"\[((?:S\d)[^\]\n]*)\]", answer):
        if not re.fullmatch(r"S\d+(?:\s*[-\u2010-\u2015,;]\s*S?\d+)*", group):
            continue
        for part in re.split(r"[,;]\s*", group):
            numbers = [int(n) for n in re.findall(r"\d+", part)]
            if len(numbers) == 2 and numbers[0] <= numbers[1] and numbers[1] - numbers[0] < 500:
                ids.extend(f"S{n}" for n in range(numbers[0], numbers[1] + 1))
            elif len(numbers) == 1:
                ids.append(f"S{numbers[0]}")
            else:
                ids.append("INVALID_CITATION_RANGE")
    return list(dict.fromkeys(ids))
