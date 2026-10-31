"""Structured Ask answers: render citations in code, then check every item."""
import re
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from app.services.evidence import cited_ids, normalize_citations


class InvalidAnswerFormat(ValueError):
    pass


class AnswerItem(BaseModel):
    topic: str = Field(default='', max_length=150)
    text: str = Field(min_length=1, max_length=2000)
    source_ids: list[str] = Field(min_length=1, max_length=4)
    evidence_ids: list[str] = Field(default_factory=list, max_length=6)

    @field_validator('source_ids')
    @classmethod
    def canonical_ids(cls, ids):
        return list(dict.fromkeys(normalize_citations(s).strip('[] ') for s in ids))


class CitedAnswer(BaseModel):
    items: list[AnswerItem] = Field(max_length=20)
    insufficient_context: bool
    missing_information: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode='after')
    def consistent(self):
        if self.insufficient_context == bool(self.items):
            raise ValueError('Use empty items only when no supported answer is possible.')
        return self


def answer_schema(available_ids: list[str], passage_ids: list[str] | None = None, topics: list[str] | None = None) -> dict:
    schema = CitedAnswer.model_json_schema()
    schema.setdefault('required', []).append('missing_information')
    schema['$defs']['AnswerItem']['properties']['source_ids']['items'] = {
        'type': 'string', 'enum': available_ids}
    if passage_ids is not None:
        schema['$defs']['AnswerItem']['required'].append('evidence_ids')
        # IDs already accompany every passage in the prompt. Repeating the full
        # list in the schema wastes scarce provider input tokens. Parsing below
        # still enforces exact membership in the visible evidence.
        schema['$defs']['AnswerItem']['properties']['evidence_ids'].update(minItems=1, items={'type':'string', 'pattern':r'^S[0-9]+:P[0-9]+$'})
    if topics:
        schema['$defs']['AnswerItem']['required'].append('topic')
        schema['$defs']['AnswerItem']['properties']['topic']['enum'] = topics
    return schema


def parse_answer(payload: dict, chunks: list[dict]) -> tuple[list[dict], bool]:
    try:
        answer = CitedAnswer.model_validate(payload)
    except ValidationError as exc:
        fields = '; '.join('.'.join(map(str, e['loc'])) + ': ' + e['type'] for e in exc.errors(include_input=False)[:3])
        raise InvalidAnswerFormat('Return items with nonempty text and source_ids for each item; use only the allowed source IDs. Invalid fields: ' + fields) from exc
    available = {c['citation_id'] for c in chunks}
    blocks = []
    for i, item in enumerate(answer.items, 1):
        unknown = set(item.source_ids) - available
        if unknown:
            raise InvalidAnswerFormat('Unknown source IDs: ' + ', '.join(sorted(unknown)))
        text = normalize_citations(item.text).strip()
        if set(cited_ids(text)) - set(item.source_ids):
            raise InvalidAnswerFormat('Inline citations must agree with the item source_ids.')
        # Only remove well-formed source labels; mathematical brackets are retained.
        text = re.sub(r'\[S\d+(?:\s*[-\u2010-\u2015,;]\s*S?\d+)*\]', '', text).strip()
        if not text:
            raise InvalidAnswerFormat('An answer item must contain explanatory text, not just citations.')
        if item.topic and item.topic.casefold() not in text.casefold():
            text = item.topic + ': ' + text
        sources = item.source_ids
        if item.evidence_ids:
            # Passage IDs locate evidence; they never prove semantic support.
            from app.services.evidence import visible_passages
            allowed = {p['id'] for c in chunks for p in visible_passages(c)}
            anchors = [p for p in dict.fromkeys(p.strip('[] ') for p in item.evidence_ids) if p in allowed]
            anchored_sources = list(dict.fromkeys(p.split(':')[0] for p in anchors))
            # Keep additional cited records: they may establish a category or
            # qualification not repeated in the defining passage. Correct an
            # entirely mismatched citation, but never drop valid joint evidence.
            expanded = list(dict.fromkeys(sources + anchored_sources))
            if len(expanded) <= 4:
                sources = expanded
        block = {'id': f'B{i}', 'text': text, 'source_ids': sources}
        if item.topic:
            block['topic'] = item.topic
        if item.evidence_ids:
            block['evidence_ids'] = anchors
        blocks.append(block)
    return blocks, answer.insufficient_context


def render_answer(blocks: list[dict]) -> str:
    return '\n\n'.join(b['text'] + ' ' + ' '.join(f'[{sid}]' for sid in b['source_ids']) for b in blocks)
