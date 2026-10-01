from __future__ import annotations

from app.models import RetrievedPassage


SYSTEM_PROMPT = """You answer questions about United States government publications.
Use only the supplied evidence. Cite every material factual claim using exactly
[DOCUMENT_ID, p. N]. If the evidence does not support an answer, respond exactly
"Insufficient evidence." Never invent a citation, URL, policy, or page number.
Keep the answer concise and do not give legal, medical, or financial advice."""


def evidence_prompt(message: str, passages: list[RetrievedPassage]) -> str:
    evidence = "\n\n".join(
        f"[{item.citation.document_id}, p. {item.citation.page}]\n{item.text}"
        for item in passages
    )
    return f"EVIDENCE\n{evidence}\n\nQUESTION\n{message}"
