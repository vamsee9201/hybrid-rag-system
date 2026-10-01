#!/usr/bin/env python3
"""Generate frozen benchmark answers for every retriever with Gemini Flash."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time

from google import genai
from google.genai import types
from google.oauth2 import service_account

from app.prompts import SYSTEM_PROMPT


SCOPE = "https://www.googleapis.com/auth/cloud-platform"
CITATION = re.compile(r"\[([^\],]+),\s*p\.\s*(\d+)\]")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=Path("ai-lab-fasa.json"))
    parser.add_argument("--project", default="ai-lab-502500")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--output", type=Path, default=Path("results/generated/answers.jsonl"))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    questions = {row["question_id"]: row for path in args.questions for row in read_jsonl(path)}
    retrieval = read_jsonl(args.retrieval)
    credentials = service_account.Credentials.from_service_account_file(args.credentials, scopes=[SCOPE])
    client = genai.Client(vertexai=True, project=args.project, location="global", credentials=credentials,
                          http_options=types.HttpOptions(api_version="v1"))
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, temperature=0, max_output_tokens=300,
        thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    def run(record: dict) -> dict:
        question = questions[record["question_id"]]
        evidence = "\n\n".join(
            f"[{p['citation']['document_id']}, p. {p['citation']['page']}]\n{p['text']}"
            for p in record["passages"][:5]
        )
        prompt = f"EVIDENCE\n{evidence}\n\nQUESTION\n{question['question']}"
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=args.model, contents=prompt, config=config)
            answer = (response.text or "").strip()
            if not answer:
                raise ValueError("empty answer")
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            supplied = {
                (p["citation"]["document_id"], int(p["citation"]["page"])) for p in record["passages"][:5]
            }
            cited = {(doc.strip(), int(page)) for doc, page in CITATION.findall(answer)}
            return {
                "question_id": record["question_id"], "category": question["category"],
                "mode": record["mode"], "status": "ok", "answer": answer,
                "validated_citations": sorted([list(pair) for pair in cited & supplied]),
                "invalid_citations": sorted([list(pair) for pair in cited - supplied]),
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "estimated_cost_usd": (input_tokens * 0.75 + output_tokens * 3.75) / 1_000_000,
                "generation_ms": (time.perf_counter() - started) * 1_000,
                "model": args.model,
            }
        except Exception as exc:
            return {
                "question_id": record["question_id"], "category": question["category"],
                "mode": record["mode"], "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "model": args.model,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {(row["question_id"], row["mode"]) for row in existing if row.get("status") == "ok"}
    pending = [row for row in retrieval if (row["question_id"], row["mode"]) not in completed]
    results = list(existing)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run, row) for row in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(retrieval)} {result['question_id']} {result['mode']} {result['status']}", flush=True)
            args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))


if __name__ == "__main__":
    main()
