#!/usr/bin/env python3
"""Run two blinded, randomized Gemini Pro judging passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time

from google import genai
from google.genai import types
from google.oauth2 import service_account


SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=Path("ai-lab-fasa.json"))
    parser.add_argument("--project", default="ai-lab-502500")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--output", type=Path, default=Path("results/generated/judgments.jsonl"))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    questions = {row["question_id"]: row for path in args.questions for row in read_jsonl(path)}
    answers = [row for row in read_jsonl(args.answers) if row.get("status") == "ok"]
    credentials = service_account.Credentials.from_service_account_file(args.credentials, scopes=[SCOPE])
    client = genai.Client(vertexai=True, project=args.project, location="global", credentials=credentials,
                          http_options=types.HttpOptions(api_version="v1"))
    schema = {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer", "minimum": 0, "maximum": 2},
            "groundedness": {"type": "integer", "minimum": 0, "maximum": 2},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 2},
            "unsupported_claim": {"type": "boolean"},
            "rationale": {"type": "string", "maxLength": 160},
        },
        "required": ["correctness", "groundedness", "completeness", "unsupported_claim", "rationale"],
    }
    config = types.GenerateContentConfig(
        temperature=0, max_output_tokens=1024, response_mime_type="application/json",
        response_json_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
    )

    jobs = []
    for answer in answers:
        for judge_pass in (1, 2):
            jobs.append((answer, judge_pass))
    random.Random(20260914).shuffle(jobs)

    def judge(job) -> dict:
        answer, judge_pass = job
        question = questions[answer["question_id"]]
        prompt = f"""Rate the anonymous candidate answer against the reference.
0 = incorrect/absent, 1 = partially correct, 2 = fully correct.
Groundedness means claims are supported by the supplied gold passages or the answer correctly abstains.
Do not reward verbosity. Do not infer which retrieval system produced the answer.

QUESTION: {question['question']}
REFERENCE ANSWER: {question.get('reference_answer', 'Insufficient evidence.')}
ANSWERABLE: {question.get('answerable', True)}
GOLD PASSAGES: {json.dumps(question.get('gold_passages', []), ensure_ascii=False)}
CANDIDATE: {answer['answer']}"""
        response = None
        payload = None
        for attempt in range(5):
            try:
                response = client.models.generate_content(model=args.model, contents=prompt, config=config)
                payload = json.loads(response.text or "")
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        usage = response.usage_metadata
        return {
            "anonymous_id": hashlib.sha256(f"{answer['question_id']}:{answer['mode']}:{judge_pass}".encode()).hexdigest()[:16],
            "question_id": answer["question_id"], "mode": answer["mode"], "judge_pass": judge_pass,
            "judge_model": args.model, **payload,
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {(row["question_id"], row["mode"], row["judge_pass"]) for row in existing}
    pending = [job for job in jobs if (job[0]["question_id"], job[0]["mode"], job[1]) not in completed]
    results = list(existing)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(judge, job) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(jobs)} {result['anonymous_id']}", flush=True)
            args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))


if __name__ == "__main__":
    main()
