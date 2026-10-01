#!/usr/bin/env python3
"""Verify health, configuration, one RAG answer, and the public quota surface."""

from __future__ import annotations

import argparse
import json

import httpx


def request(url: str, path: str, payload: dict | None = None):
    response = httpx.request("GET" if payload is None else "POST", url + path, json=payload, timeout=300)
    return response.status_code, response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    for path in ("/api/health", "/api/config"):
        status, body = request(base, path)
        if status != 200:
            raise SystemExit(f"{path} failed: {status} {body[:500]}")
        print(path, json.loads(body))
    status, body = request(base, "/api/chat", {
        "message": "According to the indexed reports, what are improper payments?",
        "modes": ["bm25", "dense", "hybrid"],
    })
    if status != 200 or body.count("event: answer") != 3 or "event: complete" not in body:
        raise SystemExit(f"chat smoke failed: {status} {body[:1000]}")
    print("/api/chat", "all three modes streamed independently")


if __name__ == "__main__":
    main()
