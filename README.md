# Hybrid RAG Evaluation

A production-deployed comparison of lexical, semantic, and hybrid retrieval over
500 searchable U.S. government publications. The application sends one question
through each selected retrieval pipeline and uses the same Gemini Flash model,
prompt, evidence limit, and answer format so retrieval strategy remains the
primary experimental variable.

**[Open the live demo](https://govlens-265050340558.us-central1.run.app)**

![Hybrid RAG Evaluation overview with lexical, semantic, and hybrid retrieval cards](docs/screenshots/hybrid-rag-overview.png)

- **Lexical:** SQLite FTS5/BM25 for exact terms, names, dates, and citations.
- **Semantic:** exact cosine search over 768-dimensional Vertex AI embeddings.
- **Hybrid:** reciprocal-rank fusion of the BM25 and dense rankings.

The interface is intentionally single-turn: it retains no conversational
context, offers 30 curated GovInfo questions as a rotating set of three
suggestions, streams each answer independently, and exposes the retrieved
evidence behind every result. Users can compare all three systems or run one
retriever to reduce generation cost.

![Completed Hybrid RAG comparison showing cited answers, latency, source count, and estimated cost](docs/screenshots/hybrid-rag-comparison-results.png)

The public demo allows five requests per minute and 20 generated answers per
client each day. Compare mode uses three answers per question, while Single RAG
mode uses one. A shared $0.50 daily model budget protects the project from
unexpected usage. Daily limits reset at midnight UTC.

## What this project demonstrates

- A controlled RAG experiment with a shared corpus, chunking strategy, generator,
  prompt, context limit, and citation contract.
- Exact in-memory vector search over 109,226 normalized Vertex AI embeddings,
  without the fixed cost of a managed vector database.
- Concurrent SSE streaming with isolated failures, evidence inspection, and
  anonymous answer-preference feedback.
- Reproducible retrieval and answer evaluation with confidence intervals,
  blinded judging, token usage, latency, and measured cost.
- A cost-capped public deployment with least-privilege runtime identity,
  cross-instance quotas, structured logging, and no stored questions or answers.

## Architecture

| Layer | Technology | Responsibility |
|---|---|---|
| Interface | React and TypeScript | Sends a single question to one or all retrieval modes and streams results independently. |
| API | FastAPI with server-sent events | Coordinates retrieval, generation, citations, feedback, and partial failures. |
| Lexical retrieval | SQLite FTS5/BM25 | Retrieves the top 30 passages using exact terms, names, dates, and phrases. |
| Semantic retrieval | Vertex AI embeddings and NumPy | Embeds the query and performs exact cosine search over memory-mapped vectors. |
| Hybrid retrieval | Reciprocal-rank fusion | Combines the BM25 and semantic rankings with equal weights and `k=60`. |
| Context selection | Shared passage filter | Returns five passages per system with no more than two passages from one document. |
| Answer generation | Gemini Flash | Generates three parallel, evidence-grounded answers with document and page citations. |
| Runtime artifact | Cloud Storage | Stores the versioned SQLite database, embedding matrix, chunk IDs, and manifests loaded at startup. |
| Usage limits | Firestore | Maintains shared per-minute, daily-client, and daily-budget counters across Cloud Run instances. |

## Corpus and configuration

| Component | Setting |
|---|---|
| Corpus | 500 extractable GovInfo PDFs |
| Pages | 56,920 |
| Extracted characters | 486,287,346 |
| Chunks | 109,226 |
| Chunking | 350 words, 50-word overlap, page bounded |
| Dense model | `gemini-embedding-001` |
| Dense representation | normalized 768-dimensional float32 prefix |
| Dense search | exact matrix cosine search |
| BM25 | SQLite FTS5 `unicode61` |
| Candidate depth | 30 per retriever |
| Final evidence | 5 chunks, maximum 2 per document |
| Hybrid | equal-weight reciprocal-rank fusion, `k=60` |
| Generator | `gemini-3.8-flash` |
| Generation | temperature 0, thinking off, maximum 300 tokens |

The corpus contains 500 searchable GovInfo PDFs spanning 56,920 pages. Page text
is split into 109,226 deterministic, page-bounded chunks of approximately 350
words with a 50-word overlap. Every chunk retains its document ID, title, page,
source URL, and citation metadata.

## Evaluation protocol

The evaluation uses 100 source-grounded questions drawn from the GovInfo corpus.

| Question type | Count |
|---|---:|
| Direct factual | 40 |
| Numeric or date | 20 |
| Multi-passage | 20 |
| Cross-document | 10 |
| Unanswerable | 10 |

Every question includes a reference answer and exact supporting document and
page labels. Each question is run through BM25, semantic, and hybrid retrieval
using the same generation settings. Retrieved passages are frozen before answer
generation so retries cannot change the evidence.

Retrieval reporting includes all-gold recall@5, passage recall, MRR, nDCG@5,
cross-document recall, and latency. Answer reporting includes deterministic
checks, exact citation validation, abstention accuracy, token F1, two randomized
blinded Gemini 3.1 Pro judging passes, judge stability, cost, and latency. Paired
differences use 10,000 bootstrap samples.

### Measured retrieval results

Metrics below use the 90 answerable questions; latency is local warm retrieval
and excludes the shared Vertex query-embedding call.

| Retriever | All-gold R@5 | Passage R@5 | MRR | nDCG@5 | Mean retrieval |
|---|---:|---:|---:|---:|---:|
| BM25 | 62.2% | 68.9% | **0.692** | **0.721** | 160.5 ms |
| Dense | 45.6% | 53.9% | 0.511 | 0.513 | **4.8 ms** |
| Hybrid | **62.2%** | **70.0%** | 0.667 | 0.706 | 165.5 ms |

Hybrid recovered the most gold passages, but BM25 ranked the first useful result
and the top-five list slightly better. Dense retrieval alone was fastest and
substantially weaker on this terminology-heavy government corpus.

### Measured answer results

The score is the mean of correctness, groundedness, and completeness, normalized
to 0–1 and averaged across two randomized passes of the same blinded judge.

| System | Completed | Judge score | Token F1 | Correct abstention | Valid citation syntax | Mean generation |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 100/100 | **0.637** | **0.229** | **71.0%** | 100% | 3.49 s |
| Dense | 100/100 | 0.536 | 0.199 | 61.0% | 100% | **3.30 s** |
| Hybrid | 99/100 | 0.617 | 0.223 | 69.7% | 100% | 4.74 s |

BM25 and Hybrid were statistically inconclusive on judge score: Hybrid minus
BM25 was -0.017 (95% paired bootstrap CI -0.073 to 0.039; Holm-adjusted
`p=0.556`). Hybrid beat Dense by 0.086 (CI 0.026 to 0.149; adjusted `p=0.010`).
Dense trailed BM25 by 0.101 (CI -0.178 to -0.027; adjusted `p=0.014`). Judge-pass
exact agreement was 91.0% for BM25 and Dense and 85.9% for Hybrid. One Hybrid
answer is recorded as a repeated Vertex `429 RESOURCE_EXHAUSTED` failure.

Machine-readable headline results are in [`results/summary.json`](results/summary.json).
Multi-passage and cross-document questions were the clearest common failure
area; unanswerable questions were easiest for all three systems.

## Conclusion

| System | Advantages | Tradeoffs | Best fit |
|---|---|---|---|
| BM25 | Strongest judged answer score, MRR, and nDCG@5; excellent with exact terminology, identifiers, dates, and citations; simple and inexpensive to operate. | Can miss relevant passages when the question and source use different vocabulary. | Structured government, legal, technical, and policy collections where exact language matters. |
| Dense | Handles paraphrases and meaning across different wording; very fast local vector search after the query is embedded. | Requires an embedding call and vector storage; produced the weakest recall and answer scores on this terminology-heavy corpus. | Collections where users describe concepts differently from the source documents. |
| Hybrid | Highest passage recall; combines exact matching with semantic coverage; more resilient when either retriever misses useful evidence. | Adds retrieval complexity and had the highest end-to-end latency; its answer score was not significantly better than BM25. | General-purpose search where missing a relevant passage is more costly than additional latency. |

**Best overall for this corpus: BM25.** It produced the strongest answer score
and ranking quality with lower complexity than Hybrid. Hybrid is the better
choice when retrieval recall is the priority because it found the largest share
of supporting passages. Dense retrieval is valuable as a complementary signal,
but the evaluation does not support using it alone for this collection.

## Local development

Prerequisites: Python 3.11+, Node 20.19+, prepared GovInfo source data, and GCP
credentials for paid operations.

```bash
make install
make corpus
make index
make preflight
cd web && npm run build
uvicorn app.main:app --reload
```

Paid embedding is deliberately double-gated. It requires a successful preflight
report and the explicit `--execute` acknowledgement:

```bash
python3 scripts/run_vertex_batch.py \
  --bucket ai-lab-502500-hybrid-rag \
  --output data/indexes/dense \
  --execute
```

Package, upload, and deploy an immutable index:

```bash
python3 scripts/package_artifact.py
scripts/upload_artifact.sh data/artifacts/VERSION.tar.gz
ARTIFACT_GCS_URI=gs://BUCKET/artifacts/VERSION/runtime.tar.gz \
ARTIFACT_SHA256=SHA256 scripts/deploy.sh
```

Run the frozen evaluation:

```bash
python3 scripts/prepare_benchmark.py
python3 scripts/evaluate_retrieval.py \
  --artifact data/artifacts/VERSION \
  --questions data/generated/benchmark/*.jsonl
python3 scripts/run_answers.py \
  --questions data/generated/benchmark/*.jsonl \
  --retrieval results/generated/retrieval.jsonl
python3 scripts/judge_answers.py \
  --questions data/generated/benchmark/*.jsonl \
  --answers results/generated/answers.jsonl
python3 scripts/summarize_evaluation.py \
  --questions data/generated/benchmark/*.jsonl \
  --answers results/generated/answers.jsonl \
  --judgments results/generated/judgments.jsonl
```

## API

- `GET /api/health`: readiness and model/index versions.
- `GET /api/config`: public corpus, mode, and limit configuration.
- `POST /api/chat`: validated, single-turn SSE question answering for one or all retrievers.
- `POST /api/feedback`: anonymous winner selection and reason tags.
