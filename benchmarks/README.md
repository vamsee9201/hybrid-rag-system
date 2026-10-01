# Benchmark sources

Evaluation uses two frozen, source-grounded benchmark sets from the sibling
`lean-rag` research repository. The source files remain there to avoid copying
the 21 GB research workspace into this deployable repository.

- Legacy replication: `../lean-rag/data/benchmark/questions.jsonl`
- Confirmatory candidate pool: `../lean-rag/data/confirmatory/benchmark/questions.jsonl`

`scripts/prepare_benchmark.py` selects a document-disjoint 50-question
confirmatory sample with the same 20/10/10/5/5 category distribution as the
legacy benchmark. The generated benchmark and its source hashes are written to
`data/generated/benchmark/`, which is excluded from Git until it is sealed.
