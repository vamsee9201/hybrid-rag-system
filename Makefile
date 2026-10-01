.PHONY: install test lint web corpus index preflight run

install:
	python3 -m pip install -e '.[dev]'
	cd web && npm install

test:
	pytest -q
	cd web && npm test

lint:
	ruff check app scripts tests
	cd web && npm run build

corpus:
	python3 scripts/materialize_corpus.py --lean-rag-root ../lean-rag

index:
	python3 scripts/build_bm25.py

preflight:
	python3 scripts/preflight_cost.py

run:
	uvicorn app.main:app --reload
