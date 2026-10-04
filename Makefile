# ChainGuard Makefile

.PHONY: ml-test contracts-test serve lint

# --- ML ---
ml-test:
	cd ml && python -m pytest tests/ -v

ml-build-graph:
	python ml/src/features/build_graph.py

ml-download:
	python ml/src/ingest/download_elliptic.py

ml-splits:
	python ml/src/eval/temporal_split.py

# --- Contracts ---
contracts-test:
	cd contracts && npx hardhat test

# --- Service ---
serve:
	cd service && uvicorn app.main:app --reload --port 8000

# --- Lint ---
lint:
	cd ml && python -m ruff check src/ tests/
