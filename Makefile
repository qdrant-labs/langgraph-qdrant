.PHONY: test test-local test-cluster start-qdrant stop-qdrant start-cluster stop-cluster lint format type

TEST ?= tests

start-qdrant:
	docker compose -f tests/compose-qdrant.yml up -V --force-recreate --wait

stop-qdrant:
	docker compose -f tests/compose-qdrant.yml down

start-cluster:
	docker compose -f tests/compose-cluster.yml up -V --force-recreate --wait

stop-cluster:
	docker compose -f tests/compose-cluster.yml down

test:
	uv run pytest $(TEST)

test-cluster:
	QDRANT_URL=http://localhost:7002 QDRANT_GRPC_PORT=7012 uv run pytest $(TEST)

test-local:
	QDRANT_SKIP_SERVER=true uv run pytest $(TEST)

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

type:
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests
