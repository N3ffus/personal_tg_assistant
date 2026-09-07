UV ?= uv
IMAGE_NAME ?= personal-ai-assistant
CONTAINER_NAME ?= personal-ai-assistant

.PHONY: install run lint format typecheck test test-integration eval eval-report check docker-build docker-run docker-remove docker-stop docker-logs knowledge-demo neo4j-up neo4j-down

install:
	$(UV) sync --all-groups

run:
	$(UV) run python -m src.main

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

typecheck:
	$(UV) run --group eval mypy src tests evals scripts

test:
	$(UV) run pytest --cov=src --cov-branch --cov-report=term-missing

test-integration:
	$(UV) run pytest tests/integration -m integration -v

eval:
	$(UV) run --group eval pytest evals --run-llm-evals -m llm_eval -v --junitxml=eval-results/junit.xml -o junit_family=xunit1

eval-report:
	$(UV) run --group eval python -m scripts.eval_report

check: lint typecheck test

docker-build:
	docker build --tag $(IMAGE_NAME) .

docker-run: docker-build docker-remove
	docker run --detach --name $(CONTAINER_NAME) --env-file .env --publish 8080:8080 --mount type=volume,source=$(CONTAINER_NAME)-data,target=/app/data --restart unless-stopped $(IMAGE_NAME)

docker-remove:
	-docker rm --force $(CONTAINER_NAME)

docker-stop:
	docker stop $(CONTAINER_NAME)

docker-logs:
	docker logs --follow $(CONTAINER_NAME)

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose stop neo4j

knowledge-demo:
	$(UV) run python -m scripts.knowledge_demo
