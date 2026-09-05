UV ?= uv
IMAGE_NAME ?= personal-ai-assistant
CONTAINER_NAME ?= personal-ai-assistant

.PHONY: install run lint format typecheck test eval check docker-build docker-run docker-remove docker-stop docker-logs

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
	$(UV) run --group eval mypy src tests evals

test:
	$(UV) run pytest --cov=src --cov-branch --cov-report=term-missing

eval:
	$(UV) run --group eval pytest evals --run-llm-evals -m llm_eval -v --junitxml=eval-results/junit.xml -o junit_family=xunit1

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
