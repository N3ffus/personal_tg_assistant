UV ?= uv
IMAGE_NAME ?= personal-ai-assistant
CONTAINER_NAME ?= personal-ai-assistant

.PHONY: install run lint format typecheck test check docker-build docker-run docker-remove docker-stop docker-logs

install:
	$(UV) sync --all-groups

run:
	$(UV) run python -m src.main

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest

check: lint typecheck test

docker-build:
	docker build --tag $(IMAGE_NAME) .

docker-run: docker-build docker-remove
	docker run --detach --name $(CONTAINER_NAME) --env-file .env --restart unless-stopped $(IMAGE_NAME)

docker-remove:
	-docker rm --force $(CONTAINER_NAME)

docker-stop:
	docker stop $(CONTAINER_NAME)

docker-logs:
	docker logs --follow $(CONTAINER_NAME)
