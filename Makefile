.PHONY: help docker-up docker-down docker-logs docker-clean db-setup db-shell

help:
	@echo "Available commands:"
	@echo "  make docker-up     - Start PostgreSQL with pgvector"
	@echo "  make docker-down   - Stop PostgreSQL"
	@echo "  make docker-logs   - View PostgreSQL logs"
	@echo "  make docker-clean  - Remove containers and volumes"
	@echo "  make db-setup      - Initialize database schema"
	@echo "  make db-shell      - Open PostgreSQL shell"

docker-up:
	docker-compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	@echo "PostgreSQL with pgvector is running on port 5433"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f postgres-vector

docker-clean:
	docker-compose down -v
	@echo "Removed containers and volumes"

db-setup: docker-up
	@echo "Setting up database..."
	cp .env.docker .env
	python scripts/setup_database.py

db-shell:
	docker exec -it cashly-ai-postgres psql -U cashly_ai -d cashly_ai_vectors