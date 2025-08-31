lint-fix:
	uv run black src --line-length 80
	uv run isort src
	uv run ruff check src --fix

start:
	@cd src/frontend && npm i && npm run web