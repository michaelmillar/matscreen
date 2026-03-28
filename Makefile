.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
