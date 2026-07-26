.PHONY: install lint type-check test check fix clean

install:
	pip install -e ".[dev]"
	pre-commit install

# Non-mutating verification: fails if lint or formatting is off. Used by `check` and CI.
lint:
	ruff check .
	ruff format --check .

type-check:
	mypy src/

test:
	pytest

# The gate: lint + type-check + test. CI and pre-commit invoke this.
check: lint type-check test

# Local convenience: apply autofixes and format in place.
fix:
	ruff check . --fix
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
