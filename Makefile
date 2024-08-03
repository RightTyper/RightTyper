all:
	black -l 79 righttyper && mypy righttyper && ruff check righttyper
