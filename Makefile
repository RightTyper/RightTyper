all: black ruff pyright

black:
	black -l 79 righttyper

ruff:
	ruff check righttyper

pyright:
	pyright righttyper
