lint: fmt
	ruff check src/imageomics scripts/

fmt:
	ruff format src/imageomics scripts/

test: lint
	pytest src/imageomics/
