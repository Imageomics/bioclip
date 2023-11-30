install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-training:
	python -m pip install -r requirements-training.txt

lint:
	find src/evaluation -iname '*.py' | xargs ruff check
