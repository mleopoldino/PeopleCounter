run:
	python src/app.py --source 0

test:
	pytest -q

lint:
	ruff check src tests
	black --check src tests
