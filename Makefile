.PHONY: setup check download notebook batch test clean

setup:
	poetry install

check:
	python scripts/setup_check.py

download:
	python scripts/download_dataset.py --config config.yaml

notebook:
	jupyter lab notebooks/01_explore_and_tune.ipynb

batch:
	python scripts/run_batch.py --config config.yaml

test:
	poetry run pytest tests/ -v --cov=src

clean:
	rm -rf .tmp_opensim .download_cache
