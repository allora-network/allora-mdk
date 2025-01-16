.PHONY: lint format test clean train eval pyreqs fullreqs package package-all node-env run compose compose-clean $(MODEL_DIRS)

# Development tools
lint:
	find . -name "*.py" | xargs pylint --rcfile=.pylintrc

format:
	black .

test:
	pytest -m unittest discover -s tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage \
		trained_models/ logs/ test_results/

# Model operations
train:
	python train.py

eval:
	python test.py

# Packaging
MODEL_DIRS := $(shell find trained_models -type d -maxdepth 1 -mindepth 1 -exec basename {} \;)
package-all: $(addprefix package-, $(MODEL_DIRS))

package-%:
	python package_model_worker.py $*

# Requirements management
pyreqs:
	pipdeptree --freeze --warn silence | grep -E '^[a-zA-Z0-9\-]+' > requirements.txt

fullreqs:
	pip freeze > requirements.txt

# Worker operations
node-env:
	python allonode-data/generate_envfile.py

run:
	MODEL=$(MODEL) uvicorn main:app --reload --port 8000

# Docker operations
build:
	docker build -f docker/Dockerfile -t allora-worker .

compose:
	MODEL=$(MODEL) docker compose -f docker/dev.docker-compose.yaml up

compose-clean:
	MODEL=$(MODEL) docker compose -f docker/dev.docker-compose.yaml up --build --force-recreate
