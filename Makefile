# Courtesy of luminovo.ai
# Automates your development workflows

# ----------Environment Variables---------

# set environment variables to be used in the commands here

# ----------------Commands----------------

# change the 20 value in printf to adjust width
# Use ' ## some comment' behind a command and it will be added to the help message automatically
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: build-docs ## Build the docker container and docs
	docker build --file docker/Dockerfile --tag luminovo/pcbauto:latest .

check: ## Run all static checks (like pre-commit hooks)
	pre-commit run --all-files

clean:
	-find . -type f -name "*.py[co]" -delete
	-find . -type d -name "__pycache__" -delete
	-find . -type d -name ".pytest_cache" -exec rm -r "{}" \;

build-docs:
	docker run --rm -it -v `pwd`/docs:/docs squidfunk/mkdocs-material build
	
docs: build-docs ## Serve all docs
	docker run --rm -p 8000:8000 -v `pwd`/docs:/docs squidfunk/mkdocs-material

unit-test: clean
	poetry run pytest --cov=pcbauto tests/

test: unit-test ## Run all tests

dev-install: ## Install all the packages in the local python environment for development
	# there is a bug with pip 19
	poetry run pip install --upgrade pip==18.1
	poetry install


# --------------Configuration-------------

.ONESHELL: ; # recipes execute in same shell
.NOTPARALLEL: ; # wait for this target to finish
.EXPORT_ALL_VARIABLES: ; # send all vars to shell

.PHONY: docs all # All targets are accessible for user
.DEFAULT: help # Running Make will run the help target

MAKEFLAGS += --no-print-directory # dont add message about entering and leaving the working directory
