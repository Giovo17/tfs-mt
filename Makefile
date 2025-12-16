.PHONY: install
install:
	@echo "Creating virtual environment"
	@uv sync
	@uv run prek install

.PHONY: check
check:
	@echo "Checks"
	@uv lock --locked
	@uv run prek run -a
	@uv run ty check

.PHONY: test
test:
	@echo "Running pytes"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build
	@echo "Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build:
	@echo "Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish:
	@echo "Publishing to PyPi"
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish

.PHONY: docs-test
docs-test:
	@echo "Building doc"
	@uv run zensical build -s

.PHONY: docs
docs:
	@echo "Serve doc"
	@uv run zensical serve
