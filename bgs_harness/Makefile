.PHONY: setup convert run clean help

# Environment configuration
VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

# Vendored Frigate NVR (motion detector algorithms, used unmodified,
# pinned by the frigate git submodule)
FRIGATE_ROOT = ../frigate
FRIGATE_NAME = frigate
FRIGATE_VERSION = 0.18.0

help:
	@echo "Available commands:"
	@echo "  make setup   - Create virtualenv, install dependencies, vendor Frigate"
	@echo "  make convert - Convert input_videos/ to the configured stream format"
	@echo "  make run     - Run the pybgs benchmark pipeline"
	@echo "  make clean   - Remove generated output videos, logs, and venv"

# 1. Setup virtual environment and dependencies using uv
setup:
	@echo "Setting up environment using uv..."
	$(UV) venv --allow-existing $(VENV)
	$(UV) pip install -e .
	@echo "Preparing vendored Frigate NVR submodule (downloads full repo on first run)..."
	git -C .. submodule update --init $(FRIGATE_NAME)
	@if [ -d "$(FRIGATE_ROOT)/frigate" ]; then \
		echo "Writing $(FRIGATE_ROOT)/frigate/version.py..."; \
		echo "VERSION = \"$(FRIGATE_VERSION)-$$(git -C $(FRIGATE_ROOT) rev-parse --short HEAD)\"" > $(FRIGATE_ROOT)/frigate/version.py; \
	fi

# 2. Convert input videos to the configured stream format
convert:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtualenv not found. Running 'make setup' first..."; \
		$(MAKE) setup; \
	fi
	@echo "Converting input videos to the configured stream format..."
	$(PYTHON) convert_videos.py

# 3. Run the main evaluation harness
run:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtualenv not found. Running 'make setup' first..."; \
		$(MAKE) setup; \
	fi
	@if [ -z "$$(ls -A processed_videos 2>/dev/null)" ]; then \
		echo "No processed videos found. Running 'make convert' first..."; \
		$(MAKE) convert; \
	fi
	@echo "Running background subtraction harness..."
	$(PYTHON) harness.py

# 4. Clean generated artifacts
clean:
	@echo "Cleaning up generated files and virtual environment..."
	rm -rf outputs/*
	rm -rf processed_videos/*
	rm -rf config/*.xml
	rm -rf $(VENV)
	rm -rf *.egg-info
	@echo "Cleanup complete."
