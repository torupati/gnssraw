# AGENTS.md

## Repository Overview

**gnss-remote-sensing** is a Python project for GNSS (Global Navigation Satellite System) remote sensing, including RINEX observation file plotting, raw measurement analysis, signal processing, and single-point positioning (SPP).

Key areas of the codebase:

- `app/gnss/` – Core library: signal combinations, ionosphere/troposphere models, ephemeris, coordinates, epoch series, RTCM3 parsing, database, ambiguity resolution, and plotting.
- `app/rnxproc.py`, `app/rnxproc2.py` – RINEX observation processing scripts.
- `app/spp.py` – Single-point positioning entry point.
- `tests/` – Pytest test suite mirroring the `app/gnss/` structure.
- `doc/` – Markdown documentation on GNSS fundamentals and signal processing techniques.
- `sample_data/` – Sample RINEX and measurement files used by tests and scripts.
- `misc/` – Miscellaneous utility scripts (excluded from linting, formatting, and type checking).

## Environment Setup

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency and virtual-environment management. Python 3.10 or later is required.

```bash
# Install uv (if not already installed)
pip install uv

# Install the project and all dev dependencies
uv pip install -e ".[dev]"
# or via Makefile:
make install
```

## Development Commands

All common tasks are available as `make` targets (see `Makefile`) or directly via `uv run`:

| Task | Command |
|---|---|
| Lint (ruff) | `uv run ruff check . --output-format=github` or `make lint` |
| Format check | `uv run ruff format --check .` or `make format` |
| Format fix | `uv run ruff format .` |
| Type check (mypy) | `uv run mypy --explicit-package-bases --exclude '(^|/)misc/' ./app --pretty` or `make type-check` |
| Run tests | `uv run pytest tests/ -v` or `make test` |
| Run tests with coverage | `uv run pytest tests/ -v --cov=app --cov-report=term-missing` or `make test-cov` |
| Run all CI checks | `make ci` |
| Clean artifacts | `make clean` |

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Ruff lint** – `ruff check . --output-format=github`
2. **Ruff format check** – `ruff format --check .`
3. **Mypy type check** – applied to `app/` only; `misc/` is excluded

Tests are not part of the automated CI workflow but should be run locally before opening a PR.

## Code Style and Quality

- **Linter / Formatter**: [ruff](https://docs.astral.sh/ruff/) — configured in `pyproject.toml`. The `misc/` directory is excluded.
- **Type checker**: [mypy](https://mypy.readthedocs.io/) — strict settings applied to `app/`; `misc/` is excluded.
- **Pre-commit hooks** (`.pre-commit-config.yaml`): ruff lint+format, trailing-whitespace, end-of-file-fixer, YAML syntax check, large-file guard, and mypy.

Install pre-commit hooks with:

```bash
uv run pre-commit install
```

## Testing

Tests live in `tests/` and are discovered by pytest (see `pytest.ini`).

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/gnss/test_ephemeris.py -v

# Run only unit tests
uv run pytest tests/ -v -m unit
```

Available markers: `slow`, `integration`, `unit`.

## Notes for Agents

- Always run lint, format check, and type check after modifying files under `app/`.
- Do **not** apply linting or type checking to files under `misc/` — they are intentionally excluded.
- Keep `tests/` in sync with `app/`; add or update tests when changing library behaviour.
- Do not commit large binary or data files; the pre-commit hook will block them.
- Use `uv run` to execute tools to ensure the correct virtual environment is used.
