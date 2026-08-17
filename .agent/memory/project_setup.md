---
name: Project setup
description: How the project venv and mjlab dependency are configured
type: project
originSessionId: a6d0896f-1841-49f6-a2b9-837c74cd984b
---
`mjlab/` is a git submodule installed editably into `.venv` via `[tool.uv.sources]` in root `pyproject.toml`. Run `uv sync` to install/update. Never `pip install mjlab` — it would shadow the local source.

Verify: `.venv/bin/python -c "import mjlab; print(mjlab.__file__)"` should print `.../mjlab/src/mjlab/__init__.py`.

**Why:** Previously .venv had mjlab 1.2.0 from PyPI while the submodule was 1.3.0, causing runtime import errors. Fixing this required changing `build-backend` from `setuptools.backends.legacy:build` to `setuptools.build_meta` and adding `[tool.uv.sources]`.

**How to apply:** If something breaks with mjlab imports, check which version is active first.
