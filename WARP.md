# guide

# Development Environment

This project uses a **Nix + devenv + uv** stack for reproducible development:

Important files:
├── notes/                  # Documentation and research notes
│   ├── datasets.md        # Dataset overview
│   ├── bcmi_datasets_explainer.md # Detailed BCMI event documentation

## Important Rules
1. **Always run Python through uv**: i.e. `uv run script.py`
2. **Never pip install**: Dependencies are managed via `pyproject.toml` or `devenv.nix`
3. **No shebangs in Python files**: python files NEVER start with "#!/usr/bin/env"
4. **run tests**  from file <filename> with: `uv run pytest -k <filename>` and all tests with `uv run pytest`

## Style

1. write short, concise code. if possible avoid declaring every step of calculation as new variable, prefer bigger expression. Prefer functional, pure style.
2. never `import` anywhere other than at the top of a file