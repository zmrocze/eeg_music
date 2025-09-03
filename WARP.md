# guide

# Development Environment

This project uses a **Nix + devenv + uv** stack for reproducible development:

## Important Rules
1. **Always run Python through uv**: i.e. `uv run script.py`
2. **Never pip install**: Dependencies are managed via `pyproject.toml` or `devenv.nix`
3. **No shebangs in Python files**: python files NEVER start with "#!/usr/bin/env"

## Essential Commands

Important files:
├── notes/                  # Documentation and research notes
│   ├── datasets.md        # Dataset overview
│   ├── bcmi_datasets_explainer.md # Detailed BCMI event documentation
