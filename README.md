
## 🚀 Development Setup

### 1. Local Installation with Editable Mode
Navigate to the project root directory containing `pyproject.toml` and run the installation command using the local virtual environment's explicit Python binary:

<div style="padding: 15px; border-left: 5px solid #ff4757; background-color: #ffffff; color: #2f3542; border-radius: 4px; margin: 15px 0;">
    <strong>⚠️ IMPORTANT</strong><br>
    Executing installation directly through <code>.venv/bin/python -m pip</code> <strong>safeguards the global system</strong> by ensuring packages are strictly isolated inside the active virtual environment.
</div>



```bash
# Move to the folder containing pyproject.toml
cd rag_lib

# Safely install package in editable mode with dev tools using the explicit virtual environment binary
.venv/bin/python -m pip install --editable ".[dev]"
```

*   `--editable`: Links the `rag_lib` package source folder directly to the environment so code changes take effect instantly.
*   `".[dev]"`: Instructs `pip` to install the optional development dependencies (such as `pytest` and `pre-commit`) defined in `pyproject.toml`.


### 2. Set Up Git Hooks
The `pre-commit` framework is utilized to maintain code quality and prevent broken code from being committed. Initialize the hooks by running:

```bash
pre-commit install
```

Once installed, the hooks automatically run linting and validation checks during execution of `git commit`.
## 💻 VS Code Integration
The project includes a pre-configured `.vscode/settings.json` file to initialize the native **VS Code Testing Panel** automatically.
