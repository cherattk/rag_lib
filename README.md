# rag_lib

A Python library for RAG (Retrieval-Augmented Generation) applications, featuring transformer embeddings and T5 inference modules.

## 🚀 Development Setup

### 1. Local Installation with Editable Mode
Navigate to the project root directory containing `pyproject.toml` and run the installation command using the local virtual environment's explicit Python binary:

```bash
# Move to the folder containing pyproject.toml
cd rag_lib

# Safely install package in editable mode with dev tools using the explicit virtual environment binary
.venv/bin/python -m pip install --editable ".[dev]"
```

*   `--editable`: Links the `rag_lib` package source folder directly to the environment so code changes take effect instantly.
*   `".[dev]"`: Instructs `pip` to install the optional development dependencies (such as `pytest` and `pre-commit`) defined in `pyproject.toml`.

> [!IMPORTANT]
> Executing installation directly through `.venv/bin/python -m pip` **safeguards the global system** by ensuring packages are strictly isolated inside the active virtual environment.

### 2. Set Up Git Hooks
The `pre-commit` framework is utilized to maintain code quality and prevent broken code from being committed. Initialize the hooks by running:

```bash
pre-commit install
```

Once installed, the hooks automatically run linting and validation checks during execution of `git commit`.

---

## 🧪 Testing

The `pytest` framework handles the test suite execution. Testing commands are executed as follows:

```bash
# Run the entire test suite
pytest

# Re-run only the tests that failed during the last run
pytest --lf

# Stop immediately on the first test failure
pytest -x

# Drop into the debugger on a test failure
pytest --pdb
```

### 💻 VS Code Integration
The project includes a pre-configured `.vscode/settings.json` file to initialize the native **VS Code Testing Panel** automatically. 

Execution steps:
1. Open the project folder in Visual Studio Code.
2. Select the active virtual environment as the Python interpreter (`Ctrl+Shift+P` / `Cmd+Shift+P` -> *Python: Select Interpreter*).
3. Click the **Beaker/Testing Icon** on the left Activity Bar to discover, run, and debug tests visually directly within the editor.

---

## 📌 Versioning & Releases

This project utilizes **single-source dynamic versioning** powered by `setuptools-scm`. The package version is automatically derived from **Git tags**, eliminating manual version string updates inside repository files.

### Release Workflow

To publish a new version of the package, the following exact sequence is required:

1. **Commit changes:** Ensure the working tree is clean.
   ```bash
   git add .
   git commit -m "feat: add new embedding features"
   ```

2. **Create an annotated Git tag:** Point a new semantic version tag to the target commit.
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   ```

3. **Push to GitHub:** Push both the branch commits and the corresponding tags to the remote repository.
   ```bash
   git push origin main
   git push origin v1.2.0
   ```

### Managing Git Tags

```bash
# List all existing tags
git tag

# List tags with their annotation messages
git tag -n

# Sort tags correctly by Semantic Versioning order
git tag -l --sort=v:refname
```
