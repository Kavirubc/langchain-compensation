# Publishing to PyPI Guide

## Prerequisites

1. **Create PyPI Account**: https://pypi.org/account/register/
2. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Steps to Publish

### 1. Update Package Metadata

Edit `pyproject.toml`:
- Update `version` (e.g., "0.1.0" → "0.1.1")
- Add your name and email in `authors`
- Update GitHub URLs with your username

### 2. Build the Package

```bash
cd langchain-compensation
python -m build
```

This creates files in `dist/`:
- `langchain_compensation-0.1.0-py3-none-any.whl`
- `langchain_compensation-0.1.0.tar.gz`

### 3. Test on TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ langchain-compensation
```

### 4. Publish to Real PyPI

```bash
python -m twine upload dist/*
```

Enter your PyPI credentials when prompted.

### 5. Verify Installation

```bash
pip install langchain-compensation
```

## Updating the Package

When you make changes:

1. Update version in `pyproject.toml`
2. Delete old `dist/` folder
3. Rebuild: `python -m build`
4. Upload: `python -m twine upload dist/*`

## Local Development Installation

To test locally before publishing:

```bash
cd langchain-compensation
pip install -e .
```

The `-e` flag installs in "editable" mode, so changes take effect immediately.

## Version Numbering

Follow semantic versioning (semver):
- **0.1.0** → **0.1.1**: Bug fixes
- **0.1.0** → **0.2.0**: New features (backward compatible)
- **0.1.0** → **1.0.0**: Breaking changes

## Testing Before Release

```bash
# Run tests
pytest tests/

# Check code formatting
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```
