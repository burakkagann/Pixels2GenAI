# Contributing to Pixels2GenAI

Thank you for your interest in contributing to Pixels2GenAI! This project is an open-source educational curriculum that teaches generative art and AI through progressive modules.

## How to Contribute

### Reporting Issues

- Use the [Issues page](https://github.com/burakkagann/Pixels2GenAI/issues) to report bugs, broken links, or unclear instructions.
- Include the module number and exercise name (e.g., "Module 4.1.1, exercise2_modify.py").
- For script errors, include the full error traceback and your Python version.

### Suggesting Improvements

- Open an issue describing the improvement before submitting a pull request.
- For new exercises or modules, describe the learning objective and which module it fits into.

### Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Follow the existing code and documentation conventions (see below).
3. Test that all modified scripts run without errors.
4. Verify documentation builds cleanly: `sphinx-build -b html . build/html`
5. Submit a pull request with a clear description of the changes.

## Conventions

### Python Scripts

- Target Python 3.11 (compatible with 3.9-3.12).
- Use descriptive variable names (`pixel_color`, not `pc`).
- Include inline comments explaining the logic.
- All scripts must produce output (saved image or printed result).
- Follow the scaffolding pattern:
  - `exercise1_*.py` -- Execute (complete, run and observe)
  - `exercise2_*.py` -- Modify (CONFIG section with parameters to change)
  - `exercise3_*.py` -- Create (starter with TODO comments, 60-85% complete)

### Documentation (RST)

- Use reStructuredText for all tutorial documentation.
- Place output images in the same directory as the script that generates them.
- Use relative paths in `.. figure::` and `.. image::` directives.
- Cite sources using APA 7th edition with RST footnote syntax.
- No emojis in documentation text.

### Dependencies

- Do not add new dependencies without discussion in an issue first.
- Core modules (0-6) should only require packages listed in `pyproject.toml` base dependencies.
- ML modules (7+) may use packages from the `[ml]` optional group.

## Building the Documentation

```bash
pip install .[docs]
sphinx-build -b html . build/html
```

Open `build/html/index.html` in a browser to preview.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Open an issue or reach out via the repository's [Discussions](https://github.com/burakkagann/Pixels2GenAI/discussions) page.
