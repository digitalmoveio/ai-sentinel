# Contributing to AI Sentinel

Thank you for your interest in contributing to AI Sentinel! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-sentinel.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Development

```bash
# Run in debug mode
FLASK_DEBUG=true python -m app.main

# Run tests
python -m pytest tests/ -v
```

## How to Contribute

### Bug Reports
Open an issue with a clear description, steps to reproduce, and expected vs. actual behavior.

### Feature Requests
Open an issue describing the feature, its use case, and (if possible) a proposed approach.

### Pull Requests

1. Keep PRs focused on a single change
2. Write clear commit messages
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Update documentation if needed

## Areas Where Help Is Needed

- Improving detection accuracy with new analysis methods
- Adding support for more AI generation tools (signatures, patterns)
- Training and integrating ML-based classifiers
- Performance optimization for large files and batch processing
- Browser extension for in-page detection
- Internationalization (i18n)

## Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Write docstrings for public methods
- Keep functions focused and under 50 lines where possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
