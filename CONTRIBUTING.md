# Contributing to FASTLight

We welcome contributions to FASTLight! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/PushpakAg/fastlight.git
   cd fastlight
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev,examples]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastlight --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Writing Tests

- **Test files**: `tests/test_*.py`
- **Test functions**: `test_*`
- **Test classes**: `Test*`
- **Fixtures**: Use pytest fixtures for setup
- **Markers**: Use `@pytest.mark.slow` for slow tests

## Bug Reports

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version** from main branch
3. **Minimize reproduction case** to essential code

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Code example
2. Expected vs actual behavior

**Environment**
- Python version: 3.8+
- FASTLight version: 1.0.0
- OS: Linux/macOS/Windows

**Additional Context**
Any other relevant information.
```

## Feature Requests

### Before Submitting

1. **Check existing issues** for similar requests
2. **Consider scope** - is it within project goals?
3. **Provide use case** - why is this feature needed?

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other relevant information.
```

## Pull Request Process

### Before Submitting

1. **Create feature branch** from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following code style guidelines

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Run tests** to ensure everything passes:
   ```bash
   pytest
   ```

### Pull Request Template

```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests pass

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation** review
5. **Merge** to main branch

## Adding New Features

1. **Core functionality**: Add to appropriate module
2. **Utilities**: Add to utils.py or new module
3. **Examples**: Add to examples/ directory
4. **Tests**: Add comprehensive tests
5. **Documentation**: Update README and docstrings

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in setup.py and __init__.py
2. **Update CHANGELOG.md** with new features/fixes
3. **Run full test suite** on multiple Python versions
4. **Update documentation** if needed
5. **Create release** on GitHub
6. **Publish to PyPI** (maintainers only)

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Check README and docstrings first

## Contact

- **Maintainers**: FASTLight Team
- **GitHub**: [@PushpakAg/fastlight](https://github.com/PushpakAg/fastlight)

Thank you for contributing to FASTLight!