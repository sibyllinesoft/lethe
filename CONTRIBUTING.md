# Contributing to Lethe

Thank you for your interest in contributing to Lethe! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm 10+
- Python 3.8+ (for research components)
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/sibyllinesoft/lethe.git
   cd lethe
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Build the project**
   ```bash
   npm run build
   ```

4. **Run tests**
   ```bash
   npm run test
   ```

5. **Start development mode**
   ```bash
   npm run dev
   ```

## 📋 Project Structure

```
lethe/
├── packages/           # Monorepo packages
│   ├── core/          # Core context management logic
│   ├── search/        # Vector search implementation
│   ├── embeddings/    # Embedding models and utilities
│   └── cli/           # Command-line interface
├── ctx-run/           # Main NPX package
├── lethe-research/    # Research framework (optional)
├── test-env/          # Testing environment
└── docs/              # Documentation
```

## 🛠 Development Workflow

### Code Style

We use TypeScript with strict mode and follow these conventions:

- **ESLint**: Automatic linting with `npm run lint`
- **Prettier**: Code formatting (integrated with ESLint)
- **Naming**: Use camelCase for variables/functions, PascalCase for classes/types
- **Comments**: JSDoc for public APIs, inline comments for complex logic

### Testing

- **Unit Tests**: Vitest for component testing
- **Integration Tests**: End-to-end testing of key workflows
- **Research Tests**: Benchmark validation in `lethe-research/`

Run tests with:
```bash
npm run test              # All tests
npm run test:unit         # Unit tests only
npm run test:integration  # Integration tests
```

### Git Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes with clear commits**
   ```bash
   git add .
   git commit -m "feat: add semantic search optimization"
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**

## 📝 Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: implement hybrid BM25+vector search
fix: resolve memory leak in embeddings cache
docs: update API documentation for search methods
test: add comprehensive benchmarking suite
```

## 🎯 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Node.js version, OS, etc)
- **Code samples** or error logs if applicable

### 💡 Feature Requests

For new features:

- **Describe the use case** and why it's needed
- **Provide examples** of how it would be used
- **Consider implementation complexity** and breaking changes
- **Check existing issues** to avoid duplicates

### 🔬 Research Contributions

We welcome research contributions! Areas of interest:

- **Search algorithms**: New retrieval methods or optimizations
- **Embedding models**: Integration with new embedding approaches  
- **Benchmarking**: New datasets or evaluation metrics
- **Performance**: Memory and speed optimizations

### 📚 Documentation

Documentation improvements are always appreciated:

- **API documentation**: JSDoc improvements
- **User guides**: Setup, usage, and troubleshooting  
- **Code examples**: Practical usage demonstrations
- **Research documentation**: Academic and technical papers

## 🔍 Code Review Process

### Pull Request Guidelines

- **Clear title and description** explaining the changes
- **Link to related issues** using `Closes #123` or `Fixes #456`
- **Include tests** for new functionality
- **Update documentation** as needed
- **Keep changes focused** - one feature/fix per PR

### Review Criteria

We review PRs based on:

- **Code quality**: Readability, maintainability, performance
- **Testing**: Adequate coverage and test quality
- **Documentation**: Clear comments and updated docs
- **Compatibility**: No breaking changes without good reason
- **Research rigor**: For research contributions, proper methodology

## 📊 Performance Considerations

### Benchmarking

For performance-related changes:

1. **Establish baselines** before making changes
2. **Use the research framework** for consistent measurement
3. **Test with realistic datasets** from LetheBench
4. **Document performance impact** in PR description

### Memory and CPU

- **Profile memory usage** for embedding operations
- **Optimize hot paths** in search algorithms  
- **Consider async/await** for I/O operations
- **Monitor bundle size** for client-side components

## 🤝 Community Guidelines

### Be Respectful

- **Inclusive language** in all communications
- **Constructive feedback** in code reviews
- **Patience with newcomers** learning the codebase
- **Professional tone** in all interactions

### Be Collaborative

- **Ask questions** when something is unclear
- **Offer help** to other contributors
- **Share knowledge** through documentation and examples
- **Coordinate** on larger changes through issues

## 🏆 Recognition

Contributors will be:

- **Listed in CONTRIBUTORS.md** for significant contributions
- **Mentioned in release notes** for notable features/fixes
- **Credited in research publications** for academic contributions
- **Given maintainer access** for consistent, high-quality contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs, features, and questions
- **Discussions**: For open-ended conversations
- **Email**: For sensitive security issues
- **Documentation**: Check existing docs first

## 📜 License

By contributing to Lethe, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Lethe! Your efforts help make AI-powered development tools better for everyone. 🚀