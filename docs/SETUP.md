# Setup Guide

This guide walks you through setting up Lethe for development and usage.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js 18+** - [Download from nodejs.org](https://nodejs.org/)
- **npm 10+** - Usually comes with Node.js
- **Git** - For version control
- **Python 3.8+** - For research components (optional)

## 🚀 Quick Installation

### Using NPX (Recommended)

The fastest way to get started:

```bash
npx lethe init
npx lethe search "your search query"
```

### Global Installation

```bash
npm install -g lethe
lethe --help
```

### Project Installation

```bash
npm install lethe
```

## 🛠 Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sibyllinesoft/lethe.git
cd lethe
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Build the Project

```bash
npm run build
```

### 4. Run Tests

```bash
npm run test
```

### 5. Start Development Mode

```bash
npm run dev
```

## 📁 Project Structure

```
lethe/
├── packages/           # Monorepo packages
│   ├── core/          # Core context management
│   ├── search/        # Vector search implementation
│   ├── embeddings/    # Embedding models
│   └── cli/           # Command-line interface
├── ctx-run/           # Main NPX package
├── lethe-research/    # Research framework
├── test-env/          # Testing environment
├── docs/              # Documentation
└── dist/              # Built output
```

## ⚙ Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# OpenAI API (for embeddings)
OPENAI_API_KEY=your_api_key_here

# Vector Database (optional)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Search Configuration
LETHE_INDEX_NAME=lethe_conversations
LETHE_EMBEDDING_MODEL=text-embedding-ada-002
LETHE_MAX_CONTEXT_TOKENS=8000
```

### Configuration File

Create `lethe.config.json` in your project:

```json
{
  "search": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "dimensions": 1536
  },
  "context": {
    "maxTokens": 8000,
    "windowSize": 512,
    "overlap": 64
  },
  "storage": {
    "provider": "local",
    "path": "./.lethe"
  }
}
```

## 🔧 Development Scripts

| Script | Description |
|--------|-------------|
| `npm run build` | Build all packages |
| `npm run dev` | Start development mode |
| `npm run test` | Run all tests |
| `npm run test:unit` | Run unit tests only |
| `npm run test:integration` | Run integration tests |
| `npm run lint` | Lint code |
| `npm run type-check` | Run TypeScript checks |
| `npm run clean` | Clean build artifacts |

## 🐍 Research Environment Setup

If you plan to work with the research components:

### 1. Python Environment

```bash
cd lethe-research
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Research Configuration

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 3. Run Research Pipeline

```bash
python run_experiments.py
```

## 🔍 Troubleshooting

### Common Issues

#### Node.js Version Issues

```bash
# Check your Node.js version
node --version

# Use nvm to manage Node.js versions
nvm install 18
nvm use 18
```

#### Permission Errors on macOS/Linux

```bash
# Fix npm permissions
sudo chown -R $(whoami) ~/.npm
```

#### Module Resolution Issues

```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### TypeScript Compilation Errors

```bash
# Clean TypeScript cache
npx tsc --build --clean

# Rebuild project
npm run build
```

#### Python Environment Issues

```bash
# Ensure Python 3.8+
python --version

# Create fresh virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Performance Issues

#### Slow Search Performance

1. **Check embedding cache**: Ensure embeddings are cached locally
2. **Optimize index**: Use FAISS for large datasets
3. **Batch operations**: Group multiple searches together
4. **Memory usage**: Monitor Node.js heap size

#### High Memory Usage

1. **Limit context window**: Reduce `maxTokens` in config
2. **Clear caches**: Regularly clear embedding caches
3. **Streaming**: Use streaming for large conversations

## 📊 Monitoring & Debugging

### Enable Debug Logging

```bash
DEBUG=lethe:* npx lethe search "query"
```

### Performance Monitoring

```bash
# Monitor memory usage
node --inspect src/index.js

# Profile performance
node --prof src/index.js
node --prof-process isolate-*.log
```

### Health Checks

```bash
# Test search functionality
npx lethe test

# Validate configuration
npx lethe config --validate

# Check dependencies
npm doctor
```

## 🔐 Security Considerations

### API Keys

- Store API keys in environment variables, not code
- Use `.env` files for local development
- Set up secret management in production

### Data Privacy

- Conversation data is stored locally by default
- Vector embeddings may be sent to external APIs
- Review data handling policies of embedding providers

## 🚀 Production Deployment

### Environment Setup

```bash
# Production environment variables
NODE_ENV=production
LETHE_LOG_LEVEL=warn
LETHE_CACHE_SIZE=1000
```

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Performance Tuning

- Enable clustering for multi-core usage
- Set up Redis for distributed caching
- Use CDN for static assets
- Monitor with APM tools

## 📞 Getting Help

- **Documentation**: Check [docs/](../) directory
- **Issues**: [GitHub Issues](https://github.com/sibyllinesoft/lethe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sibyllinesoft/lethe/discussions)
- **Email**: For security issues only

## 📈 Next Steps

1. **Basic Usage**: Try the [API documentation](API.md)
2. **Advanced Features**: Explore [benchmarking guide](BENCHMARKING.md)
3. **Contributing**: Read the [contributing guide](../CONTRIBUTING.md)
4. **Research**: Dive into [research documentation](../lethe-research/README.md)

---

Happy coding with Lethe! 🚀