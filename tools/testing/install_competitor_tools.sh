#!/bin/bash
# Competitor Tool Installation Script for Lethe Benchmarking
# Generated on: 2025-01-06
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Checking system requirements..."

# Check if we're on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    echo "❌ This script requires apt-get (Ubuntu/Debian)"
    exit 1
fi

# Check available space
available_space=$(df / | awk 'NR==2{print $4}')
if [ "$available_space" -lt 2000000 ]; then
    echo "⚠️  Warning: Less than 2GB free space available"
fi

echo "📦 Installing competitor search tools..."

# Update package list
sudo apt-get update -q

# Install fzf (fuzzy finder)
echo "Installing fzf..."
sudo apt-get install -y fzf

# Install ctags for code indexing
echo "Installing universal-ctags..."
sudo apt-get install -y universal-ctags

# Install additional search tools
echo "Installing additional search tools..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3-pip

# Install comby (structural search tool)
echo "Installing comby..."
if ! command -v comby &> /dev/null; then
    # Download and install comby binary
    curl -L https://github.com/comby-tools/comby/releases/download/1.8.1/comby-1.8.1-x86_64-linux.tar.gz \
        -o /tmp/comby.tar.gz
    tar -xzf /tmp/comby.tar.gz -C /tmp
    sudo mv /tmp/comby /usr/local/bin/
    sudo chmod +x /usr/local/bin/comby
    rm -f /tmp/comby.tar.gz
fi

# Install OpenGrok (requires Java)
echo "Installing OpenGrok dependencies..."
sudo apt-get install -y openjdk-11-jdk tomcat9

# Download OpenGrok
if [ ! -f /opt/opengrok/opengrok-1.13.7.tar.gz ]; then
    sudo mkdir -p /opt/opengrok
    sudo wget -O /opt/opengrok/opengrok-1.13.7.tar.gz \
        https://github.com/oracle/opengrok/releases/download/1.13.7/opengrok-1.13.7.tar.gz
    
    cd /opt/opengrok
    sudo tar -xzf opengrok-1.13.7.tar.gz
    sudo chown -R $USER:$USER /opt/opengrok
fi

# Try to install serena if available
echo "Attempting to install serena..."
pip3 install --user serena-lsp 2>/dev/null || echo "serena-lsp not available via pip"

# Install additional Python dependencies for benchmarking
echo "Installing Python benchmarking dependencies..."
pip3 install --user \
    matplotlib \
    seaborn \
    scipy \
    numpy \
    pandas \
    scikit-learn \
    plotly \
    tqdm

echo "✅ Verifying installations..."

# Verify installations
echo "Installed tools:"
for tool in rg ag fzf comby git ctags; do
    if command -v "$tool" &> /dev/null; then
        version=$("$tool" --version 2>/dev/null | head -1 || echo "version unknown")
        echo "  ✓ $tool: $version"
    else
        echo "  ✗ $tool: not found"
    fi
done

# Check Java for OpenGrok
if command -v java &> /dev/null; then
    java_version=$(java -version 2>&1 | head -1)
    echo "  ✓ Java: $java_version"
else
    echo "  ✗ Java: not found"
fi

# Check Python tools
if python3 -c "import matplotlib, scipy, numpy" 2>/dev/null; then
    echo "  ✓ Python scientific stack: installed"
else
    echo "  ✗ Python scientific stack: missing dependencies"
fi

echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure OpenGrok indexing"
echo "2. Set up benchmark corpus"
echo "3. Run competitive benchmark suite"