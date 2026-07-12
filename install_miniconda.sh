#!/bin/bash
set -e

INSTALL_DIR="$HOME/miniconda3"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "Miniconda not found. Installing..."

    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"

    rm -f /tmp/miniconda.sh

    "$INSTALL_DIR/bin/conda" init bash >/dev/null 2>&1
else
    echo "Miniconda already installed. Skipping installation."
fi

# Load conda into current shell
eval "$("$INSTALL_DIR/bin/conda" shell.bash hook)"

echo "Conda version:"
conda --version

# Install uv only if it is not already installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    pip install -q uv
else
    echo "uv already installed."
fi

echo "========================================"
echo "Setup completed successfully!"
echo "If this is the first installation, run:"
echo "    source ~/.bashrc"
echo "or open a new terminal."
echo "========================================"