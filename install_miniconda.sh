#!/bin/bash
set -e

INSTALL_DIR="$HOME/miniconda3"

echo "Downloading Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

echo "Installing Miniconda..."
bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"

rm -f /tmp/miniconda.sh

echo "Initializing conda..."
"$INSTALL_DIR/bin/conda" init bash >/dev/null 2>&1

# Make conda available for current shell
eval "$("$INSTALL_DIR/bin/conda" shell.bash hook)"

echo "Updating conda..."
conda update -n base -c defaults conda -y

echo ""
echo "========================================"
echo "Miniconda installed successfully!"
echo "Restart your shell or run:"
echo "source ~/.bashrc"
echo ""
echo "Conda version:"
conda --version
echo "========================================"
pip install -q uv
echo "========================================"