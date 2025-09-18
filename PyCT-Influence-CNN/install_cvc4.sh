#!/bin/bash
set -e
# Clone and checkout the required commit of CVC4
git clone https://github.com/CVC4/CVC4.git
cd CVC4
git checkout d1f3225e26b9d64f065048885053392b10994e715
# Build dependencies (ANTLR)
./contrib/get-antlr-3.4
# Configure with Python bindings enabled
./configure.sh --language-bindings=python --python3
# Build and install
cd build
make -j$(nproc)
make install

echo "export PYTHONPATH=$(pwd)/src/bindings/python" >> ~/.bashrc
echo "CVC4 installed successfully. Please run: source ~/.bashrc"
