#!/bin/bash

set -e

BUILD_DIR="build"

echo "Cleaning project..."
rm -rf "$BUILD_DIR"

echo "Configuring CMake..."
cmake -S . -B "$BUILD_DIR"

echo "Building project..."
cmake --build "$BUILD_DIR" --parallel 8 --verbose