#!/usr/bin/env bash
# format_all.sh
# Usage: ./format_all.sh 4  -> runs 4 jobs in parallel

# default to 4 parallel jobs if not provided
JOBS=${1:-4}

# Make sure GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel not found. Install it with: sudo apt install parallel"
    exit 1
fi

# Find all relevant files and format them in parallel
find . \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cuh' \) -print0 |
    parallel -0 -j "$JOBS" clang-format -i {}