#!/usr/bin/env bash
# Reproduce everything from data/raw. Equivalent to: python run_all.py
set -euo pipefail
cd "$(dirname "$0")"
python run_all.py "$@"
