#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import ast
from pathlib import Path
sources=sorted(Path('.').rglob('*.py'))
for path in sources:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'AST syntax: PASS ({len(sources)} Python files)')
PY
python3 -m pytest -q tests/test_offline_regression.py
