"""Validate public-repository hygiene without installing project dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".db", ".onnx", ".pt", ".pth", ".sqlite", ".sqlite3", ".tif", ".tiff", ".whl", ".zip"}
MAX_FILE_SIZE = 25 * 1024 * 1024
TEXT_SUFFIXES = {".html", ".ipynb", ".md", ".py", ".txt", ".yml", ".yaml"}
PATTERNS = {
    "hard-coded Sentinel Hub client ID": re.compile(r"config\.sh_client_id\s*=\s*['\"][^'\"]+['\"]"),
    "hard-coded Sentinel Hub client secret": re.compile(r"config\.sh_client_secret\s*=\s*['\"][^'\"]+['\"]"),
    "hard-coded Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\"),
    "hard-coded Kaggle path": re.compile(r"/kaggle/(?:input|working)/"),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    relative = path.relative_to(ROOT)
    if ".ipynb_checkpoints" in path.parts:
        fail(f"notebook checkpoint is tracked: {relative}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        fail(f"private or large artifact is tracked: {relative}")
    if path.stat().st_size > MAX_FILE_SIZE:
        fail(f"file exceeds 25 MB: {relative}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for description, pattern in PATTERNS.items():
        if pattern.search(text):
            fail(f"{description} in {relative}")

for path in ROOT.rglob("*.ipynb"):
    with path.open(encoding="utf-8") as handle:
        json.load(handle)

for path in (ROOT / "app").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

print("Repository validation passed.")
