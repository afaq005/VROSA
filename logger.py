"""
logger.py — Lightweight timestamp logger shared across all modules.
"""

from datetime import datetime

log_entries: list[str] = []


def log(text: str, tag: str = "INFO") -> None:
    ts    = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] [{tag}] {text}"
    log_entries.append(entry[-200:])
    print(entry)