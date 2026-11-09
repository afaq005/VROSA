"""logger.py — Console, dashboard and persistent runtime text log."""

from datetime import datetime

log_entries: list[str] = []


def log(text: str, tag: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    entry = f"[{ts}] [{tag}] {text}"
    log_entries.append(entry[-500:])
    if len(log_entries) > 2000:
        del log_entries[:-1000]
    print(entry)
    try:
        from experiment_logger import RUN_DIR
        with (RUN_DIR / "runtime.log").open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass
