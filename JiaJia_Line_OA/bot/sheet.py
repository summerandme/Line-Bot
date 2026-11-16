"""Live FAQ from a published Google Sheet (File → Share → Publish to web → CSV).

Set SHEET_CSV_URL to one or more CSV links (comma-separated). The text is cached
for 60 seconds, so edits in the sheet reach the bot within a minute without a
redeploy. If the sheet can't be fetched, the last good copy is used.
"""

import csv
import io
import logging
import os
import threading
import time
import urllib.request

log = logging.getLogger("jiajia.sheet")

CACHE_SECONDS = 60
_lock = threading.Lock()
_cache = {"at": 0.0, "text": ""}


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=6) as resp:
        raw = resp.read().decode("utf-8-sig")
    rows = csv.reader(io.StringIO(raw))
    lines = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in rows]
    return "\n".join(line for line in lines if line)


def sheet_text() -> str:
    urls = [u.strip() for u in os.environ.get("SHEET_CSV_URL", "").split(",") if u.strip()]
    if not urls:
        return ""
    with _lock:
        if time.time() - _cache["at"] < CACHE_SECONDS:
            return _cache["text"]
    try:
        text = "\n\n".join(_fetch(u) for u in urls)
        with _lock:
            _cache.update(at=time.time(), text=text)
        return text
    except Exception as exc:
        log.warning("sheet fetch failed, using last good copy: %s", exc)
        with _lock:
            _cache["at"] = time.time()  # don't hammer a failing sheet
            return _cache["text"]
