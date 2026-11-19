"""Where the bot keeps per-customer memory (recent messages, "talk to admin" pauses).

On Vercel every request can land on a fresh copy of the bot, so memory has to live
outside it. If an Upstash Redis database is connected (Vercel → Storage → Upstash),
we use it through its REST API. Otherwise we fall back to in-process memory, which
works locally and on always-on servers.
"""

import json
import logging
import os
import threading
import time
import urllib.request

log = logging.getLogger("jiajia.store")


class MemoryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}  # key -> (expires_at, value)

    def get(self, key: str):
        with self._lock:
            expires, value = self._data.get(key, (0, None))
            return value if expires > time.time() else None

    def set(self, key: str, value, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl_seconds, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class UpstashStore:
    """Tiny Upstash Redis REST client (no extra packages). Values are stored as JSON."""

    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def _cmd(self, *args):
        req = urllib.request.Request(
            self.url,
            data=json.dumps([str(a) for a in args]).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp).get("result")

    def get(self, key: str):
        try:
            raw = self._cmd("GET", key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.warning("store get failed: %s", exc)
            return None

    def set(self, key: str, value, ttl_seconds: int) -> None:
        try:
            self._cmd("SET", key, json.dumps(value, ensure_ascii=False), "EX", int(ttl_seconds))
        except Exception as exc:
            log.warning("store set failed: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self._cmd("DEL", key)
        except Exception as exc:
            log.warning("store delete failed: %s", exc)


def _env_ending(*suffixes):
    """First env var named exactly like, or ending in, one of the suffixes (Vercel may add a prefix)."""
    for suffix in suffixes:
        if os.environ.get(suffix):
            return os.environ[suffix]
    for suffix in suffixes:
        for name, value in sorted(os.environ.items()):
            if name.endswith("_" + suffix) and value:
                return value
    return None


def make_store():
    url = _env_ending("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL")
    token = _env_ending("UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN")
    if url and token:
        log.info("using Upstash Redis for chat memory")
        return UpstashStore(url, token)
    return MemoryStore()
