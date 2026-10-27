"""Stage 4 (part 1): a polite Overpass API client.

Documented public method: Overpass API (https://overpass-api.de/api/interpreter).
Public-instance etiquette implemented here:
  * one request at a time with a minimum gap between requests (RateLimiter)
  * identifying User-Agent on every request
  * connect/read timeouts on every request
  * a small, bounded number of retries with exponential backoff + jitter
  * HTTP 429 / 502 / 503 / 504, timeouts, connection errors and Overpass "runtime error"
    remarks are retried; other HTTP errors (e.g. 400 = bad query) are NOT retried
  * a hard cap on total HTTP requests per run (max_requests)
  * successful responses are cached on disk; failures are never cached
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
RETRYABLE_STATUS = {429, 502, 503, 504}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class RequestBudgetExceeded(RuntimeError):
    """Raised when the per-run HTTP request cap is reached (not a per-cluster failure)."""


class RateLimiter:
    """Guarantees at least `min_interval_s` between consecutive wait() returns."""

    def __init__(self, min_interval_s: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval_s = float(min_interval_s)
        self._clock = clock
        self._sleep = sleep
        self._last: Optional[float] = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


class DiskCache:
    """One JSON file per query hash. Corrupt or unreadable files are treated as a miss."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        try:
            with open(self._path(key), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and "response" in data else None

    def put(self, key: str, payload: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self._path(key)
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f)
        os.replace(tmp, final)


@dataclass
class FetchResult:
    ok: bool
    response: Optional[dict] = None
    from_cache: bool = False
    attempts: int = 0                 # HTTP attempts made (0 for a cache hit)
    fetched_utc: Optional[str] = None
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    http_status: Optional[int] = None


class OverpassClient:
    def __init__(self, cache: DiskCache, user_agent: str,
                 rate_limiter: Optional[RateLimiter] = None,
                 endpoint: str = DEFAULT_ENDPOINT,
                 connect_timeout_s: float = 10.0, read_timeout_s: float = 90.0,
                 max_attempts: int = 3, backoff_base_s: float = 10.0,
                 backoff_factor: float = 3.0, max_backoff_s: float = 120.0,
                 max_requests: Optional[int] = None,
                 session=None, sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.cache = cache
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter or RateLimiter(5.0)
        self.endpoint = endpoint
        self.timeout = (connect_timeout_s, read_timeout_s)
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_factor = backoff_factor
        self.max_backoff_s = max_backoff_s
        self.max_requests = max_requests
        self.session = session or requests.Session()
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.requests_made = 0
        self.cache_hits = 0

    def _backoff_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        delay = min(self.max_backoff_s, self.backoff_base_s * self.backoff_factor ** (attempt - 1))
        delay *= 1.0 + self._rng.uniform(0.0, 0.25)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.max_backoff_s))
        return delay

    def fetch(self, query: str) -> FetchResult:
        key = query_hash(query)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return FetchResult(True, cached["response"], from_cache=True, attempts=0,
                               fetched_utc=cached.get("fetched_utc"))

        error_type = error_detail = None
        http_status: Optional[int] = None
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            if self.max_requests is not None and self.requests_made >= self.max_requests:
                raise RequestBudgetExceeded(f"request cap of {self.max_requests} reached")
            self.rate_limiter.wait()
            self.requests_made += 1
            attempts = attempt
            retryable = True
            retry_after: Optional[float] = None
            http_status = None
            try:
                resp = self.session.post(self.endpoint, data={"data": query},
                                         headers={"User-Agent": self.user_agent},
                                         timeout=self.timeout)
            except requests.exceptions.Timeout as e:
                error_type, error_detail = "timeout", str(e)[:200]
            except requests.exceptions.ConnectionError as e:
                error_type, error_detail = "connection_error", str(e)[:200]
            except requests.exceptions.RequestException as e:
                error_type, error_detail, retryable = "request_error", str(e)[:200], False
            else:
                http_status = resp.status_code
                if http_status == 200:
                    try:
                        payload = resp.json()
                    except ValueError:
                        error_type, error_detail = "invalid_json", "HTTP 200 body was not JSON"
                    else:
                        remark = payload.get("remark") if isinstance(payload, dict) else None
                        if not isinstance(payload, dict) or "elements" not in payload:
                            error_type, error_detail = "unexpected_payload", "no 'elements' in response"
                        elif remark and "runtime error" in str(remark).lower():
                            error_type, error_detail = "overpass_runtime_error", str(remark)[:200]
                        else:
                            fetched = utc_now_iso()
                            self.cache.put(key, {"query_hash": key, "endpoint": self.endpoint,
                                                 "fetched_utc": fetched, "response": payload})
                            return FetchResult(True, payload, attempts=attempt, fetched_utc=fetched,
                                               http_status=200)
                elif http_status in RETRYABLE_STATUS:
                    error_type, error_detail = f"http_{http_status}", f"HTTP {http_status}"
                    try:
                        retry_after = float(resp.headers.get("Retry-After"))
                    except (TypeError, ValueError):
                        retry_after = None
                else:
                    error_type, error_detail, retryable = f"http_{http_status}", f"HTTP {http_status}", False

            if not retryable or attempt == self.max_attempts:
                break
            self._sleep(self._backoff_delay(attempt, retry_after))

        return FetchResult(False, attempts=attempts, error_type=error_type,
                           error_detail=error_detail, http_status=http_status)
