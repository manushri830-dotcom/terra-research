"""Mocked-HTTP tests for the Overpass client. No sockets are opened; no internet is needed."""
import _path  # noqa: F401
import random
import tempfile
import unittest
from pathlib import Path

import requests

from terra.osm_client import (DiskCache, FetchResult, OverpassClient, RateLimiter,
                              RequestBudgetExceeded, query_hash)

OK_BODY = {"version": 0.6, "elements": [{"type": "node", "id": 1, "lat": 1.0, "lon": 2.0,
                                          "tags": {"landuse": "industrial"}}]}


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, bad_json=False):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Plays back a scripted list of FakeResponse objects or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class _ZeroJitter(random.Random):
    def uniform(self, a, b):
        return 0.0


def make_client(tmp, script, **kw):
    clock = FakeClock()
    session = FakeSession(script)
    client = OverpassClient(
        DiskCache(Path(tmp) / "cache"), "TERRA-test (contact: test@example.com)",
        rate_limiter=RateLimiter(kw.pop("min_interval_s", 0.0), clock.time, clock.sleep),
        session=session, sleep=clock.sleep, rng=_ZeroJitter(), **kw)
    return client, session, clock


class RateLimiterTests(unittest.TestCase):
    def test_enforces_minimum_gap(self):
        clock = FakeClock()
        rl = RateLimiter(5.0, clock.time, clock.sleep)
        rl.wait()                      # first call: no sleep
        self.assertEqual(clock.sleeps, [])
        clock.now += 2.0               # 2 s of "work"
        rl.wait()
        self.assertEqual(clock.sleeps, [3.0])
        clock.now += 10.0              # already longer than the gap
        rl.wait()
        self.assertEqual(clock.sleeps, [3.0])


class ClientTests(unittest.TestCase):
    def test_success_is_cached_and_second_fetch_makes_no_http_call(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, _ = make_client(t, [FakeResponse(200, OK_BODY)])
            r1 = c.fetch("Q1")
            r2 = c.fetch("Q1")
            self.assertTrue(r1.ok and not r1.from_cache and r1.attempts == 1)
            self.assertTrue(r2.ok and r2.from_cache and r2.attempts == 0)
            self.assertEqual(len(s.calls), 1)
            self.assertEqual(c.requests_made, 1)
            self.assertEqual(c.cache_hits, 1)

    def test_request_headers_timeout_endpoint_and_body(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, _ = make_client(t, [FakeResponse(200, OK_BODY)])
            c.fetch("MY QUERY")
            call = s.calls[0]
            self.assertIn("test@example.com", call["headers"]["User-Agent"])
            self.assertEqual(call["timeout"], (10.0, 90.0))
            self.assertEqual(call["data"], {"data": "MY QUERY"})
            self.assertTrue(call["url"].startswith("https://overpass-api.de/"))

    def test_zero_elements_is_a_success_not_a_failure(self):
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [FakeResponse(200, {"elements": []})])
            r = c.fetch("Q")
            self.assertTrue(r.ok)
            self.assertEqual(r.response["elements"], [])

    def test_retry_on_504_then_success_with_backoff(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, clock = make_client(t, [FakeResponse(504), FakeResponse(200, OK_BODY)])
            r = c.fetch("Q")
            self.assertTrue(r.ok)
            self.assertEqual(r.attempts, 2)
            self.assertEqual(clock.sleeps, [10.0])      # base backoff, zero jitter

    def test_backoff_grows_and_gives_up_after_max_attempts(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, clock = make_client(t, [FakeResponse(504)] * 3, max_attempts=3)
            r = c.fetch("Q")
            self.assertFalse(r.ok)
            self.assertEqual(r.error_type, "http_504")
            self.assertEqual(r.attempts, 3)
            self.assertEqual(len(s.calls), 3)
            self.assertEqual(clock.sleeps, [10.0, 30.0])  # no sleep after the final attempt
            self.assertIsNone(c.cache.get(query_hash("Q")))  # failures are never cached

    def test_non_retryable_status_fails_immediately(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, clock = make_client(t, [FakeResponse(400)])
            r = c.fetch("Q")
            self.assertFalse(r.ok)
            self.assertEqual((r.error_type, r.attempts), ("http_400", 1))
            self.assertEqual(clock.sleeps, [])

    def test_timeout_and_connection_error_are_retried(self):
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [requests.exceptions.Timeout("slow"),
                                      requests.exceptions.ConnectionError("down"),
                                      FakeResponse(200, OK_BODY)])
            r = c.fetch("Q")
            self.assertTrue(r.ok)
            self.assertEqual(r.attempts, 3)
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [requests.exceptions.Timeout("slow")] * 3)
            r = c.fetch("Q")
            self.assertEqual((r.ok, r.error_type), (False, "timeout"))

    def test_http_200_with_runtime_error_remark_is_a_failure_and_not_cached(self):
        body = {"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 3"}
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [FakeResponse(200, body)] * 3)
            r = c.fetch("Q")
            self.assertFalse(r.ok)
            self.assertEqual(r.error_type, "overpass_runtime_error")
            self.assertIsNone(c.cache.get(query_hash("Q")))

    def test_invalid_json_and_unexpected_payload(self):
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [FakeResponse(200, bad_json=True)] * 3)
            self.assertEqual(c.fetch("Q").error_type, "invalid_json")
        with tempfile.TemporaryDirectory() as t:
            c, _, _ = make_client(t, [FakeResponse(200, {"hello": 1})] * 3)
            self.assertEqual(c.fetch("Q").error_type, "unexpected_payload")

    def test_retry_after_header_is_honoured_and_capped(self):
        with tempfile.TemporaryDirectory() as t:
            c, _, clock = make_client(t, [FakeResponse(429, headers={"Retry-After": "45"}),
                                          FakeResponse(200, OK_BODY)])
            c.fetch("Q")
            self.assertEqual(clock.sleeps, [45.0])
        with tempfile.TemporaryDirectory() as t:
            c, _, clock = make_client(t, [FakeResponse(429, headers={"Retry-After": "9999"}),
                                          FakeResponse(200, OK_BODY)])
            c.fetch("Q")
            self.assertEqual(clock.sleeps, [120.0])       # capped at max_backoff_s

    def test_request_budget_stops_the_run(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, _ = make_client(t, [FakeResponse(504)] * 5, max_attempts=3, max_requests=2)
            with self.assertRaises(RequestBudgetExceeded):
                c.fetch("Q")
            self.assertEqual(len(s.calls), 2)
            self.assertEqual(c.requests_made, 2)

    def test_cache_hit_does_not_consume_budget_or_rate_limit(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, clock = make_client(t, [FakeResponse(200, OK_BODY)], max_requests=1,
                                      min_interval_s=5.0)
            c.fetch("Q")
            for _ in range(3):
                self.assertTrue(c.fetch("Q").from_cache)
            self.assertEqual(c.requests_made, 1)
            self.assertEqual(clock.sleeps, [])

    def test_corrupt_cache_file_is_treated_as_miss(self):
        with tempfile.TemporaryDirectory() as t:
            c, s, _ = make_client(t, [FakeResponse(200, OK_BODY)])
            c.cache.directory.mkdir(parents=True)
            (c.cache.directory / f"{query_hash('Q')}.json").write_text("{not json", encoding="utf-8")
            self.assertTrue(c.fetch("Q").ok)
            self.assertEqual(len(s.calls), 1)

    def test_rate_limiter_applied_before_every_attempt(self):
        with tempfile.TemporaryDirectory() as t:
            c, _, clock = make_client(t, [FakeResponse(504), FakeResponse(200, OK_BODY)],
                                      min_interval_s=5.0)
            c.fetch("Q")
            # attempt 1 (no wait), backoff 10 s, attempt 2: gap already >5 s so no extra limiter sleep
            self.assertEqual(clock.sleeps, [10.0])


if __name__ == "__main__":
    unittest.main()
