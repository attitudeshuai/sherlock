"""Offline tests for tiered timeouts and the batch time budget.

These tests never touch the network: the futures session is monkeypatched
with deterministic fake requests whose delays are fully controlled.
"""
import os
import json
import time

import pytest
from concurrent.futures import TimeoutError as FuturesTimeoutError
from jsonschema import validate

from sherlock_project.sherlock import (
    BUDGET_POLICIES,
    SherlockFuturesSession,
    resolve_budget_policy,
    resolve_site_timeouts,
    sherlock,
    timeouts_are_tiered,
)
from sherlock_project.notify import QueryNotify, QueryNotifyPrint
from sherlock_project.result import QueryStatus


class FakeResponse:
    """Minimal stand-in for a requests.Response (404 -> Available)."""

    def __init__(self, status_code=404):
        self.status_code = status_code
        self.text = ""
        self.encoding = "utf-8"
        self.elapsed = 0.0


class FakeFuture:
    """Future whose result() honours a wait timeout like concurrent.futures."""

    def __init__(self, delay, response):
        self.delay = delay
        self.response = response
        self.cancel_called = False

    def result(self, timeout=None):
        wait = self.delay if timeout is None else min(self.delay, timeout)
        if wait:
            time.sleep(wait)
        if timeout is not None and self.delay > timeout:
            raise FuturesTimeoutError()
        return self.response

    def cancel(self):
        self.cancel_called = True
        return True


def _make_site(host, **extra):
    data = {
        "url": f"https://{host}.example/{{}}",
        "urlMain": f"https://{host}.example/",
        "errorType": "status_code",
    }
    data.update(extra)
    return data


@pytest.fixture
def patched_session(monkeypatch):
    """Replace HTTP verbs with controllable fakes; expose capture + delays."""
    captured = {}
    delays = {}

    def make_fake():
        def fake_verb(self, url=None, **kwargs):
            host = url.split("//", 1)[1].split("/", 1)[0]
            captured[host] = kwargs
            return FakeFuture(delays.get(host, 0.0), FakeResponse())

        return fake_verb

    monkeypatch.setattr(SherlockFuturesSession, "get", make_fake())
    monkeypatch.setattr(SherlockFuturesSession, "head", make_fake())
    monkeypatch.setattr(SherlockFuturesSession, "post", make_fake())
    monkeypatch.setattr(SherlockFuturesSession, "put", make_fake())
    return captured, delays


class RecordingNotify(QueryNotify):
    def __init__(self):
        self.seen = []

    def update(self, result):
        self.seen.append((result.site_name, result.status))


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def test_resolve_global_default_is_passed_through_verbatim():
    # No declaration: the exact global scalar (int 60) is handed to requests.
    request_timeout, connect, read, level = resolve_site_timeouts(None, 60)
    assert request_timeout == 60
    assert type(request_timeout) is int
    assert (connect, read, level) == (60, 60, "global")


def test_resolve_scalar_site_timeout_covers_both_phases():
    request_timeout, connect, read, level = resolve_site_timeouts(15, 60)
    assert request_timeout == 15.0
    assert (connect, read, level) == (15.0, 15.0, "site")


def test_resolve_object_site_timeout_splits_phases():
    request_timeout, connect, read, level = resolve_site_timeouts(
        {"connect": 3, "read": 9}, 60
    )
    assert request_timeout == (3.0, 9.0)
    assert connect == 3.0
    assert read == 9.0
    assert level == "site"


def test_resolve_partial_object_falls_back_phase_by_phase():
    request_timeout, connect, read, level = resolve_site_timeouts(
        {"read": 7}, 60
    )
    # The undeclared connect phase falls back to the global value, verbatim.
    assert request_timeout == (60, 7.0)
    assert connect == 60
    assert read == 7.0
    assert level == "site"


@pytest.mark.parametrize("declaration", [0, -1, "10", True, [], {}, {"connect": 0},
                                         {"read": -2}, {"bogus": 1}])
def test_resolve_rejects_invalid_declarations(declaration):
    with pytest.raises(ValueError):
        resolve_site_timeouts(declaration, 60)


def test_tiered_flag():
    assert timeouts_are_tiered({"a": _make_site("a")}, None) is False
    assert timeouts_are_tiered({"a": _make_site("a")}, 30) is True
    assert timeouts_are_tiered(
        {"a": _make_site("a", timeout=5)}, None
    ) is True


def test_budget_policy_fallback_and_validation():
    assert resolve_budget_policy(None, "abandon") == "abandon"
    assert resolve_budget_policy("wait", "abandon") == "wait"
    assert BUDGET_POLICIES == ("abandon", "wait")
    with pytest.raises(ValueError):
        resolve_budget_policy("nope", "abandon")


# ---------------------------------------------------------------------------
# sherlock() integration (fully offline via the fake session)
# ---------------------------------------------------------------------------

def test_default_mode_passes_scalar_global_timeout_to_requests(patched_session):
    captured, _delays = patched_session
    results = sherlock(
        "user", {"PlainSite": _make_site("plain")}, QueryNotify()
    )

    # The exact historical scalar default reaches requests unchanged.
    assert captured["plain.example"]["timeout"] == 60
    assert type(captured["plain.example"]["timeout"]) is int

    row = results["PlainSite"]
    assert row["status"].status is QueryStatus.AVAILABLE
    assert row["timeout_level"] == "global"
    assert (row["timeout_connect"], row["timeout_read"]) == (60, 60)


def test_site_level_timeouts_reach_requests_and_results(patched_session):
    captured, _delays = patched_session
    site_data = {
        "BothPhases": _make_site("both", timeout={"connect": 3, "read": 9}),
        "ScalarSite": _make_site("scalar", timeout=15),
        "ReadOnly": _make_site("readonly", timeout={"read": 7}),
        "PlainSite": _make_site("plain"),
    }

    results = sherlock("user", site_data, QueryNotify())

    assert captured["both.example"]["timeout"] == (3.0, 9.0)
    assert captured["scalar.example"]["timeout"] == 15.0
    assert captured["readonly.example"]["timeout"] == (60, 7.0)
    assert captured["plain.example"]["timeout"] == 60

    assert results["BothPhases"]["status"].timeout_level == "site"
    assert results["BothPhases"]["timeout_connect"] == 3.0
    assert results["BothPhases"]["timeout_read"] == 9.0
    assert results["ReadOnly"]["timeout_connect"] == 60
    assert results["ReadOnly"]["timeout_read"] == 7.0
    assert results["PlainSite"]["status"].timeout_level == "global"


def test_invalid_site_timeout_raises(patched_session):
    _captured, _delays = patched_session
    with pytest.raises(ValueError):
        sherlock(
            "user",
            {"BadSite": _make_site("bad", timeout=0)},
            QueryNotify(),
        )


def test_budget_abandon_marks_pending_timeout_and_keeps_results(patched_session):
    _captured, delays = patched_session
    delays["fast.example"] = 0.05
    delays["slow.example"] = 30

    site_data = {
        "FastSite": _make_site("fast"),
        "SlowSite": _make_site("slow"),
    }
    notify = RecordingNotify()
    started = time.monotonic()
    results = sherlock(
        "user", site_data, notify, timeout=60, budget=1, budget_policy="abandon"
    )
    elapsed = time.monotonic() - started

    # The batch returns on the budget's scale, never waiting the slow tail.
    assert elapsed < 5

    # Results already obtained are preserved ...
    assert results["FastSite"]["status"].status is QueryStatus.AVAILABLE

    # ... and the pending site is explicitly Timed Out, distinct from both
    # Available (not found) and Unknown (request error).
    slow = results["SlowSite"]
    assert slow["status"].status is QueryStatus.TIMEOUT
    assert str(slow["status"].status) == "Timed Out"
    assert "budget" in slow["status"].context.lower()
    assert slow["http_status"] == ""
    assert ("SlowSite", QueryStatus.TIMEOUT) in notify.seen
    assert ("FastSite", QueryStatus.AVAILABLE) in notify.seen


def test_budget_wait_policy_exempts_site(patched_session):
    _captured, delays = patched_session
    delays["patient.example"] = 0.4

    site_data = {
        "PatientSite": _make_site("patient", budgetPolicy="wait"),
    }
    results = sherlock(
        "user", site_data, QueryNotify(), timeout=60, budget=0.1
    )

    # Budget shorter than the response, but "wait" lets the site finish.
    assert results["PatientSite"]["status"].status is QueryStatus.AVAILABLE


def test_budget_global_wait_policy(patched_session):
    _captured, delays = patched_session
    delays["patient.example"] = 0.4

    results = sherlock(
        "user",
        {"PatientSite": _make_site("patient")},
        QueryNotify(),
        budget=0.1,
        budget_policy="wait",
    )
    assert results["PatientSite"]["status"].status is QueryStatus.AVAILABLE


def test_timeout_status_is_distinct_from_available_in_notifier(capsys, patched_session):
    _captured, delays = patched_session
    delays["slow.example"] = 30

    notifier = QueryNotifyPrint(print_all=False, show_timeout_tiers=True)
    sherlock(
        "user",
        {"SlowSite": _make_site("slow")},
        notifier,
        budget=0.3,
    )
    output = capsys.readouterr().out
    # Printed unconditionally (not gated by --print-all) and clearly labelled.
    assert "SlowSite" in output
    assert "Timed Out" in output
    assert "Not Found" not in output


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

def test_schema_accepts_timeout_and_budget_policy_declarations():
    schema_path = os.path.join(
        os.path.dirname(__file__),
        "../sherlock_project/resources/data.schema.json",
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    instance = {
        "Example": {
            "url": "https://example.com/{}",
            "urlMain": "https://example.com/",
            "errorType": "status_code",
            "username_claimed": "blue",
            "timeout": {"connect": 5, "read": 30},
            "budgetPolicy": "wait",
        }
    }
    validate(instance=instance, schema=schema)


def test_schema_rejects_non_positive_timeout():
    schema_path = os.path.join(
        os.path.dirname(__file__),
        "../sherlock_project/resources/data.schema.json",
    )
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    from jsonschema import ValidationError

    instance = {
        "Example": {
            "url": "https://example.com/{}",
            "urlMain": "https://example.com/",
            "errorType": "status_code",
            "username_claimed": "blue",
            "timeout": {"connect": 0},
        }
    }
    with pytest.raises(ValidationError):
        validate(instance=instance, schema=schema)
