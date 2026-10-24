"""Tests for resumable batch scans (--resume / Checkpoint).

These tests are fully offline: state-file handling is exercised directly,
and the resume behaviour inside sherlock() is tested against a local HTTP
server on the loopback interface.
"""

import http.server
import json
import os
import threading

import pytest

from sherlock_project.checkpoint import (
    Checkpoint,
    CheckpointError,
    compute_site_fingerprint,
)
from sherlock_project.notify import QueryNotify
from sherlock_project.result import QueryStatus
from sherlock_project.sherlock import sherlock, SherlockFuturesSession
from sherlock_interactives import Interactives
from sherlock_interactives import InteractivesSubprocessError


SITE_DATA = {
    "GitHub": {
        "url": "https://github.com/{}",
        "urlMain": "https://github.com",
        "errorType": "status_code",
    }
}


def _load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# State file lifecycle
# ---------------------------------------------------------------------------

def test_checkpoint_creates_file_and_roundtrips(tmp_path):
    path = str(tmp_path / "state.json")
    checkpoint, resumed = Checkpoint.load_or_create(
        path,
        usernames=["alice"],
        site_data=SITE_DATA,
        sherlock_version="0.16.2",
    )
    assert resumed is False
    # The file exists from the very start and is human-readable JSON.
    assert os.path.exists(path)
    raw = _load_raw(path)
    assert raw["tool"] == "sherlock-project"
    assert raw["batch"]["usernames"] == ["alice"]
    assert raw["batch"]["site_count"] == 1

    checkpoint.record(
        "alice",
        "GitHub",
        status=QueryStatus.CLAIMED.value,
        site_url_user="https://github.com/alice",
        url_user="https://github.com/alice",
        http_status=200,
        query_time=0.25,
        context=None,
    )

    # A second process loads the exact same file and sees the conclusion.
    checkpoint2, resumed2 = Checkpoint.load_or_create(
        path,
        usernames=["alice"],
        site_data=SITE_DATA,
        sherlock_version="0.16.2",
    )
    assert resumed2 is True
    assert checkpoint2.completed_unit_count() == 1
    record = checkpoint2.get_record("alice", "GitHub")
    assert record["status"] == "Claimed"
    assert record["http_status"] == 200
    assert record["query_time"] == 0.25
    assert "completed_at" in record

    checkpoint2.mark_batch_complete()
    assert _load_raw(path)["batch"]["completed"] is True


def test_checkpoint_file_with_zero_units_is_not_considered_resumed(tmp_path):
    path = str(tmp_path / "state.json")
    checkpoint, resumed = Checkpoint.load_or_create(
        path, usernames=["alice"], site_data=SITE_DATA, sherlock_version="x"
    )
    assert resumed is False
    # Re-opening an empty (brand new) file also reports a fresh start.
    _, resumed_again = Checkpoint.load_or_create(
        path, usernames=["alice"], site_data=SITE_DATA, sherlock_version="x"
    )
    assert resumed_again is False


def test_checkpoint_corrupt_json_is_rejected_with_reset_hint(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        Checkpoint.load_or_create(
            str(path), usernames=["alice"], site_data=SITE_DATA,
            sherlock_version="x",
        )
    with pytest.raises(CheckpointError, match="delete this file"):
        Checkpoint.load_or_create(
            str(path), usernames=["alice"], site_data=SITE_DATA,
            sherlock_version="x",
        )


@pytest.mark.parametrize("payload", [
    [],
    "not-an-object",
    {},
    {"tool": "something-else"},
    {"tool": "sherlock-project"},  # missing format version
    {"tool": "sherlock-project", "format_version": 999},
    {"tool": "sherlock-project", "format_version": 1},  # missing batch/results
])
def test_checkpoint_malformed_structure_is_rejected(tmp_path, payload):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointError):
        Checkpoint.load_or_create(
            str(path), usernames=["alice"], site_data=SITE_DATA,
            sherlock_version="x",
        )


def test_checkpoint_record_with_unknown_status_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    checkpoint, _ = Checkpoint.load_or_create(
        str(path), usernames=["alice"], site_data=SITE_DATA,
        sherlock_version="x",
    )
    checkpoint.record(
        "alice", "GitHub",
        status=QueryStatus.CLAIMED.value,
        site_url_user="u", url_user="u", http_status=200,
        query_time=None, context=None,
    )
    raw = _load_raw(str(path))
    raw["results"]["alice"]["GitHub"]["status"] = "Bogus"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CheckpointError, match="unknown status"):
        Checkpoint.load_or_create(
            str(path), usernames=["alice"], site_data=SITE_DATA,
            sherlock_version="x",
        )


def test_checkpoint_usernames_mismatch_is_rejected(tmp_path):
    path = str(tmp_path / "state.json")
    checkpoint, _ = Checkpoint.load_or_create(
        path, usernames=["alice"], site_data=SITE_DATA, sherlock_version="x"
    )
    checkpoint.record(
        "alice", "GitHub",
        status=QueryStatus.CLAIMED.value,
        site_url_user="u", url_user="u", http_status=200,
        query_time=None, context=None,
    )
    # Same site set, different batch of usernames.
    with pytest.raises(CheckpointError, match="batch parameters do not match"):
        Checkpoint.load_or_create(
            path, usernames=["bob"], site_data=SITE_DATA, sherlock_version="x"
        )


def test_checkpoint_site_set_mismatch_is_rejected(tmp_path):
    path = str(tmp_path / "state.json")
    Checkpoint.load_or_create(
        path, usernames=["alice"], site_data=SITE_DATA, sherlock_version="x"
    )
    changed_sites = {
        "GitHub": dict(SITE_DATA["GitHub"]),
        "GitLab": {
            "url": "https://gitlab.com/{}",
            "urlMain": "https://gitlab.com",
            "errorType": "status_code",
        },
    }
    with pytest.raises(CheckpointError, match="target site set does not match"):
        Checkpoint.load_or_create(
            path, usernames=["alice"], site_data=changed_sites,
            sherlock_version="x",
        )

    # A change in an existing site's detection rules must mismatch too.
    changed_rules = {"GitHub": dict(SITE_DATA["GitHub"], errorCode=[405])}
    with pytest.raises(CheckpointError, match="target site set does not match"):
        Checkpoint.load_or_create(
            path, usernames=["alice"], site_data=changed_rules,
            sherlock_version="x",
        )


def test_site_fingerprint_is_stable_and_sensitive():
    other_order = dict(reversed(list(SITE_DATA.items())))
    assert compute_site_fingerprint(SITE_DATA) == compute_site_fingerprint(other_order)

    changed = {"GitHub": dict(SITE_DATA["GitHub"], errorCode=[404])}
    assert compute_site_fingerprint(SITE_DATA) != compute_site_fingerprint(changed)


# ---------------------------------------------------------------------------
# sherlock() resume behaviour
# ---------------------------------------------------------------------------

def test_sherlock_restores_units_without_any_request(tmp_path, monkeypatch):
    site_data = {
        "RestoredSite": {
            "urlMain": "https://restored.example",
            "url": "http://127.0.0.1:9/restored/{}",
            "errorType": "status_code",
        },
        "IllegalSite": {
            "urlMain": "https://illegal.example",
            "url": "https://illegal.example/{}",
            "errorType": "message",
            "errorMsg": "no such user",
            "regexCheck": r"^[a-z]+$",
        },
    }
    path = str(tmp_path / "state.json")
    checkpoint, _ = Checkpoint.load_or_create(
        path, usernames=["Bad-User_1"], site_data=site_data,
        sherlock_version="x",
    )
    checkpoint.record(
        "Bad-User_1",
        "RestoredSite",
        status=QueryStatus.CLAIMED.value,
        site_url_user="http://127.0.0.1:9/restored/Bad-User_1",
        url_user="http://127.0.0.1:9/restored/Bad-User_1",
        http_status=200,
        query_time=0.42,
        context=None,
    )

    def forbid_request(*args, **kwargs):
        raise AssertionError("resume must not issue a request for a done unit")

    monkeypatch.setattr(SherlockFuturesSession, "request", forbid_request)

    results = sherlock(
        "Bad-User_1",
        site_data,
        QueryNotify(),
        checkpoint=checkpoint,
    )

    # Dict order must be identical to a full run.
    assert list(results.keys()) == ["RestoredSite", "IllegalSite"]

    restored = results["RestoredSite"]
    assert restored["status"].status is QueryStatus.CLAIMED
    assert restored["http_status"] == 200
    assert restored["status"].query_time == 0.42
    assert restored["url_user"] == "http://127.0.0.1:9/restored/Bad-User_1"

    # The deterministic ILLEGAL unit is evaluated once and persisted.
    assert results["IllegalSite"]["status"].status is QueryStatus.ILLEGAL
    assert results["IllegalSite"]["url_user"] == ""
    assert (
        checkpoint.get_record("Bad-User_1", "IllegalSite")["status"] == "Illegal"
    )
    # Restored conclusion must not have been overwritten.
    assert (
        checkpoint.get_record("Bad-User_1", "RestoredSite")["query_time"] == 0.42
    )


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    """Local test server: /found/<user> claims, /missing/<user> 404s."""

    counts = {"found": 0, "missing": 0}

    def _handle(self):
        first = self.path.lstrip("/").split("/", 1)[0]
        if first in self.counts:
            type(self).counts[first] += 1
        if first == "found":
            body = b"<html>welcome to the profile</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
        else:
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)

    do_GET = _handle
    do_HEAD = _handle

    def log_message(self, *args):
        pass


@pytest.fixture()
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _CountingHandler.counts.update({"found": 0, "missing": 0})
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_sherlock_resume_only_queries_outstanding_units(tmp_path, local_server):
    port = local_server
    site_data = {
        "FoundSite": {
            "urlMain": f"http://127.0.0.1:{port}",
            "url": f"http://127.0.0.1:{port}/found/{{}}",
            "errorType": "message",
            "errorMsg": "no such user",
        },
        "MissingSite": {
            "urlMain": f"http://127.0.0.1:{port}",
            "url": f"http://127.0.0.1:{port}/missing/{{}}",
            "errorType": "status_code",
            "errorCode": 404,
        },
    }
    username = "alice"
    path = str(tmp_path / "state.json")

    # First, uninterrupted run against a fresh checkpoint.
    checkpoint, resumed = Checkpoint.load_or_create(
        path, usernames=[username], site_data=site_data, sherlock_version="x"
    )
    assert resumed is False
    full_results = sherlock(username, site_data, QueryNotify(), checkpoint=checkpoint)

    assert full_results["FoundSite"]["status"].status is QueryStatus.CLAIMED
    assert full_results["MissingSite"]["status"].status is QueryStatus.AVAILABLE
    assert _CountingHandler.counts == {"found": 1, "missing": 1}
    first_query_time = full_results["FoundSite"]["status"].query_time
    assert first_query_time is not None

    # Simulate the process dying before MissingSite's conclusion was saved.
    raw = _load_raw(path)
    del raw["results"][username]["MissingSite"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(raw, handle)

    # A new process resumes with the same batch parameters.
    checkpoint2, resumed2 = Checkpoint.load_or_create(
        path, usernames=[username], site_data=site_data, sherlock_version="x"
    )
    assert resumed2 is True
    merged_results = sherlock(
        username, site_data, QueryNotify(), checkpoint=checkpoint2
    )

    # The already-concluded unit was never queried again; only the
    # outstanding unit was.
    assert _CountingHandler.counts == {"found": 1, "missing": 2}

    # The merged result set is shaped and ordered like a full run.
    assert list(merged_results.keys()) == list(full_results.keys())
    for site_name, merged in merged_results.items():
        assert set(merged.keys()) == set(full_results[site_name].keys())

    assert merged_results["FoundSite"]["status"].status is QueryStatus.CLAIMED
    assert merged_results["FoundSite"]["status"].query_time == first_query_time
    assert merged_results["MissingSite"]["status"].status is QueryStatus.AVAILABLE
    assert merged_results["MissingSite"]["http_status"] == 404

    # Resuming once more when everything is done issues zero requests.
    checkpoint3, resumed3 = Checkpoint.load_or_create(
        path, usernames=[username], site_data=site_data, sherlock_version="x"
    )
    assert resumed3 is True
    final_results = sherlock(username, site_data, QueryNotify(), checkpoint=checkpoint3)
    assert _CountingHandler.counts == {"found": 1, "missing": 2}
    assert final_results["FoundSite"]["status"].status is QueryStatus.CLAIMED
    assert final_results["MissingSite"]["status"].status is QueryStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_corrupt_checkpoint_aborts_before_scan(tmp_path):
    state_path = tmp_path / "broken.json"
    state_path.write_text("{corrupt", encoding="utf-8")
    # --local keeps manifest loading offline; the checkpoint error must
    # terminate the run before any site is contacted.
    args = (
        f'--resume "{state_path}" --local --no-color --site GitHub '
        "zzz-not-a-real-user-xyz"
    )
    with pytest.raises(InteractivesSubprocessError, match="corrupt"):
        Interactives.run_cli(args)
    with pytest.raises(InteractivesSubprocessError, match="delete this file"):
        Interactives.run_cli(args)


def test_sherlock_without_checkpoint_queries_everything_as_before(local_server):
    port = local_server
    site_data = {
        "FoundSite": {
            "urlMain": f"http://127.0.0.1:{port}",
            "url": f"http://127.0.0.1:{port}/found/{{}}",
            "errorType": "message",
            "errorMsg": "no such user",
        },
    }
    # No checkpoint argument at all: unchanged behaviour, one request.
    results = sherlock("alice", site_data, QueryNotify())
    assert results["FoundSite"]["status"].status is QueryStatus.CLAIMED
    assert _CountingHandler.counts == {"found": 1, "missing": 0}
