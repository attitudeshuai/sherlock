"""Tests for the site-selection ledger (upstream exclusions + age-rating filter).

All tests are offline: the exclusions fetch is stubbed via monkeypatch.
"""
import hashlib
import json

import pytest
import requests

from sherlock_project import sites as sites_module
from sherlock_project.sites import (
    AGE_RATING_SOURCE,
    EXCLUSIONS_URL,
    SitesInformation,
)


class FakeResponse:
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


MANIFEST = {
    "$schema": "https://example.invalid/schema.json",
    "Alpha": {
        "urlMain": "https://alpha.example",
        "url": "https://alpha.example/{}",
        "username_claimed": "alice",
        "isNSFW": True,
    },
    "Beta": {
        "urlMain": "https://beta.example",
        "url": "https://beta.example/{}",
        "username_claimed": "bob",
    },
    "Gamma": {
        "urlMain": "https://gamma.example",
        "url": "https://gamma.example/{}",
        "username_claimed": "gwen",
    },
}

EXCLUSIONS_BODY = "Beta\nGhost\n\nGamma\n"
EXCLUSIONS_HEADERS = {"ETag": '"abc123"', "Last-Modified": "Thu, 23 Oct 2025 10:00:00 GMT"}


@pytest.fixture()
def manifest_path(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    return str(path)


@pytest.fixture()
def successful_exclusions(monkeypatch):
    def fake_get(url=None, timeout=None, **kwargs):
        assert url == EXCLUSIONS_URL
        return FakeResponse(200, EXCLUSIONS_BODY, dict(EXCLUSIONS_HEADERS))

    monkeypatch.setattr(sites_module.requests, "get", fake_get)


@pytest.fixture()
def failing_exclusions(monkeypatch):
    def fake_get(url=None, timeout=None, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(sites_module.requests, "get", fake_get)


@pytest.fixture()
def server_error_exclusions(monkeypatch):
    def fake_get(url=None, timeout=None, **kwargs):
        return FakeResponse(503, "service unavailable")

    monkeypatch.setattr(sites_module.requests, "get", fake_get)


def site_names(sites_obj):
    return {site.name for site in sites_obj}


def test_exclusion_success_records_full_ledger(manifest_path, successful_exclusions):
    sites_obj = SitesInformation(
        data_file_path=manifest_path, do_not_exclude=["Gamma", "Gamma"]
    )

    # Site-selection outcome is unchanged: Beta dropped, Gamma spared by name.
    assert site_names(sites_obj) == {"Alpha", "Gamma"}

    ledger = sites_obj.ledger.to_dict()
    assert ledger["manifest_entries"] == 3
    assert ledger["sites_loaded"] == 2

    exclusions = ledger["exclusions"]
    assert exclusions["state"] == "applied"
    assert exclusions["requested"] is True
    assert exclusions["fetched"] is True
    assert exclusions["status_code"] == 200
    assert exclusions["source_url"] == EXCLUSIONS_URL
    assert exclusions["entry_count"] == 3
    assert exclusions["blank_entries"] == 1
    assert exclusions["kept_by_name"] == ["Gamma"]
    assert exclusions["listed_not_in_manifest"] == ["Ghost"]
    assert exclusions["excluded_sites"] == [
        {"name": "Beta", "source": EXCLUSIONS_URL}
    ]
    # Exact version of the fetched list.
    assert exclusions["etag"] == '"abc123"'
    assert exclusions["last_modified"] == "Thu, 23 Oct 2025 10:00:00 GMT"
    assert exclusions["fetched_at"] is not None
    assert (
        exclusions["content_sha256"]
        == hashlib.sha256(EXCLUSIONS_BODY.encode("utf-8")).hexdigest()
    )

    age = ledger["age_rating_filter"]
    assert age["applied"] is False
    assert age["filtered_count"] == 0


def test_exclusion_fetch_failure_is_recorded_and_warning_kept(
    manifest_path, failing_exclusions, capsys
):
    sites_obj = SitesInformation(data_file_path=manifest_path)

    # Nothing is dropped when the fetch fails ...
    assert site_names(sites_obj) == {"Alpha", "Beta", "Gamma"}

    ledger = sites_obj.ledger.to_dict()
    exclusions = ledger["exclusions"]
    assert exclusions["state"] == "unavailable"
    assert exclusions["fetched"] is False
    assert "ConnectionError" in exclusions["error"]
    assert "boom" in exclusions["error"]
    # ... and the degraded behavior is explicitly recorded.
    assert exclusions["fallback"] is not None
    assert "without the upstream exclusion list" in exclusions["fallback"]
    assert exclusions["excluded_sites"] == []

    # The historical one-line warning is still printed, unchanged.
    captured = capsys.readouterr()
    assert (
        "Warning: Could not load exclusions, continuing without them."
        in captured.out
    )


def test_non_200_is_recorded_as_failure_without_new_output(
    manifest_path, server_error_exclusions, capsys
):
    sites_obj = SitesInformation(data_file_path=manifest_path)

    assert site_names(sites_obj) == {"Alpha", "Beta", "Gamma"}

    exclusions = sites_obj.ledger.to_dict()["exclusions"]
    assert exclusions["state"] == "unavailable"
    assert exclusions["fetched"] is False
    assert exclusions["status_code"] == 503
    assert "503" in exclusions["error"]
    assert exclusions["fallback"] is not None

    # A non-200 response was historically silent on stdout; that stays true.
    captured = capsys.readouterr()
    assert "Warning" not in captured.out


def test_disabled_exclusions_need_no_fetch(manifest_path, monkeypatch):
    def forbid_fetch(url=None, timeout=None, **kwargs):
        raise AssertionError("exclusions must not be fetched when disabled")

    monkeypatch.setattr(sites_module.requests, "get", forbid_fetch)

    sites_obj = SitesInformation(
        data_file_path=manifest_path, honor_exclusions=False
    )

    assert site_names(sites_obj) == {"Alpha", "Beta", "Gamma"}
    exclusions = sites_obj.ledger.to_dict()["exclusions"]
    assert exclusions["state"] == "disabled"
    assert exclusions["requested"] is False
    assert exclusions["fetched"] is False


def test_nsfw_filter_is_recorded_with_kept_by_name(manifest_path):
    sites_obj = SitesInformation(
        data_file_path=manifest_path, honor_exclusions=False
    )
    sites_obj.remove_nsfw_sites()

    assert site_names(sites_obj) == {"Beta", "Gamma"}

    age = sites_obj.ledger.to_dict()["age_rating_filter"]
    assert age["applied"] is True
    assert age["filtered_count"] == 1
    assert age["filtered_sites"] == [
        {"name": "Alpha", "source": AGE_RATING_SOURCE}
    ]
    assert age["kept_by_name"] == []

    # Case-insensitive name protection, same as the pre-existing --site behavior.
    sites_obj = SitesInformation(
        data_file_path=manifest_path, honor_exclusions=False
    )
    sites_obj.remove_nsfw_sites(do_not_remove=["alpha"])
    assert site_names(sites_obj) == {"Alpha", "Beta", "Gamma"}
    age = sites_obj.ledger.to_dict()["age_rating_filter"]
    assert age["filtered_count"] == 0
    assert age["kept_by_name"] == ["Alpha"]


def test_coverage_lines_break_down_the_gap(manifest_path, successful_exclusions):
    # Gamma is spared from the upstream list by name; Beta is excluded.
    sites_obj = SitesInformation(
        data_file_path=manifest_path, do_not_exclude=["Gamma"]
    )
    sites_obj.remove_nsfw_sites()

    # Beta excluded upstream, Alpha age-filtered: only Gamma is scanned.
    assert site_names(sites_obj) == {"Gamma"}
    lines = sites_obj.ledger.coverage_lines(scanned_count=1)
    joined = "\n".join(lines)

    assert "scanned 1 of 3 manifest sites (2 not covered)" in joined
    assert "1 removed by upstream exclusion list" in joined
    assert "1 removed by age-rating (NSFW) filter" in joined
    assert "not selected for this run" not in joined
    assert "Upstream exclusion list: applied" in joined
    assert "sha256:" in joined


def test_coverage_lines_account_for_targeted_site_selection(
    manifest_path, successful_exclusions
):
    # Beta survives by name, Gamma is upstream-excluded; the run scans one site.
    sites_obj = SitesInformation(
        data_file_path=manifest_path, do_not_exclude=["Beta"]
    )
    assert site_names(sites_obj) == {"Alpha", "Beta"}
    lines = sites_obj.ledger.coverage_lines(scanned_count=1)
    joined = "\n".join(lines)

    assert "scanned 1 of 3 manifest sites (2 not covered)" in joined
    assert "1 removed by upstream exclusion list" in joined
    assert "1 not selected for this run" in joined


def test_coverage_lines_report_fetch_failure(manifest_path, failing_exclusions):
    sites_obj = SitesInformation(data_file_path=manifest_path)
    joined = "\n".join(sites_obj.ledger.coverage_lines(scanned_count=3))

    assert "scanned 3 of 3 manifest sites (0 not covered)" in joined
    assert "NOT applied; fetch failed" in joined
    assert "Fallback:" in joined


def test_render_contains_names_sources_and_failure(manifest_path, failing_exclusions):
    sites_obj = SitesInformation(data_file_path=manifest_path)
    rendered = sites_obj.ledger.render()

    assert "Site selection ledger" in rendered
    assert "unavailable" in rendered
    assert EXCLUSIONS_URL in rendered
    assert "Fallback" in rendered


def test_loading_prints_nothing_extra_on_success(
    manifest_path, successful_exclusions, capsys
):
    SitesInformation(data_file_path=manifest_path)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
