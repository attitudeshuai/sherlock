import json

import pytest
import requests
from sherlock_project import sherlock
from sherlock_interactives import Interactives
from sherlock_interactives import InteractivesSubprocessError

def test_remove_nsfw(sites_obj):
    nsfw_target: str = 'Xvideos'
    assert nsfw_target in {site.name: site.information for site in sites_obj}
    sites_obj.remove_nsfw_sites()
    assert nsfw_target not in {site.name: site.information for site in sites_obj}


# Parametrized sites should *not* include Motherless, which is acting as the control
@pytest.mark.parametrize('nsfwsites', [
    ['Xvideos'],
    ['Xvideos', 'Erome'],
])
def test_nsfw_explicit_selection(sites_obj, nsfwsites):
    for site in nsfwsites:
        assert site in {site.name: site.information for site in sites_obj}
    sites_obj.remove_nsfw_sites(do_not_remove=nsfwsites)
    for site in nsfwsites:
        assert site in {site.name: site.information for site in sites_obj}
        assert 'Motherless' not in {site.name: site.information for site in sites_obj}

def test_wildcard_username_expansion():
    assert sherlock.check_for_parameter('test{?}test') is True
    assert sherlock.check_for_parameter('test{.}test') is False
    assert sherlock.check_for_parameter('test{}test') is False
    assert sherlock.check_for_parameter('testtest') is False
    assert sherlock.check_for_parameter('test{?test') is False
    assert sherlock.check_for_parameter('test?}test') is False
    assert sherlock.multiple_usernames('test{?}test') == ["test_test" , "test-test" , "test.test"]


@pytest.mark.parametrize('cliargs', [
    '',
    '--site urghrtuight --egiotr',
    '--',
])
def test_no_usernames_provided(cliargs):
    with pytest.raises(InteractivesSubprocessError, match=r"error: the following arguments are required: USERNAMES"):
        Interactives.run_cli(cliargs)


def _plan_sites(plan):
    return {entry["site"]: entry for entry in plan["sites"]}


def test_build_probe_plan_shapes_requests_without_network(monkeypatch):
    def _no_requests(*args, **kwargs):
        raise AssertionError("dry-run probe plan must not perform any request")

    monkeypatch.setattr(requests.sessions.Session, "request", _no_requests)

    site_data = {
        "Status Site": {
            "url": "https://status.example.com/u/{}",
            "urlMain": "https://status.example.com",
            "errorType": "status_code",
            "errorCode": 404,
            "username_claimed": "blue",
        },
        "Message Site": {
            "url": "https://message.example.com/u/{}",
            "urlMain": "https://message.example.com",
            "errorType": "message",
            "errorMsg": ["gone", "missing"],
            "regexCheck": "^[a-z]{2,4}$",
            "username_claimed": "blue",
        },
        "Post Site": {
            "url": "https://post.example.com/profile",
            "urlMain": "https://post.example.com",
            "urlProbe": "https://api.post.example.com/lookup?q={}",
            "errorType": "response_url",
            "errorUrl": "https://post.example.com/missing",
            "request_method": "POST",
            "request_payload": {"user": "{}", "n": 1},
            "headers": {"X-Test": "yes"},
            "username_claimed": "blue",
        },
    }

    plan = sherlock.build_probe_plan("blue", site_data, timeout=12.5)

    assert plan["username"] == "blue"
    assert plan["expanded_from"] is None
    assert plan["skipped"] == []
    assert [entry["site"] for entry in plan["sites"]] == [
        "Message Site",
        "Post Site",
        "Status Site",
    ]

    by_site = _plan_sites(plan)

    # status_code detection without an explicit method defaults to HEAD.
    status_site = by_site["Status Site"]
    assert status_site["url_user"] == "https://status.example.com/u/blue"
    assert status_site["request"]["method"] == "HEAD"
    assert status_site["request"]["url"] == "https://status.example.com/u/blue"
    assert status_site["request"]["payload"] is None
    assert status_site["request"]["allow_redirects"] is True
    assert status_site["request"]["timeout"] == 12.5
    assert status_site["detection"] == {
        "methods": ["status_code"],
        "error_messages": None,
        "error_codes": [404],
        "error_url": None,
    }

    # Explicit POST with a dedicated probe URL, JSON payload and no redirects.
    post_site = by_site["Post Site"]
    assert post_site["request"]["method"] == "POST"
    assert post_site["request"]["url"] == "https://api.post.example.com/lookup?q=blue"
    assert post_site["request"]["payload"] == {"user": "blue", "n": 1}
    assert post_site["request"]["allow_redirects"] is False
    assert post_site["request"]["headers"]["X-Test"] == "yes"
    assert post_site["detection"]["methods"] == ["response_url"]
    assert post_site["detection"]["error_url"] == "https://post.example.com/missing"

    # Plain message detection defaults to GET and allows redirects.
    message_site = by_site["Message Site"]
    assert message_site["request"]["method"] == "GET"
    assert message_site["request"]["allow_redirects"] is True
    assert message_site["detection"] == {
        "methods": ["message"],
        "error_messages": ["gone", "missing"],
        "error_codes": None,
        "error_url": None,
    }


def test_build_probe_plan_lists_illegal_sites_with_reason():
    site_data = {
        "Strict Site": {
            "url": "https://strict.example.com/{}",
            "urlMain": "https://strict.example.com",
            "errorType": "message",
            "errorMsg": "nope",
            "regexCheck": "^[a-z]{2,4}$",
            "username_claimed": "blue",
        },
    }
    plan = sherlock.build_probe_plan("TooLong1", site_data, timeout=60)
    assert plan["sites"] == []
    assert plan["skipped"] == [
        {
            "site": "Strict Site",
            "reason": "illegal_username",
            "username": "TooLong1",
            "regex_check": "^[a-z]{2,4}$",
            "url_user": "https://strict.example.com/TooLong1",
        }
    ]


def test_build_probe_plan_space_interpolation_matches_runtime():
    # Profile URLs percent-encode spaces; a separate urlProbe is interpolated
    # with the raw username, exactly like sherlock().
    site_data = {
        "Spaced": {
            "url": "https://example.com/u/{}",
            "urlMain": "https://example.com",
            "urlProbe": "https://api.example.com/lookup?name={}",
            "errorType": "status_code",
            "username_claimed": "blue",
        },
    }
    plan = sherlock.build_probe_plan("a b", site_data, timeout=60)
    entry = plan["sites"][0]
    assert entry["url_user"] == "https://example.com/u/a%20b"
    assert entry["request"]["url"] == "https://api.example.com/lookup?name=a b"


def test_build_probe_plan_does_not_mutate_site_data():
    payload = {"name": "{}"}
    site_data = {
        "Post": {
            "url": "https://example.com",
            "urlMain": "https://example.com",
            "urlProbe": "https://example.com/api",
            "errorType": "message",
            "errorMsg": "no",
            "request_method": "POST",
            "request_payload": payload,
            "username_claimed": "blue",
        },
    }
    sherlock.build_probe_plan("blue", site_data, timeout=60)
    assert site_data["Post"]["request_payload"] == {"name": "{}"}
    assert "request_future" not in site_data["Post"]


def test_dry_run_cli_outputs_machine_readable_plan():
    output = Interactives.run_cli("--dry-run --site GitHub --site Anilist blue")
    document = json.loads(output)

    assert document["format"] == "sherlock.probe-plan"
    assert document["format_version"] == 1
    assert document["dry_run"] is True
    assert document["username_expansion"] == []
    assert document["manifest"]["exclusions_honored"] is False
    assert document["options"]["sites_requested"] == ["GitHub", "Anilist"]

    assert len(document["plans"]) == 1
    plan = document["plans"][0]
    assert plan["username"] == "blue"
    assert plan["expanded_from"] is None
    assert plan["skipped"] == []

    by_site = _plan_sites(plan)
    assert by_site["GitHub"]["request"]["method"] == "HEAD"
    assert by_site["GitHub"]["request"]["url"] == "https://www.github.com/blue"
    assert by_site["GitHub"]["detection"]["methods"] == ["status_code"]
    assert by_site["Anilist"]["request"]["method"] == "POST"
    assert by_site["Anilist"]["request"]["payload"]["variables"] == {"name": "blue"}


def test_dry_run_cli_lists_expansions_skips_and_timeout():
    output = Interactives.run_cli(
        "--dry-run --site GitHub --site Anilist --timeout 5 user{?}1"
    )
    document = json.loads(output)

    assert document["username_expansion"] == [
        {
            "input": "user{?}1",
            "placeholder": "{?}",
            "symbols": ["_", "-", "."],
            "variants": ["user_1", "user-1", "user.1"],
        }
    ]
    assert [plan["username"] for plan in document["plans"]] == [
        "user_1",
        "user-1",
        "user.1",
    ]
    assert all(plan["expanded_from"] == "user{?}1" for plan in document["plans"])

    skipped_by_username = {
        plan["username"]: [entry["site"] for entry in plan["skipped"]]
        for plan in document["plans"]
    }
    # Anilist allows only [A-Za-z0-9]; every variant is illegal there.
    # GitHub additionally allows internal hyphens, so only "_" and "." fail.
    assert "Anilist" in skipped_by_username["user_1"]
    assert "GitHub" in skipped_by_username["user_1"]
    assert "GitHub" in skipped_by_username["user.1"]
    assert "GitHub" not in skipped_by_username["user-1"]
    for skipped in document["plans"][0]["skipped"]:
        assert skipped["reason"] == "illegal_username"
        assert "regex_check" in skipped

    # The effective timeout reaches every planned request.
    assert document["options"]["timeout"] == 5.0
    for plan in document["plans"]:
        for entry in plan["sites"]:
            assert entry["request"]["timeout"] == 5.0


@pytest.mark.parametrize("flag", [
    "--csv",
    "--xlsx",
    "--txt",
    "-o plan.txt",
    "-fo outdir",
    "-b",
])
def test_dry_run_rejects_file_and_browser_options(flag):
    with pytest.raises(
        InteractivesSubprocessError, match="--dry-run cannot be combined"
    ):
        Interactives.run_cli(f"--dry-run --site GitHub blue {flag}")


def test_dry_run_rejects_remote_manifest():
    with pytest.raises(
        InteractivesSubprocessError, match="performs no network requests"
    ):
        Interactives.run_cli(
            "--dry-run --site GitHub blue --json https://example.com/data.json"
        )
