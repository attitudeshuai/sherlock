"""Offline tests for redirect-chain recording.

These tests drive sherlock.sherlock() with a fake futures session that plays
back scripted response sequences, so redirect behaviour can be asserted
without any network access.  They cover:

* followed chains (status-code / message sites): ordered hops, target
  addresses, final landing URL;
* refused first hop on response_url sites (redirects still disabled, the
  rejected status code and Location target are registered);
* redirect loops cut off at the hop limit (bounded chain + truncation flag,
  verdict unchanged);
* the chain serialization used by CSV/XLSX exports.
"""
import pytest
import requests

from sherlock_project import sherlock as sherlock_mod
from sherlock_project.notify import QueryNotify
from sherlock_project.result import QueryStatus, RedirectHop
from sherlock_project.sherlock import (
    MAX_REDIRECT_HOPS,
    build_redirect_chain,
    format_redirect_chain,
)


class FakeResponse:
    """Minimal stand-in for the parts of requests.Response sherlock uses."""

    def __init__(self, url, status_code=200, location=None, text=""):
        self.url = url
        self.status_code = status_code
        self.headers = {"Location": location} if location is not None else {}
        self.text = text
        self.encoding = "utf-8"
        self.elapsed = 0.01
        self.history = []

    @property
    def is_redirect(self):
        return (
            "Location" in self.headers
            and self.status_code in (301, 302, 303, 307, 308)
        )


class FakeFuture:
    """Plays scripted responses through the response hooks, then returns.

    Mirrors requests: the response hook fires once per observed response
    (including every intermediate 3xx), and the returned final response has
    the preceding responses attached as history.
    """

    def __init__(self, responses, hooks, raise_at_end=None):
        self._responses = responses
        self._hooks = list(hooks or [])
        self._raise_at_end = raise_at_end

    def result(self):
        for response in self._responses:
            for hook in self._hooks:
                hook(response)
        if self._raise_at_end is not None:
            raise self._raise_at_end
        final = self._responses[-1]
        final.history = self._responses[:-1]
        return final


def _loop_responses():
    """31 redirect responses bouncing between /b and /a forever.

    Each response's Location equals the URL of the following response, as in
    a real redirect loop.
    """
    base = "https://loop.example/u/alice"
    urls = [base]
    for index in range(MAX_REDIRECT_HOPS + 1):
        urls.append(
            f"https://loop.example/{'b' if index % 2 == 0 else 'a'}"
        )

    responses = []
    for index in range(MAX_REDIRECT_HOPS + 1):
        if index < MAX_REDIRECT_HOPS:
            location = urls[index + 1]
        else:
            # The hop refused because the limit was hit still carries a
            # Location header pointing back into the cycle.
            location = "https://loop.example/a"
        responses.append(FakeResponse(urls[index], 302, location=location))
    return responses


def _script_for(url):
    """Return (responses, terminal_exception) for a given probe URL."""
    if "follow.example" in url:
        return [
            FakeResponse(url, 302, location="/u/alice/real"),
            FakeResponse(
                "https://follow.example/u/alice/real",
                301,
                location="https://cdn.follow.example/p/alice",
            ),
            FakeResponse(
                "https://cdn.follow.example/p/alice",
                200,
                text="profile page",
            ),
        ], None

    if "missing.example" in url:
        # Nonexistent user is bounced to a registration page.
        return [
            FakeResponse(url, 302, location="/register"),
            FakeResponse(
                "https://missing.example/register",
                200,
                text="user not found, please register",
            ),
        ], None

    if "respurl.example" in url:
        if url.endswith("/u/exists"):
            return [FakeResponse(url, 200, text="profile")], None
        return [FakeResponse(url, 302, location="/signup")], None

    if "status.example" in url:
        return [
            FakeResponse(url, 302, location="/p/alice"),
            FakeResponse("https://status.example/p/alice", 200, text="ok"),
        ], None

    if "loop.example" in url:
        return _loop_responses(), requests.exceptions.TooManyRedirects(
            f"Exceeded {MAX_REDIRECT_HOPS} redirects.",
            response=None,
        )

    if "direct.example" in url:
        return [FakeResponse(url, 200, text="profile page")], None

    raise AssertionError(f"unexpected probe url: {url}")


class FakeFuturesSession:
    """Drop-in replacement for SherlockFuturesSession used in these tests."""

    calls = {}

    def __init__(self, max_workers=None, session=None):
        pass

    def _request(self, method, url, **kwargs):
        FakeFuturesSession.calls[url] = kwargs
        responses, terminal_exception = _script_for(url)
        hooks = (kwargs.get("hooks") or {}).get("response", [])
        if callable(hooks):
            hooks = [hooks]
        return FakeFuture(responses, hooks, terminal_exception)

    def get(self, url, **kwargs):
        return self._request("GET", url, **kwargs)

    def head(self, url, **kwargs):
        return self._request("HEAD", url, **kwargs)

    def post(self, url, **kwargs):
        return self._request("POST", url, **kwargs)

    def put(self, url, **kwargs):
        return self._request("PUT", url, **kwargs)


def _info(url, error_type, **extra):
    information = {
        "urlMain": "https://main.example/",
        "url": url,
        "errorType": error_type,
    }
    information.update(extra)
    return information


def _site(url, error_type, **extra):
    return {"TestSite": _info(url, error_type, **extra)}


def _run(monkeypatch, site_data, username="alice"):
    monkeypatch.setattr(sherlock_mod, "SherlockFuturesSession", FakeFuturesSession)
    results = sherlock_mod.sherlock(
        username=username,
        site_data=site_data,
        query_notify=QueryNotify(),
    )
    return results["TestSite"]


def test_followed_chain_records_hops_and_final_landing(monkeypatch):
    site_data = _site(
        "https://follow.example/u/{}",
        "message",
        errorMsg="this profile does not exist",
    )
    result = _run(monkeypatch, site_data)

    # Existing verdict semantics are preserved.
    assert result["status"].status is QueryStatus.CLAIMED

    chain = result["redirect_chain"]
    assert [hop.status_code for hop in chain] == [302, 301]
    assert chain[0].location == "https://follow.example/u/alice/real"
    assert chain[1].location == "https://cdn.follow.example/p/alice"
    assert result["final_url"] == "https://cdn.follow.example/p/alice"
    assert result["redirect_truncated"] is False
    assert result["status"].final_url == result["final_url"]
    assert result["status"].redirect_chain is chain


def test_redirect_to_register_page_keeps_chain_for_fp_investigation(monkeypatch):
    site_data = _site(
        "https://missing.example/u/{}",
        "message",
        errorMsg="user not found",
    )
    result = _run(monkeypatch, site_data)

    # Verdict says unavailable, but the chain now explains where it landed.
    assert result["status"].status is QueryStatus.AVAILABLE
    chain = result["redirect_chain"]
    assert len(chain) == 1
    assert chain[0].status_code == 302
    assert chain[0].location == "https://missing.example/register"
    assert result["final_url"] == "https://missing.example/register"


def test_status_code_site_chain_recorded_without_changing_verdict(monkeypatch):
    site_data = _site(
        "https://status.example/u/{}",
        "status_code",
        errorCode=404,
    )
    result = _run(monkeypatch, site_data)

    assert result["status"].status is QueryStatus.CLAIMED
    assert result["http_status"] == 200
    chain = result["redirect_chain"]
    assert [(hop.status_code, hop.location) for hop in chain] == [
        (302, "https://status.example/p/alice"),
    ]
    assert result["final_url"] == "https://status.example/p/alice"


def test_response_url_site_redirects_still_disabled_but_first_hop_logged(
    monkeypatch,
):
    site_data = _site("https://respurl.example/u/{}", "response_url")
    result = _run(monkeypatch, site_data)

    # Verdict semantics unchanged: a 3xx means unavailable.
    assert result["status"].status is QueryStatus.AVAILABLE
    # The request really was issued with redirects disabled.
    assert FakeFuturesSession.calls["https://respurl.example/u/alice"][
        "allow_redirects"
    ] is False

    chain = result["redirect_chain"]
    assert len(chain) == 1
    assert chain[0].status_code == 302
    # Relative Location resolved against the request URL, straight from the
    # refused response header.
    assert chain[0].location == "https://respurl.example/signup"
    assert result["final_url"] == ""
    assert result["redirect_truncated"] is False


def test_response_url_site_no_redirect_has_empty_chain(monkeypatch):
    site_data = _site("https://respurl.example/u/{}", "response_url")
    result = _run(monkeypatch, site_data, username="exists")

    assert result["status"].status is QueryStatus.CLAIMED
    assert result["redirect_chain"] == []
    assert result["final_url"] == ""


def test_redirect_loop_is_bounded_and_marked_truncated(monkeypatch):
    site_data = _site(
        "https://loop.example/u/{}",
        "message",
        errorMsg="not found",
    )
    result = _run(monkeypatch, site_data)

    # Loop failures were and remain UNKNOWN; the trail is what is new.
    assert result["status"].status is QueryStatus.UNKNOWN
    assert result["status"].redirect_truncated is True
    assert result["redirect_truncated"] is True

    chain = result["redirect_chain"]
    # 30 followed hops plus one trailing un-followed hop, never unbounded.
    assert len(chain) == MAX_REDIRECT_HOPS + 1
    assert all(isinstance(hop, RedirectHop) for hop in chain)
    assert chain[0].status_code == 302
    # The cycle is visible in the recorded targets.
    assert chain[0].location.endswith("/b")
    assert chain[1].location.endswith("/a")
    # Last entry records the hop refused because the limit was hit.
    assert chain[-1].location == "https://loop.example/a"
    assert result["final_url"] == ""


def test_direct_response_has_no_chain(monkeypatch):
    site_data = _site(
        "https://direct.example/u/{}",
        "message",
        errorMsg="not found",
    )
    result = _run(monkeypatch, site_data)

    assert result["status"].status is QueryStatus.CLAIMED
    assert result["redirect_chain"] == []
    assert result["final_url"] == ""
    assert result["redirect_truncated"] is False


def test_format_redirect_chain_plain_and_truncated():
    hops = [
        RedirectHop(302, "https://example.com/a"),
        RedirectHop(301, "https://example.com/b"),
    ]
    assert format_redirect_chain(hops, False) == (
        "302 https://example.com/a | 301 https://example.com/b"
    )
    assert format_redirect_chain(hops, True).endswith("| TRUNCATED")
    assert format_redirect_chain([], False) == ""
    assert format_redirect_chain([], True) == "TRUNCATED"


def test_build_redirect_chain_refused_relative_location():
    response = FakeResponse("https://x.test/u/me", 302, location="../login")
    hops, final_url = build_redirect_chain(
        response=response,
        redirect_trace=[response],
        follow_redirects=False,
        truncated=False,
    )
    assert final_url is None
    assert len(hops) == 1
    assert hops[0].status_code == 302
    assert hops[0].location == "https://x.test/login"


def test_traces_stay_isolated_between_concurrent_sites(monkeypatch):
    """Regression: per-request trace hooks must not share state across sites.

    All requests are submitted before any future is collected, and hooks run
    later (in worker threads).  A late-bound loop closure would append every
    site's responses to the last site's trace, corrupting the salvaged chain
    of a truncated site.
    """
    site_data = {
        "Follow": _info(
            "https://follow.example/u/{}",
            "message",
            errorMsg="this profile does not exist",
        ),
        "Missing": _info(
            "https://missing.example/u/{}",
            "message",
            errorMsg="user not found",
        ),
        "RespUrl": _info("https://respurl.example/u/{}", "response_url"),
        "Loop": _info(
            "https://loop.example/u/{}",
            "status_code",
            errorCode=404,
        ),
    }
    monkeypatch.setattr(sherlock_mod, "SherlockFuturesSession", FakeFuturesSession)
    results = sherlock_mod.sherlock(
        username="alice",
        site_data=site_data,
        query_notify=QueryNotify(),
    )

    loop_result = results["Loop"]
    assert loop_result["redirect_truncated"] is True
    loop_chain = loop_result["redirect_chain"]
    assert len(loop_chain) == MAX_REDIRECT_HOPS + 1
    # Not a single foreign response may leak into the loop's chain.
    assert all(
        "loop.example" in (hop.location or "") for hop in loop_chain
    )

    assert results["Follow"]["status"].status is QueryStatus.CLAIMED
    assert len(results["Follow"]["redirect_chain"]) == 2
    assert results["Missing"]["status"].status is QueryStatus.AVAILABLE
    assert len(results["Missing"]["redirect_chain"]) == 1
    assert results["RespUrl"]["status"].status is QueryStatus.AVAILABLE
    assert len(results["RespUrl"]["redirect_chain"]) == 1

    # The raw per-site traces captured by the hooks are isolated too.
    assert len(site_data["Follow"]["redirect_trace"]) == 3
    assert all(
        "follow.example" in resp.url
        for resp in site_data["Follow"]["redirect_trace"]
    )
    assert all(
        "loop.example" in resp.url
        for resp in site_data["Loop"]["redirect_trace"]
    )


def test_get_response_flags_too_many_redirects():
    from concurrent.futures import Future
    from sherlock_project.sherlock import get_response

    future = Future()
    future.set_exception(
        requests.exceptions.TooManyRedirects(
            f"Exceeded {MAX_REDIRECT_HOPS} redirects."
        )
    )
    response, error_context, exception_text, truncated = get_response(
        request_future=future,
        error_type=["message"],
        social_network="TestSite",
    )
    assert response is None
    assert error_context == "Too Many Redirects"
    assert truncated is True


def test_illegal_username_produces_empty_chain(monkeypatch):
    site_data = _site(
        "https://follow.example/u/{}",
        "message",
        errorMsg="nope",
        regexCheck=r"^[a-z]+$",
    )
    result = _run(monkeypatch, site_data, username="Bad-Name!")

    assert result["status"].status is QueryStatus.ILLEGAL
    assert result["redirect_chain"] == []
    assert result["final_url"] == ""
    assert result["redirect_truncated"] is False
