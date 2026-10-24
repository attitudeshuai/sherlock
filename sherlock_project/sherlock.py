#! /usr/bin/env python3

"""
Sherlock: Find Usernames Across Social Networks Module

This module contains the main logic to search for usernames at social
networks.
"""

import sys

try:
    from sherlock_project.__init__ import import_error_test_var # noqa: F401
except ImportError:
    print("Did you run Sherlock with `python3 sherlock/sherlock.py ...`?")
    print("This is an outdated method. Please see https://sherlockproject.xyz/installation for up to date instructions.")
    sys.exit(1)

import csv
import signal
import pandas as pd
import os
import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from json import loads as json_loads
from time import monotonic
from typing import Optional

import requests
from concurrent.futures import TimeoutError as FutureTimeoutError
from requests_futures.sessions import FuturesSession

from sherlock_project.__init__ import (
    __longname__,
    __shortname__,
    __version__,
    forge_api_latest_release,
)

from sherlock_project.result import QueryStatus
from sherlock_project.result import QueryResult
from sherlock_project.notify import QueryNotify
from sherlock_project.notify import QueryNotifyPrint
from sherlock_project.sites import SitesInformation
from colorama import init
from argparse import ArgumentTypeError


class SherlockFuturesSession(FuturesSession):
    def request(self, method, url, hooks=None, *args, **kwargs):
        """Request URL.

        This extends the FuturesSession request method to calculate a response
        time metric to each request.

        It is taken (almost) directly from the following Stack Overflow answer:
        https://github.com/ross/requests-futures#working-in-the-background

        Keyword Arguments:
        self                   -- This object.
        method                 -- String containing method desired for request.
        url                    -- String containing URL for request.
        hooks                  -- Dictionary containing hooks to execute after
                                  request finishes.
        args                   -- Arguments.
        kwargs                 -- Keyword arguments.

        Return Value:
        Request object.
        """
        # Record the start time for the request.
        if hooks is None:
            hooks = {}
        start = monotonic()

        def response_time(resp, *args, **kwargs):
            """Response Time Hook.

            Keyword Arguments:
            resp                   -- Response object.
            args                   -- Arguments.
            kwargs                 -- Keyword arguments.

            Return Value:
            Nothing.
            """
            resp.elapsed = monotonic() - start


        # Install hook to execute when response completes.
        # Make sure that the time measurement hook is first, so we will not
        # track any later hook's execution time.
        try:
            if isinstance(hooks["response"], list):
                hooks["response"].insert(0, response_time)
            elif isinstance(hooks["response"], tuple):
                # Convert tuple to list and insert time measurement hook first.
                hooks["response"] = list(hooks["response"])
                hooks["response"].insert(0, response_time)
            else:
                # Must have previously contained a single hook function,
                # so convert to list.
                hooks["response"] = [response_time, hooks["response"]]
        except KeyError:
            # No response hook was already defined, so install it ourselves.
            hooks["response"] = [response_time]

        return super(SherlockFuturesSession, self).request(
            method, url, hooks=hooks, *args, **kwargs
        )


def get_response(request_future, error_type, social_network, result_timeout=None):
    # Default for Response object if some failure occurs.
    response = None

    error_context = "General Unknown Error"
    exception_text = None
    budget_timed_out = False
    try:
        response = request_future.result(timeout=result_timeout)
        if response.status_code:
            # Status code exists in response object
            error_context = None
    except FutureTimeoutError:
        # The batch time budget ran out before this request returned.
        budget_timed_out = True
        error_context = "Batch Timeout"
    except requests.exceptions.HTTPError as errh:
        error_context = "HTTP Error"
        exception_text = str(errh)
    except requests.exceptions.ProxyError as errp:
        error_context = "Proxy Error"
        exception_text = str(errp)
    except requests.exceptions.ConnectionError as errc:
        error_context = "Error Connecting"
        exception_text = str(errc)
    except requests.exceptions.Timeout as errt:
        error_context = "Timeout Error"
        exception_text = str(errt)
    except requests.exceptions.RequestException as err:
        error_context = "Unknown Error"
        exception_text = str(err)
    except UnicodeError as err:
        error_context = "Encoding Error"
        exception_text = str(err)

    return response, error_context, exception_text, budget_timed_out


def interpolate_string(input_object, username):
    if isinstance(input_object, str):
        return input_object.replace("{}", username)
    elif isinstance(input_object, dict):
        return {k: interpolate_string(v, username) for k, v in input_object.items()}
    elif isinstance(input_object, list):
        return [interpolate_string(i, username) for i in input_object]
    return input_object


def check_for_parameter(username):
    """checks if {?} exists in the username
    if exist it means that sherlock is looking for more multiple username"""
    return "{?}" in username


checksymbols = ["_", "-", "."]


def multiple_usernames(username):
    """replace the parameter with with symbols and return a list of usernames"""
    allUsernames = []
    for i in checksymbols:
        allUsernames.append(username.replace("{?}", i))
    return allUsernames


BUDGET_POLICIES = ("abandon", "wait")


def _coerce_timeout_seconds(value, location):
    """Validate a single timeout value, returning a positive float.

    Keyword Arguments:
    value                  -- Timeout value declared in a manifest (must be
                              a positive number).
    location               -- String identifying the offending declaration
                              for error messages.

    Return Value:
    Floating point number of seconds.

    NOTE:  Raises ValueError if the value is not a positive number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Invalid timeout for {location}: {value!r}. "
            "Timeout must be a positive number of seconds."
        )

    float_value = float(value)
    if float_value <= 0:
        raise ValueError(
            f"Invalid timeout for {location}: {value!r}. "
            "Timeout must be a positive number of seconds."
        )

    return float_value


def resolve_site_timeouts(site_timeout, global_timeout):
    """Resolve The Effective Timeouts For One Site.

    The site manifest may declare its own timeout tier:

      * omitted/None  -> the global --timeout value covers both phases
      * a number      -> the same value covers connect and read
      * an object     -> {"connect": seconds} and/or {"read": seconds};
                         any phase not declared falls back to the global
                         --timeout value

    Keyword Arguments:
    site_timeout           -- The "timeout" entry of a site's manifest
                              (None, number, or dict).
    global_timeout         -- The global timeout (from --timeout), used as
                              the default for every undeclared phase.

    Return Value:
    Tuple of (request_timeout, connect_timeout, read_timeout, level).
    "request_timeout" is the exact object handed to requests: a scalar for
    the global tier (byte-for-byte identical to the historical behaviour)
    and a (connect, read) tuple only when a site tier is in effect.
    "level" is "global" or "site".
    """
    if site_timeout is None:
        return global_timeout, global_timeout, global_timeout, "global"

    if isinstance(site_timeout, (int, float)):
        timeout = _coerce_timeout_seconds(site_timeout, "site")
        return timeout, timeout, timeout, "site"

    if isinstance(site_timeout, dict):
        unknown_keys = set(site_timeout) - {"connect", "read"}
        if unknown_keys:
            raise ValueError(
                f"Invalid site timeout declaration: {site_timeout!r}. "
                f"Unknown key(s): {', '.join(sorted(unknown_keys))}. "
                "Only 'connect' and 'read' are allowed."
            )
        if "connect" not in site_timeout and "read" not in site_timeout:
            raise ValueError(
                f"Invalid site timeout declaration: {site_timeout!r}. "
                "Declare at least one of 'connect' or 'read'."
            )
        connect = (
            _coerce_timeout_seconds(site_timeout["connect"], "site 'connect'")
            if "connect" in site_timeout
            else global_timeout
        )
        read = (
            _coerce_timeout_seconds(site_timeout["read"], "site 'read'")
            if "read" in site_timeout
            else global_timeout
        )
        return (connect, read), connect, read, "site"

    raise ValueError(
        f"Invalid site timeout declaration: {site_timeout!r}. "
        "Use a number or an object with optional 'connect'/'read' numbers."
    )


def timeouts_are_tiered(site_data, budget):
    """Whether tiered-timeout output should be enabled for this batch.

    Tiering is considered in effect as soon as a batch time budget is
    given or any site declares its own timeout.  When neither applies,
    behaviour is identical to the historical single global timeout.
    """
    return budget is not None or any(
        net_info.get("timeout") is not None for net_info in site_data.values()
    )


def resolve_budget_policy(site_policy, global_policy):
    """Resolve The Batch-Budget Policy For One Site.

    "abandon" (default): when the batch budget is exhausted, sites still
    pending are abandoned and marked as QueryStatus.TIMEOUT.
    "wait": the site is exempted from the budget and may run until its own
    request timeout concludes.
    """
    policy = site_policy if site_policy is not None else global_policy
    if policy not in BUDGET_POLICIES:
        raise ValueError(
            f"Invalid batch budget policy: {policy!r}. "
            f"Expected one of {', '.join(BUDGET_POLICIES)}."
        )
    return policy


def sherlock(
    username: str,
    site_data: dict[str, dict[str, str]],
    query_notify: QueryNotify,
    dump_response: bool = False,
    proxy: Optional[str] = None,
    timeout: int = 60,
    budget: Optional[float] = None,
    budget_policy: str = "abandon",
) -> dict[str, dict[str, str | QueryResult]]:
    """Run Sherlock Analysis.

    Checks for existence of username on various social media sites.

    Keyword Arguments:
    username               -- String indicating username that report
                              should be created against.
    site_data              -- Dictionary containing all of the site data.
    query_notify           -- Object with base type of QueryNotify().
                              This will be used to notify the caller about
                              query results.
    proxy                  -- String indicating the proxy URL
    timeout                -- Time in seconds to wait before timing out request.
                              Default is 60 seconds. Acts as the global
                              default; individual sites may override it in
                              the manifest (per connect/read phase).
    budget                 -- Total time in seconds allowed for this whole
                              batch (one username across all sites). When
                              exhausted, sites still pending are handled per
                              their budget policy. Default of None means no
                              batch-level limit.
    budget_policy          -- Default policy for pending sites when the budget
                              is exhausted: "abandon" (mark Timed Out) or
                              "wait" (let the request finish). Individual
                              sites may override it in the manifest.

    Return Value:
    Dictionary containing results from report. Key of dictionary is the name
    of the social network site, and the value is another dictionary with
    the following keys:
        url_main:      URL of main site.
        url_user:      URL of user on site (if account exists).
        status:        QueryResult() object indicating results of test for
                       account existence.
        http_status:   HTTP status code of query which checked for existence on
                       site.
        response_text: Text that came back from request.  May be None if
                       there was an HTTP error when checking for existence.
    """

    # Notify caller that we are starting the query.
    query_notify.start(username)

    # Normal requests
    underlying_session = requests.session()

    # Limit number of workers to 20.
    # This is probably vastly overkill.
    if len(site_data) >= 20:
        max_workers = 20
    else:
        max_workers = len(site_data)

    # Create multi-threaded session for all requests.
    session = SherlockFuturesSession(
        max_workers=max_workers, session=underlying_session
    )

    # Results from analysis of all sites
    results_total = {}

    # Overall batch deadline when a total time budget was requested.
    batch_started = monotonic()
    batch_deadline = batch_started + budget if budget is not None else None
    # Set to true if at least one site is abandoned waiting for the budget.
    budget_abandoned_any = False

    # First create futures for all requests. This allows for the requests to run in parallel
    for social_network, net_info in site_data.items():
        # Results from analysis of this specific site
        results_site = {"url_main": net_info.get("urlMain")}

        # Record URL of main site

        # A user agent is needed because some sites don't return the correct
        # information since they think that we are bots (Which we actually are...)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0",
        }

        if "headers" in net_info:
            # Override/append any extra headers required by a given site.
            headers.update(net_info["headers"])

        # Resolve the effective timeout tier for this site. Sites without
        # their own declaration get the global timeout for both phases.
        (
            request_timeout,
            timeout_connect,
            timeout_read,
            timeout_level,
        ) = resolve_site_timeouts(net_info.get("timeout"), timeout)
        results_site["timeout_connect"] = timeout_connect
        results_site["timeout_read"] = timeout_read
        results_site["timeout_level"] = timeout_level

        # URL of user on site (if it exists)
        url = interpolate_string(net_info["url"], username.replace(' ', '%20'))

        # Don't make request if username is invalid for the site
        regex_check = net_info.get("regexCheck")
        if regex_check and re.search(regex_check, username) is None:
            # No need to do the check at the site: this username is not allowed.
            results_site["status"] = QueryResult(
                username,
                social_network,
                url,
                QueryStatus.ILLEGAL,
                timeout_connect=timeout_connect,
                timeout_read=timeout_read,
                timeout_level=timeout_level,
            )
            results_site["url_user"] = ""
            results_site["http_status"] = ""
            results_site["response_text"] = ""
            query_notify.update(results_site["status"])
        else:
            # URL of user on site (if it exists)
            results_site["url_user"] = url
            url_probe = net_info.get("urlProbe")
            request_method = net_info.get("request_method")
            request_payload = net_info.get("request_payload")
            request = None

            if request_method is not None:
                if request_method == "GET":
                    request = session.get
                elif request_method == "HEAD":
                    request = session.head
                elif request_method == "POST":
                    request = session.post
                elif request_method == "PUT":
                    request = session.put
                else:
                    raise RuntimeError(f"Unsupported request_method for {url}")

            if request_payload is not None:
                request_payload = interpolate_string(request_payload, username)

            if url_probe is None:
                # Probe URL is normal one seen by people out on the web.
                url_probe = url
            else:
                # There is a special URL for probing existence separate
                # from where the user profile normally can be found.
                url_probe = interpolate_string(url_probe, username)

            if request is None:
                if net_info["errorType"] == "status_code":
                    # In most cases when we are detecting by status code,
                    # it is not necessary to get the entire body:  we can
                    # detect fine with just the HEAD response.
                    request = session.head
                else:
                    # Either this detect method needs the content associated
                    # with the GET response, or this specific website will
                    # not respond properly unless we request the whole page.
                    request = session.get

            if net_info["errorType"] == "response_url":
                # Site forwards request to a different URL if username not
                # found.  Disallow the redirect so we can capture the
                # http status from the original URL request.
                allow_redirects = False
            else:
                # Allow whatever redirect that the site wants to do.
                # The final result of the request will be what is available.
                allow_redirects = True

            # This future starts running the request in a new thread, doesn't block the main thread
            if proxy is not None:
                proxies = {"http": proxy, "https": proxy}
                future = request(
                    url=url_probe,
                    headers=headers,
                    proxies=proxies,
                    allow_redirects=allow_redirects,
                    timeout=request_timeout,
                    json=request_payload,
                )
            else:
                future = request(
                    url=url_probe,
                    headers=headers,
                    allow_redirects=allow_redirects,
                    timeout=request_timeout,
                    json=request_payload,
                )

            # Store future in data for access later
            net_info["request_future"] = future

        # Add this site's results into final dictionary with all the other results.
        results_total[social_network] = results_site

    # Open the file containing account links
    for social_network, net_info in site_data.items():
        # Retrieve results again
        results_site = results_total.get(social_network)

        # Retrieve other site information again
        url = results_site.get("url_user")
        status = results_site.get("status")
        if status is not None:
            # We have already determined the user doesn't exist here
            continue

        # Get the expected error type
        error_type = net_info["errorType"]
        if isinstance(error_type, str):
            error_type: list[str] = [error_type]

        # Retrieve future and ensure it has finished
        future = net_info["request_future"]

        # Determine how long we may still wait for this site. Sites using
        # the "wait" policy are exempted from the batch budget and may run
        # until their own request timeout concludes.
        site_policy = resolve_budget_policy(
            net_info.get("budgetPolicy"), budget_policy
        )
        if batch_deadline is not None and site_policy != "wait":
            result_timeout = max(batch_deadline - monotonic(), 0)
        else:
            result_timeout = None

        r, error_text, exception_text, budget_timed_out = get_response(
            request_future=future,
            error_type=error_type,
            social_network=social_network,
            result_timeout=result_timeout,
        )

        if budget_timed_out:
            # The batch budget is exhausted and this site never returned.
            # Abandon the in-flight request, keep everything already learned
            # about other sites, and explicitly mark this one as unfinished:
            # Timed Out is a distinct status, never confused with Available.
            future.cancel()
            budget_abandoned_any = True
            result = QueryResult(
                username=username,
                site_name=social_network,
                site_url_user=url,
                status=QueryStatus.TIMEOUT,
                query_time=None,
                context=f"Batch time budget of {budget:g}s exhausted before response",
                timeout_connect=results_site["timeout_connect"],
                timeout_read=results_site["timeout_read"],
                timeout_level=results_site["timeout_level"],
            )
            query_notify.update(result)
            results_site["status"] = result
            results_site["http_status"] = ""
            results_site["response_text"] = ""
            results_total[social_network] = results_site
            continue

        # Get response time for response of our request.
        try:
            response_time = r.elapsed
        except AttributeError:
            response_time = None

        # Attempt to get request information
        try:
            http_status = r.status_code
        except Exception:
            http_status = "?"
        try:
            response_text = r.text.encode(r.encoding or "UTF-8")
        except Exception:
            response_text = ""

        query_status = QueryStatus.UNKNOWN
        error_context = None

        # As WAFs advance and evolve, they will occasionally block Sherlock and
        # lead to false positives and negatives. Fingerprints should be added
        # here to filter results that fail to bypass WAFs. Fingerprints should
        # be highly targetted. Comment at the end of each fingerprint to
        # indicate target and date fingerprinted.
        WAFHitMsgs = [
            r'.loading-spinner{visibility:hidden}body.no-js .challenge-running{display:none}body.dark{background-color:#222;color:#d9d9d9}body.dark a{color:#fff}body.dark a:hover{color:#ee730a;text-decoration:underline}body.dark .lds-ring div{border-color:#999 transparent transparent}body.dark .font-red{color:#b20f03}body.dark', # 2024-05-13 Cloudflare
            r'<span id="challenge-error-text">', # 2024-11-11 Cloudflare error page
            r'AwsWafIntegration.forceRefreshToken', # 2024-11-11 Cloudfront (AWS)
            r'{return l.onPageView}}),Object.defineProperty(r,"perimeterxIdentifiers",{enumerable:' # 2024-04-09 PerimeterX / Human Security
        ]

        if error_text is not None:
            error_context = error_text

        elif any(hitMsg in r.text for hitMsg in WAFHitMsgs):
            query_status = QueryStatus.WAF

        else:
            if any(errtype not in ["message", "status_code", "response_url"] for errtype in error_type):
                error_context = f"Unknown error type '{error_type}' for {social_network}"
                query_status = QueryStatus.UNKNOWN
            else:
                if "message" in error_type:
                    # error_flag True denotes no error found in the HTML
                    # error_flag False denotes error found in the HTML
                    error_flag = True
                    errors = net_info.get("errorMsg")
                    # errors will hold the error message
                    # it can be string or list
                    # by isinstance method we can detect that
                    # and handle the case for strings as normal procedure
                    # and if its list we can iterate the errors
                    if isinstance(errors, str):
                        # Checks if the error message is in the HTML
                        # if error is present we will set flag to False
                        if errors in r.text:
                            error_flag = False
                    else:
                        # If it's list, it will iterate all the error message
                        for error in errors:
                            if error in r.text:
                                error_flag = False
                                break
                    if error_flag:
                        query_status = QueryStatus.CLAIMED
                    else:
                        query_status = QueryStatus.AVAILABLE

                if "status_code" in error_type and query_status is not QueryStatus.AVAILABLE:
                    error_codes = net_info.get("errorCode")
                    query_status = QueryStatus.CLAIMED

                    # Type consistency, allowing for both singlets and lists in manifest
                    if isinstance(error_codes, int):
                        error_codes = [error_codes]

                    if error_codes is not None and r.status_code in error_codes:
                        query_status = QueryStatus.AVAILABLE
                    elif r.status_code >= 300 or r.status_code < 200:
                        query_status = QueryStatus.AVAILABLE

                if "response_url" in error_type and query_status is not QueryStatus.AVAILABLE:
                    # For this detection method, we have turned off the redirect.
                    # So, there is no need to check the response URL: it will always
                    # match the request.  Instead, we will ensure that the response
                    # code indicates that the request was successful (i.e. no 404, or
                    # forward to some odd redirect).
                    if 200 <= r.status_code < 300:
                        query_status = QueryStatus.CLAIMED
                    else:
                        query_status = QueryStatus.AVAILABLE

        if dump_response:
            print("+++++++++++++++++++++")
            print(f"TARGET NAME   : {social_network}")
            print(f"USERNAME      : {username}")
            print(f"TARGET URL    : {url}")
            print(f"TEST METHOD   : {error_type}")
            try:
                print(f"STATUS CODES  : {net_info['errorCode']}")
            except KeyError:
                pass
            print("Results...")
            try:
                print(f"RESPONSE CODE : {r.status_code}")
            except Exception:
                pass
            try:
                print(f"ERROR TEXT    : {net_info['errorMsg']}")
            except KeyError:
                pass
            print(">>>>> BEGIN RESPONSE TEXT")
            try:
                print(r.text)
            except Exception:
                pass
            print("<<<<< END RESPONSE TEXT")
            print("VERDICT       : " + str(query_status))
            print("+++++++++++++++++++++")

        # Notify caller about results of query.
        result: QueryResult = QueryResult(
            username=username,
            site_name=social_network,
            site_url_user=url,
            status=query_status,
            query_time=response_time,
            context=error_context,
            timeout_connect=results_site["timeout_connect"],
            timeout_read=results_site["timeout_read"],
            timeout_level=results_site["timeout_level"],
        )
        query_notify.update(result)

        # Save status of request
        results_site["status"] = result

        # Save results from request
        results_site["http_status"] = http_status
        results_site["response_text"] = response_text

        # Add this site's results into final dictionary with all of the other results.
        results_total[social_network] = results_site

    if budget_abandoned_any:
        # Drop requests still queued and do not block on the ones already in
        # flight: results already collected are final, the batch must not pay
        # the slow sites' tail a second time while shutting down.
        session.executor.shutdown(wait=False, cancel_futures=True)

    return results_total


def timeout_check(value):
    """Check Timeout Argument.

    Checks timeout for validity.

    Keyword Arguments:
    value                  -- Time in seconds to wait before timing out request.

    Return Value:
    Floating point number representing the time (in seconds) that should be
    used for the timeout.

    NOTE:  Will raise an exception if the timeout in invalid.
    """

    float_value = float(value)

    if float_value <= 0:
        raise ArgumentTypeError(
            f"Invalid timeout value: {value}. Timeout must be a positive number."
        )

    return float_value


def handler(signal_received, frame):
    """Exit gracefully without throwing errors

    Source: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
    """
    sys.exit(0)


def main():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=f"{__longname__} (Version {__version__})",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__shortname__} v{__version__}",
        help="Display version information and dependencies.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        "-d",
        "--debug",
        action="store_true",
        dest="verbose",
        default=False,
        help="Display extra debugging information and metrics.",
    )
    parser.add_argument(
        "--folderoutput",
        "-fo",
        dest="folderoutput",
        help="If using multiple usernames, the output of the results will be saved to this folder.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        help="If using single username, the output of the result will be saved to this file.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        dest="csv",
        default=False,
        help="Create Comma-Separated Values (CSV) File.",
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        dest="xlsx",
        default=False,
        help="Create the standard file for the modern Microsoft Excel spreadsheet (xlsx).",
    )
    parser.add_argument(
        "--site",
        action="append",
        metavar="SITE_NAME",
        dest="site_list",
        default=[],
        help="Limit analysis to just the listed sites. Add multiple options to specify more than one site.",
    )
    parser.add_argument(
        "--proxy",
        "-p",
        metavar="PROXY_URL",
        action="store",
        dest="proxy",
        default=None,
        help="Make requests over a proxy. e.g. socks5://127.0.0.1:1080",
    )
    parser.add_argument(
        "--dump-response",
        action="store_true",
        dest="dump_response",
        default=False,
        help="Dump the HTTP response to stdout for targeted debugging.",
    )
    parser.add_argument(
        "--json",
        "-j",
        metavar="JSON_FILE",
        dest="json_file",
        default=None,
        help="Load data from a JSON file or an online, valid, JSON file. Upstream PR numbers also accepted.",
    )
    parser.add_argument(
        "--timeout",
        action="store",
        metavar="TIMEOUT",
        dest="timeout",
        type=timeout_check,
        default=60,
        help=(
            "Default time (in seconds) to wait for responses to requests "
            "(Default: 60). Used as the global default for both the connect "
            "and read phases; individual sites may override either phase in "
            "the JSON manifest with a \"timeout\" declaration."
        ),
    )
    parser.add_argument(
        "--budget",
        action="store",
        metavar="SECONDS",
        dest="budget",
        type=timeout_check,
        default=None,
        help=(
            "Total time budget (in seconds) for one username across all "
            "sites. When the budget is exhausted, sites still pending are "
            "handled per --budget-policy and reported as 'Timed Out' while "
            "results already obtained are kept. (Default: no budget)"
        ),
    )
    parser.add_argument(
        "--budget-policy",
        action="store",
        metavar="POLICY",
        dest="budget_policy",
        choices=BUDGET_POLICIES,
        default="abandon",
        help=(
            "How to treat sites that are still pending when --budget is "
            "exhausted: 'abandon' gives up on them and marks them 'Timed "
            "Out' (default); 'wait' lets every request finish up to its own "
            "timeout. Individual sites may override this in the manifest "
            "with \"budgetPolicy\"."
        ),
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        dest="print_all",
        default=False,
        help="Output sites where the username was not found.",
    )
    parser.add_argument(
        "--print-found",
        action="store_true",
        dest="print_found",
        default=True,
        help="Output sites where the username was found (also if exported as file).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        default=False,
        help="Don't color terminal output",
    )
    parser.add_argument(
        "username",
        nargs="+",
        metavar="USERNAMES",
        action="store",
        help="One or more usernames to check with social networks. Check similar usernames using {?} (replace to '_', '-', '.').",
    )
    parser.add_argument(
        "--browse",
        "-b",
        action="store_true",
        dest="browse",
        default=False,
        help="Browse to all results on default browser.",
    )

    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        default=False,
        help="Force the use of the local data.json file.",
    )

    parser.add_argument(
        "--nsfw",
        action="store_true",
        default=False,
        help="Include checking of NSFW sites from default list.",
    )

    parser.add_argument(
        "--txt",
        action="store_true",
        dest="output_txt",
        default=False,
        help="Enable creation of a txt file",
    )

    parser.add_argument(
        "--ignore-exclusions",
        action="store_true",
        dest="ignore_exclusions",
        default=False,
        help="Ignore upstream exclusions (may return more false positives)",
    )

    args = parser.parse_args()

    # If the user presses CTRL-C, exit gracefully without throwing errors
    signal.signal(signal.SIGINT, handler)

    # Check for newer version of Sherlock. If it exists, let the user know about it
    try:
        latest_release_raw = requests.get(forge_api_latest_release, timeout=10).text
        latest_release_json = json_loads(latest_release_raw)
        latest_remote_tag = latest_release_json["tag_name"]

        if latest_remote_tag[1:] != __version__:
            print(
                f"Update available! {__version__} --> {latest_remote_tag[1:]}"
                f"\n{latest_release_json['html_url']}"
            )

    except Exception as error:
        print(f"A problem occurred while checking for an update: {error}")

    # Make prompts
    if args.proxy is not None:
        print("Using the proxy: " + args.proxy)

    if args.no_color:
        # Disable color output.
        init(strip=True, convert=False)
    else:
        # Enable color output.
        init(autoreset=True)

    # Check if both output methods are entered as input.
    if args.output is not None and args.folderoutput is not None:
        print("You can only use one of the output methods.")
        sys.exit(1)

    # Check validity for single username output.
    if args.output is not None and len(args.username) != 1:
        print("You can only use --output with a single username")
        sys.exit(1)

    # Create object with all information about sites we are aware of.
    try:
        if args.local:
            sites = SitesInformation(
                os.path.join(os.path.dirname(__file__), "resources/data.json"),
                honor_exclusions=False,
            )
        else:
            json_file_location = args.json_file
            if args.json_file:
                # If --json parameter is a number, interpret it as a pull request number
                if args.json_file.isnumeric():
                    pull_number = args.json_file
                    pull_url = f"https://api.github.com/repos/sherlock-project/sherlock/pulls/{pull_number}"
                    pull_request_raw = requests.get(pull_url, timeout=10).text
                    pull_request_json = json_loads(pull_request_raw)

                    # Check if it's a valid pull request
                    if "message" in pull_request_json:
                        print(f"ERROR: Pull request #{pull_number} not found.")
                        sys.exit(1)

                    head_commit_sha = pull_request_json["head"]["sha"]
                    json_file_location = f"https://raw.githubusercontent.com/sherlock-project/sherlock/{head_commit_sha}/sherlock_project/resources/data.json"

            sites = SitesInformation(
                data_file_path=json_file_location,
                honor_exclusions=not args.ignore_exclusions,
                do_not_exclude=args.site_list,
            )
    except Exception as error:
        print(f"ERROR:  {error}")
        sys.exit(1)

    if not args.nsfw:
        sites.remove_nsfw_sites(do_not_remove=args.site_list)

    # Create original dictionary from SitesInformation() object.
    # Eventually, the rest of the code will be updated to use the new object
    # directly, but this will glue the two pieces together.
    site_data_all = {site.name: site.information for site in sites}
    if args.site_list == []:
        # Not desired to look at a sub-set of sites
        site_data = site_data_all
    else:
        # User desires to selectively run queries on a sub-set of the site list.
        # Make sure that the sites are supported & build up pruned site database.
        site_data = {}
        site_missing = []
        for site in args.site_list:
            counter = 0
            for existing_site in site_data_all:
                if site.lower() == existing_site.lower():
                    site_data[existing_site] = site_data_all[existing_site]
                    counter += 1
            if counter == 0:
                # Build up list of sites not supported for future error message.
                site_missing.append(f"'{site}'")

        if site_missing:
            print(f"Error: Desired sites not found: {', '.join(site_missing)}.")

        if not site_data:
            sys.exit(1)

    # Tiered timeouts are in effect when a batch budget was given or any
    # selected site declares its own timeout. Drives extra result/export
    # annotation; with neither set, output is identical to old behaviour.
    timeouts_tiered = timeouts_are_tiered(site_data, args.budget)

    # Create notify object for query results.
    query_notify = QueryNotifyPrint(
        result=None,
        verbose=args.verbose,
        print_all=args.print_all,
        browse=args.browse,
        show_timeout_tiers=timeouts_tiered,
    )

    # Run report on all specified users.
    all_usernames = []
    for username in args.username:
        if check_for_parameter(username):
            for name in multiple_usernames(username):
                all_usernames.append(name)
        else:
            all_usernames.append(username)
    for username in all_usernames:
        results = sherlock(
            username,
            site_data,
            query_notify,
            dump_response=args.dump_response,
            proxy=args.proxy,
            timeout=args.timeout,
            budget=args.budget,
            budget_policy=args.budget_policy,
        )

        if args.output:
            result_file = args.output
        elif args.folderoutput:
            # The usernames results should be stored in a targeted folder.
            # If the folder doesn't exist, create it first
            os.makedirs(args.folderoutput, exist_ok=True)
            result_file = os.path.join(args.folderoutput, f"{username}.txt")
        else:
            result_file = f"{username}.txt"

        if args.output_txt:
            with open(result_file, "w", encoding="utf-8") as file:
                exists_counter = 0
                for website_name in results:
                    dictionary = results[website_name]
                    if dictionary.get("status").status == QueryStatus.CLAIMED:
                        exists_counter += 1
                        file.write(dictionary["url_user"] + "\n")
                file.write(f"Total Websites Username Detected On : {exists_counter}\n")

        if args.csv:
            result_file = f"{username}.csv"
            if args.folderoutput:
                # The usernames results should be stored in a targeted folder.
                # If the folder doesn't exist, create it first
                os.makedirs(args.folderoutput, exist_ok=True)
                result_file = os.path.join(args.folderoutput, result_file)

            with open(result_file, "w", newline="", encoding="utf-8") as csv_report:
                writer = csv.writer(csv_report)
                header_row = [
                    "username",
                    "name",
                    "url_main",
                    "url_user",
                    "exists",
                    "http_status",
                    "response_time_s",
                ]
                if timeouts_tiered:
                    # Expose the effective connect/read timeout and the tier
                    # (global vs site) that was actually used per site.
                    header_row += [
                        "timeout_connect_s",
                        "timeout_read_s",
                        "timeout_level",
                    ]
                writer.writerow(header_row)
                for site in results:
                    site_status = results[site]["status"].status
                    if (
                        args.print_found
                        and not args.print_all
                        and site_status not in (
                            QueryStatus.CLAIMED,
                            QueryStatus.TIMEOUT,
                        )
                    ):
                        continue

                    response_time_s = results[site]["status"].query_time
                    if response_time_s is None:
                        response_time_s = ""
                    row = [
                        username,
                        site,
                        results[site]["url_main"],
                        results[site]["url_user"],
                        str(site_status),
                        results[site]["http_status"],
                        response_time_s,
                    ]
                    if timeouts_tiered:
                        row += [
                            results[site]["timeout_connect"],
                            results[site]["timeout_read"],
                            results[site]["timeout_level"],
                        ]
                    writer.writerow(row)
        if args.xlsx:
            usernames = []
            names = []
            url_main = []
            url_user = []
            exists = []
            http_status = []
            response_time_s = []
            timeout_connect_s = []
            timeout_read_s = []
            timeout_level = []

            for site in results:
                site_status = results[site]["status"].status
                if (
                    args.print_found
                    and not args.print_all
                    and site_status not in (
                        QueryStatus.CLAIMED,
                        QueryStatus.TIMEOUT,
                    )
                ):
                    continue

                if response_time_s is None:
                    response_time_s.append("")
                else:
                    response_time_s.append(results[site]["status"].query_time)
                usernames.append(username)
                names.append(site)
                url_main.append(results[site]["url_main"])
                url_user.append(results[site]["url_user"])
                exists.append(str(site_status))
                http_status.append(results[site]["http_status"])
                if timeouts_tiered:
                    timeout_connect_s.append(results[site]["timeout_connect"])
                    timeout_read_s.append(results[site]["timeout_read"])
                    timeout_level.append(results[site]["timeout_level"])

            xlsx_data = {
                "username": usernames,
                "name": names,
                "url_main": [f'=HYPERLINK(\"{u}\")' for u in url_main],
                "url_user": [f'=HYPERLINK(\"{u}\")' for u in url_user],
                "exists": exists,
                "http_status": http_status,
                "response_time_s": response_time_s,
            }
            if timeouts_tiered:
                # Effective connect/read timeout and the tier actually used.
                xlsx_data["timeout_connect_s"] = timeout_connect_s
                xlsx_data["timeout_read_s"] = timeout_read_s
                xlsx_data["timeout_level"] = timeout_level

            DataFrame = pd.DataFrame(xlsx_data)
            DataFrame.to_excel(f"{username}.xlsx", sheet_name="sheet1", index=False)

        print()
    query_notify.finish()


if __name__ == "__main__":
    main()
