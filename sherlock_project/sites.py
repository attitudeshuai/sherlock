"""Sherlock Sites Information Module

This module supports storing information about websites.
This is the raw data that will be used to search for usernames.
"""
import hashlib
import json
from datetime import datetime, timezone

import requests
import secrets


MANIFEST_URL = "https://data.sherlockproject.xyz"
EXCLUSIONS_URL = "https://raw.githubusercontent.com/sherlock-project/sherlock/refs/heads/exclusions/false_positive_exclusions.txt"

# Recorded as the reason a site was dropped by the age-rating filter.
AGE_RATING_SOURCE = "manifest isNSFW flag"

# Fallback behavior whenever the upstream exclusion list cannot be applied.
EXCLUSIONS_FALLBACK = (
    "continued without the upstream exclusion list; every manifest site was retained"
)


class SelectionLedger:
    """Audit trail of how the raw manifest was narrowed into the loaded site set.

    The ledger is populated by ``SitesInformation`` as exclusions and age-rating
    filtering are applied. It only records what happened; it never influences
    which sites are kept or dropped.

    Recorded facts:
      * manifest entry count and final loaded count
      * whether the upstream exclusion list was fetched, the exact version
        applied (fetch time, HTTP caching headers, SHA-256 of the contents)
      * which sites were excluded and the source list that named them
      * which exclusion-listed sites were kept because the user named them
      * which exclusion entries did not match any manifest site
      * the degraded fallback behavior when the fetch fails
      * how many sites were removed (and later retained) by the NSFW filter
    """

    def __init__(self, exclusion_source_url: str):
        self.manifest_entries: int = 0
        self.sites_loaded: int = 0
        self.exclusions: dict = {
            "source_url": exclusion_source_url,
            # Whether a fetch was attempted at all (False for --local / --ignore-exclusions).
            "requested": False,
            # True only when the list was downloaded (HTTP 200) and applied.
            "fetched": False,
            "status_code": None,
            "fetched_at": None,
            "etag": None,
            "last_modified": None,
            "content_sha256": None,
            "entry_count": None,
            "blank_entries": 0,
            "error": None,
            "fallback": None,
            # Each item: {"name": <site name>, "source": <list URL>}
            "excluded_sites": [],
            # Sites named by the user (e.g. --site) that would otherwise have been excluded.
            "kept_by_name": [],
            # Names found in the exclusion list but absent from the manifest.
            "listed_not_in_manifest": [],
        }
        self.age_rating: dict = {
            # Set to True once remove_nsfw_sites() has run.
            "filter_applied": False,
            # Each item: {"name": <site name>, "source": AGE_RATING_SOURCE}
            "filtered_sites": [],
            # NSFW sites kept because the user named them in the filtering call.
            "kept_by_name": [],
        }

    @property
    def exclusion_state(self) -> str:
        """Short machine-readable state: 'applied', 'unavailable' or 'disabled'."""
        if not self.exclusions["requested"]:
            return "disabled"
        return "applied" if self.exclusions["fetched"] else "unavailable"

    @property
    def excluded_count(self) -> int:
        return len(self.exclusions["excluded_sites"])

    @property
    def age_filtered_count(self) -> int:
        return len(self.age_rating["filtered_sites"])

    def to_dict(self) -> dict:
        """Plain-data copy of the ledger, suitable for JSON serialization."""
        return {
            "manifest_entries": self.manifest_entries,
            "sites_loaded": self.sites_loaded,
            "exclusions": {
                **self.exclusions,
                "state": self.exclusion_state,
                "excluded_sites": [dict(entry) for entry in self.exclusions["excluded_sites"]],
                "kept_by_name": list(self.exclusions["kept_by_name"]),
                "listed_not_in_manifest": list(self.exclusions["listed_not_in_manifest"]),
            },
            "age_rating_filter": {
                "applied": self.age_rating["filter_applied"],
                "filtered_count": self.age_filtered_count,
                "filtered_sites": [dict(entry) for entry in self.age_rating["filtered_sites"]],
                "kept_by_name": list(self.age_rating["kept_by_name"]),
            },
        }

    def render(self) -> str:
        """Human-readable rendering of the full ledger."""
        ex = self.exclusions
        lines = [
            "Site selection ledger",
            f"  Manifest entries : {self.manifest_entries}",
            f"  Sites loaded     : {self.sites_loaded}",
            "",
            "Upstream exclusion list",
            f"  Source URL       : {ex['source_url']}",
            f"  State            : {self.exclusion_state}",
        ]

        if self.exclusion_state == "disabled":
            lines.append("  Fetch            : not attempted (exclusions disabled)")
        elif self.exclusion_state == "unavailable":
            lines.append(f"  HTTP status      : {ex['status_code']}")
            lines.append(f"  Error            : {ex['error']}")
            lines.append(f"  Fallback         : {ex['fallback']}")
        else:
            lines.extend([
                f"  HTTP status      : {ex['status_code']}",
                f"  Fetched at (UTC) : {ex['fetched_at']}",
                f"  ETag             : {ex['etag']}",
                f"  Last-Modified    : {ex['last_modified']}",
                f"  SHA-256          : {ex['content_sha256']}",
                f"  List entries     : {ex['entry_count']} (blank lines: {ex['blank_entries']})",
                f"  Excluded sites   : {self.excluded_count}",
            ])
            lines.extend(
                f"    - {entry['name']} (source: {entry['source']})"
                for entry in ex["excluded_sites"]
            )
            if ex["listed_not_in_manifest"]:
                lines.append(
                    f"  Listed, absent  : {', '.join(ex['listed_not_in_manifest'])}"
                )

        if ex["kept_by_name"]:
            lines.append(
                f"  Kept by name     : {', '.join(ex['kept_by_name'])}"
            )

        lines.append("")
        age = self.age_rating
        lines.append("Age-rating (NSFW) filter")
        lines.append(f"  Applied          : {age['filter_applied']}")
        if age["filter_applied"]:
            lines.append(f"  Filtered sites   : {self.age_filtered_count}")
            lines.extend(
                f"    - {entry['name']} (source: {entry['source']})"
                for entry in age["filtered_sites"]
            )
            if age["kept_by_name"]:
                lines.append(
                    f"  Kept by name     : {', '.join(age['kept_by_name'])}"
                )

        return "\n".join(lines)

    def coverage_lines(self, scanned_count: int) -> list[str]:
        """Short summary lines for the end-of-scan report.

        Shows how many of the manifest entries were actually scanned and how
        the gap breaks down (upstream exclusions, age-rating filter, targeted
        --site selection), plus the state of the upstream exclusion list.
        """
        excluded = self.excluded_count
        age_filtered = self.age_filtered_count
        unselected = max(
            self.manifest_entries - scanned_count - excluded - age_filtered, 0
        )
        not_covered = self.manifest_entries - scanned_count

        lines = [
            f"Coverage: scanned {scanned_count} of {self.manifest_entries} "
            f"manifest sites ({not_covered} not covered)."
        ]

        breakdown = []
        if excluded:
            breakdown.append(f"{excluded} removed by upstream exclusion list")
        if age_filtered:
            breakdown.append(f"{age_filtered} removed by age-rating (NSFW) filter")
        if unselected:
            breakdown.append(f"{unselected} not selected for this run")
        if breakdown:
            lines.append("  Not covered: " + "; ".join(breakdown) + ".")

        ex = self.exclusions
        if self.exclusion_state == "disabled":
            lines.append("  Upstream exclusion list: disabled; no fetch was attempted.")
        elif self.exclusion_state == "applied":
            version = ex["content_sha256"][:12] if ex["content_sha256"] else "unknown"
            lines.append(
                f"  Upstream exclusion list: applied (fetched {ex['fetched_at']}, "
                f"version sha256:{version}, {ex['entry_count']} entries)."
            )
        else:
            lines.append(
                f"  Upstream exclusion list: NOT applied; fetch failed "
                f"({ex['error']}). Fallback: {ex['fallback']}."
            )

        return lines

class SiteInformation:
    def __init__(self, name, url_home, url_username_format, username_claimed,
                information, is_nsfw, username_unclaimed=secrets.token_urlsafe(10)):
        """Create Site Information Object.

        Contains information about a specific website.

        Keyword Arguments:
        self                   -- This object.
        name                   -- String which identifies site.
        url_home               -- String containing URL for home of site.
        url_username_format    -- String containing URL for Username format
                                  on site.
                                  NOTE:  The string should contain the
                                         token "{}" where the username should
                                         be substituted.  For example, a string
                                         of "https://somesite.com/users/{}"
                                         indicates that the individual
                                         usernames would show up under the
                                         "https://somesite.com/users/" area of
                                         the website.
        username_claimed       -- String containing username which is known
                                  to be claimed on website.
        username_unclaimed     -- String containing username which is known
                                  to be unclaimed on website.
        information            -- Dictionary containing all known information
                                  about website.
                                  NOTE:  Custom information about how to
                                         actually detect the existence of the
                                         username will be included in this
                                         dictionary.  This information will
                                         be needed by the detection method,
                                         but it is only recorded in this
                                         object for future use.
        is_nsfw                -- Boolean indicating if site is Not Safe For Work.

        Return Value:
        Nothing.
        """

        self.name = name
        self.url_home = url_home
        self.url_username_format = url_username_format

        self.username_claimed = username_claimed
        self.username_unclaimed = secrets.token_urlsafe(32)
        self.information = information
        self.is_nsfw  = is_nsfw


    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string to get information about this object.
        """

        return f"{self.name} ({self.url_home})"


class SitesInformation:
    def __init__(
            self,
            data_file_path: str|None = None,
            honor_exclusions: bool = True,
            do_not_exclude: list[str] = [],
        ):
        """Create Sites Information Object.

        Contains information about all supported websites.

        Keyword Arguments:
        self                   -- This object.
        data_file_path         -- String which indicates path to data file.
                                  The file name must end in ".json".

                                  There are 3 possible formats:
                                   * Absolute File Format
                                     For example, "c:/stuff/data.json".
                                   * Relative File Format
                                     The current working directory is used
                                     as the context.
                                     For example, "data.json".
                                   * URL Format
                                     For example,
                                     "https://example.com/data.json", or
                                     "http://example.com/data.json".

                                  An exception will be thrown if the path
                                  to the data file is not in the expected
                                  format, or if there was any problem loading
                                  the file.

                                  If this option is not specified, then a
                                  default site list will be used.

        Return Value:
        Nothing.
        """

        if not data_file_path:
            # The default data file is the live data.json which is in the GitHub repo. The reason why we are using
            # this instead of the local one is so that the user has the most up-to-date data. This prevents
            # users from creating issue about false positives which has already been fixed or having outdated data
            data_file_path = MANIFEST_URL

        if data_file_path.lower().startswith("http"):
            # Reference is to a URL.
            try:
                response = requests.get(url=data_file_path, timeout=30)
            except Exception as error:
                raise FileNotFoundError(
                    f"Problem while attempting to access data file URL '{data_file_path}':  {error}"
                )

            if response.status_code != 200:
                raise FileNotFoundError(f"Bad response while accessing "
                                        f"data file URL '{data_file_path}'."
                                        )
            try:
                site_data = response.json()
            except Exception as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}':  {error}."
                )

        else:
            # Reference is to a file.
            try:
                with open(data_file_path, "r", encoding="utf-8") as file:
                    try:
                        site_data = json.load(file)
                    except Exception as error:
                        raise ValueError(
                            f"Problem parsing json contents at '{data_file_path}':  {error}."
                        )

            except FileNotFoundError:
                raise FileNotFoundError(f"Problem while attempting to access "
                                        f"data file '{data_file_path}'."
                                        )

        site_data.pop('$schema', None)

        # Ledger records every exclusion / filtering decision from here on.
        # It is purely observational: none of the logic below depends on it.
        self.ledger = SelectionLedger(EXCLUSIONS_URL)
        self.ledger.manifest_entries = len(site_data)

        if honor_exclusions:
            self.ledger.exclusions["requested"] = True
            try:
                response = requests.get(url=EXCLUSIONS_URL, timeout=10)
            except Exception as error:
                # If there was any problem loading the exclusions, just continue without them
                print("Warning: Could not load exclusions, continuing without them.")
                honor_exclusions = False
                self.ledger.exclusions["fetched"] = False
                self.ledger.exclusions["error"] = f"{type(error).__name__}: {error}"
                self.ledger.exclusions["fallback"] = EXCLUSIONS_FALLBACK
            else:
                if response.status_code == 200:
                    exclusions_text = response.text
                    exclusions = exclusions_text.splitlines()
                    exclusions = [exclusion.strip() for exclusion in exclusions]

                    # Version metrics describe the list as fetched, before
                    # user-named sites are lifted out of it.
                    list_entry_count = len(
                        [exclusion for exclusion in exclusions if exclusion]
                    )
                    list_blank_entries = len(
                        [exclusion for exclusion in exclusions if not exclusion]
                    )

                    # Record which user-named sites are about to be spared,
                    # then perform the same removals as before.
                    kept_by_name = [
                        site for site in dict.fromkeys(do_not_exclude)
                        if site in exclusions
                    ]
                    self.ledger.exclusions["kept_by_name"] = kept_by_name

                    for site in do_not_exclude:
                        if site in exclusions:
                            exclusions.remove(site)

                    removed_names = []
                    absent_names = []
                    for exclusion in exclusions:
                        if not exclusion:
                            # Blank lines never match a site; popping them was a no-op.
                            continue
                        if exclusion in removed_names:
                            # Duplicate line in the list; site was already removed.
                            continue
                        if exclusion in site_data:
                            site_data.pop(exclusion, None)
                            removed_names.append(exclusion)
                        else:
                            absent_names.append(exclusion)

                    self.ledger.exclusions.update({
                        "fetched": True,
                        "status_code": response.status_code,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "content_sha256": hashlib.sha256(
                            exclusions_text.encode("utf-8")
                        ).hexdigest(),
                        "entry_count": list_entry_count,
                        "blank_entries": list_blank_entries,
                        "excluded_sites": [
                            {"name": name, "source": EXCLUSIONS_URL}
                            for name in removed_names
                        ],
                        "listed_not_in_manifest": absent_names,
                    })
                else:
                    # A non-200 response means the list could not be obtained.
                    # Historically this was silently ignored; keep that output
                    # behavior, but record the failure and the fallback taken.
                    self.ledger.exclusions["fetched"] = False
                    self.ledger.exclusions["status_code"] = response.status_code
                    self.ledger.exclusions["error"] = (
                        f"Unexpected HTTP status code {response.status_code}"
                    )
                    self.ledger.exclusions["fallback"] = EXCLUSIONS_FALLBACK

        self.sites = {}

        # Add all site information from the json file to internal site list.
        for site_name in site_data:
            try:

                self.sites[site_name] = \
                    SiteInformation(site_name,
                                    site_data[site_name]["urlMain"],
                                    site_data[site_name]["url"],
                                    site_data[site_name]["username_claimed"],
                                    site_data[site_name],
                                    site_data[site_name].get("isNSFW",False)

                                    )
            except KeyError as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}':  Missing attribute {error}."
                )
            except TypeError:
                print(f"Encountered TypeError parsing json contents for target '{site_name}' at {data_file_path}\nSkipping target.\n")

        self.ledger.sites_loaded = len(self.sites)


    def remove_nsfw_sites(self, do_not_remove: list = []):
        """
        Remove NSFW sites from the sites, if isNSFW flag is true for site

        Keyword Arguments:
        self                   -- This object.
        do_not_remove          -- List of site names explicitly named by the
                                  user; matching NSFW sites are retained.

        Return Value:
        None

        Every removed or name-spared site is recorded in self.ledger.
        """
        sites = {}
        do_not_remove = [site.casefold() for site in do_not_remove]
        filtered_names = []
        kept_by_name = []
        for site in self.sites:
            if self.sites[site].is_nsfw:
                if site.casefold() in do_not_remove:
                    kept_by_name.append(site)
                else:
                    filtered_names.append(site)
                    continue
            sites[site] = self.sites[site]
        self.sites =  sites

        # Record the age-rating decision in the ledger. Sites already removed
        # by a previous call are gone from self.sites, so accumulation here
        # only ever adds fresh names.
        already_recorded = {
            entry["name"] for entry in self.ledger.age_rating["filtered_sites"]
        }
        for name in filtered_names:
            if name not in already_recorded:
                self.ledger.age_rating["filtered_sites"].append(
                    {"name": name, "source": AGE_RATING_SOURCE}
                )
        self.ledger.age_rating["kept_by_name"] = list(dict.fromkeys(kept_by_name))
        self.ledger.age_rating["filter_applied"] = True

    def site_name_list(self):
        """Get Site Name List.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        List of strings containing names of sites.
        """

        return sorted([site.name for site in self], key=str.lower)

    def __iter__(self):
        """Iterator For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Iterator for sites object.
        """

        for site_name in self.sites:
            yield self.sites[site_name]

    def __len__(self):
        """Length For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Length of sites object.
        """
        return len(self.sites)
