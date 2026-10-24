"""Sherlock Checkpoint Module

Persists the completion state and conclusion of every (username, site)
unit of a batch scan to a human-readable JSON state file. When a scan is
interrupted (process killed, network failure, machine reboot), relaunching
with the same batch parameters and ``--resume`` reuses every recorded
conclusion and only re-queries units that never reached a conclusion.

The state file is a regular file on purpose: it can be inspected with any
text editor and deleted by the user. A file that is corrupt, that does not
belong to Sherlock, or that does not match the current batch parameters
aborts the run with an explicit error instead of silently resuming the
wrong data.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

from sherlock_project.result import QueryStatus

#: Version of the on-disk state file format. Bump when the layout changes.
FORMAT_VERSION = 1

#: Marker used to recognise files written by Sherlock.
TOOL_NAME = "sherlock-project"

#: Every persisted unit conclusion must be one of these status values.
_VALID_STATUSES = {status.value for status in QueryStatus}

#: Filename used when ``--resume`` is given without an explicit path.
DEFAULT_CHECKPOINT_FILE = "sherlock_checkpoint.json"

#: Advice appended to every fatal checkpoint error.
_RESET_HINT = (
    "To start a brand new scan from scratch, delete this file (or point "
    "--resume at a different path) and run the same command again."
)


class CheckpointError(Exception):
    """The checkpoint file is missing, corrupt, foreign, or mismatched.

    Raised when resuming would be unsafe. The message always explains how
    to start over deliberately.
    """


def _timestamp() -> str:
    """Current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_site_fingerprint(site_data: dict[str, dict[str, Any]]) -> str:
    """Compute a stable fingerprint of the effective target site set.

    The fingerprint covers the full manifest contents Sherlock actually
    scans (after ``--site``/``--nsfw``/exclusion filtering and after any
    ``--local``/``--json`` manifest selection), so any change in either
    the selected sites or their detection rules invalidates an old
    checkpoint. The runtime-only ``request_future`` key is ignored.

    Keyword Arguments:
    site_data              -- Dictionary of site name -> site information,
                              exactly as passed to ``sherlock()``.

    Return Value:
    ``"sha256:"`` followed by the hex digest.
    """
    fingerprinted = {}
    for name, information in sorted(site_data.items()):
        fingerprinted[name] = {
            key: value
            for key, value in information.items()
            if key != "request_future"
        }
    payload = json.dumps(
        fingerprinted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class Checkpoint:
    """Batch scan progress backed by a JSON state file.

    All mutation methods rewrite the file atomically (temp file +
    ``os.replace``), so a kill or power loss at any instant leaves either
    the previous complete file or the next complete file -- never a torn
    one.
    """

    def __init__(
        self,
        path: str,
        data: dict[str, Any],
    ):
        """Create a checkpoint around an already validated data dict."""
        self.path = path
        self.data = data

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        path: str,
        *,
        usernames: list[str],
        site_data: dict[str, dict[str, Any]],
        sherlock_version: str,
        proxy: Optional[str] = None,
        timeout: float = 60,
    ) -> tuple["Checkpoint", bool]:
        """Load a compatible checkpoint, or create a fresh one.

        Keyword Arguments:
        path                   -- State file path.
        usernames              -- Expanded batch usernames, in order.
        site_data              -- Effective site data for this batch.
        sherlock_version       -- Sherlock version string (informational).
        proxy                  -- Proxy URL (informational).
        timeout                -- Request timeout (informational).

        Return Value:
        Tuple of (checkpoint, resumed) where ``resumed`` is True when an
        existing, compatible file with prior progress was loaded.

        Raises CheckpointError on corruption or batch mismatch.
        """
        fingerprint = compute_site_fingerprint(site_data)

        if os.path.exists(path):
            raw = cls._read_raw(path)
            cls._validate(path, raw, usernames, fingerprint)
            checkpoint = cls(path, raw)
            resumed = checkpoint.completed_unit_count() > 0
        else:
            checkpoint = cls(path, cls._fresh_data(
                usernames=usernames,
                fingerprint=fingerprint,
                site_count=len(site_data),
                sherlock_version=sherlock_version,
                proxy=proxy,
                timeout=timeout,
            ))
            checkpoint._flush()
            resumed = False

        return checkpoint, resumed

    @staticmethod
    def _read_raw(path: str) -> Any:
        """Read and JSON-decode the state file, rejecting bad files."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except OSError as error:
            raise CheckpointError(
                f"Could not read checkpoint file '{path}': {error}. {_RESET_HINT}"
            )
        except json.JSONDecodeError as error:
            raise CheckpointError(
                f"Checkpoint file '{path}' is corrupt (invalid JSON at "
                f"line {error.lineno}, column {error.colno}). {_RESET_HINT}"
            )
        return raw

    @staticmethod
    def _validate(
        path: str,
        raw: Any,
        usernames: list[str],
        fingerprint: str,
    ) -> None:
        """Validate structure, provenance and batch identity."""
        def fail(detail: str) -> None:
            raise CheckpointError(
                f"Cannot resume from checkpoint file '{path}': {detail} "
                f"{_RESET_HINT}"
            )

        if not isinstance(raw, dict):
            fail("the file does not contain a Sherlock checkpoint object.")

        if raw.get("tool") != TOOL_NAME:
            fail("this file was not created by Sherlock.")

        version = raw.get("format_version")
        if not isinstance(version, int):
            fail("missing or invalid 'format_version'.")
        if version != FORMAT_VERSION:
            fail(
                f"it uses checkpoint format version {version}, but this "
                f"version of Sherlock only understands version "
                f"{FORMAT_VERSION}."
            )

        batch = raw.get("batch")
        if not isinstance(batch, dict):
            fail("missing or invalid 'batch' section.")

        stored_usernames = batch.get("usernames")
        if not isinstance(stored_usernames, list) or not all(
            isinstance(name, str) for name in stored_usernames
        ):
            fail("missing or invalid batch usernames.")
        if stored_usernames != usernames:
            fail(
                "batch parameters do not match. The checkpoint was created "
                f"for usernames {stored_usernames}, but this run targets "
                f"{usernames}. Re-run with the same usernames (same order) "
                "to resume."
            )

        stored_fingerprint = batch.get("site_fingerprint")
        if not isinstance(stored_fingerprint, str):
            fail("missing or invalid 'site_fingerprint'.")
        if stored_fingerprint != fingerprint:
            fail(
                "the target site set does not match the one used when this "
                "checkpoint was created. It may be a different --site "
                "selection, a different --nsfw/--local/--json/--ignore-"
                "exclusions configuration, or the site manifest has since "
                "changed."
            )

        results = raw.get("results")
        if not isinstance(results, dict):
            fail("missing or invalid 'results' section.")

        # Deep-check every recorded unit so a half-written or hand-edited
        # file can never masquerade as a valid conclusion.
        for username, sites in results.items():
            if not isinstance(username, str) or not isinstance(sites, dict):
                fail("'results' must map usernames to per-site records.")
            for site_name, record in sites.items():
                if not isinstance(site_name, str) or not isinstance(record, dict):
                    fail(f"record for '{username}'/'{site_name}' is invalid.")
                status = record.get("status")
                if status not in _VALID_STATUSES:
                    fail(
                        f"record for '{username}'/'{site_name}' has unknown "
                        f"status {status!r}."
                    )
                if not isinstance(record.get("site_url_user"), str):
                    fail(
                        f"record for '{username}'/'{site_name}' is missing "
                        "'site_url_user'."
                    )
                if not isinstance(record.get("url_user"), str):
                    fail(
                        f"record for '{username}'/'{site_name}' is missing "
                        "'url_user'."
                    )
                http_status = record.get("http_status")
                if not (
                    http_status is None
                    or isinstance(http_status, int)
                    or isinstance(http_status, str)
                ):
                    fail(
                        f"record for '{username}'/'{site_name}' has an "
                        "invalid 'http_status'."
                    )
                query_time = record.get("query_time")
                if query_time is not None and not isinstance(query_time, (int, float)):
                    fail(
                        f"record for '{username}'/'{site_name}' has an "
                        "invalid 'query_time'."
                    )
                context = record.get("context")
                if context is not None and not isinstance(context, str):
                    fail(
                        f"record for '{username}'/'{site_name}' has an "
                        "invalid 'context'."
                    )

    @staticmethod
    def _fresh_data(
        *,
        usernames: list[str],
        fingerprint: str,
        site_count: int,
        sherlock_version: str,
        proxy: Optional[str],
        timeout: float,
    ) -> dict[str, Any]:
        """Build the data structure for a brand new checkpoint file."""
        now = _timestamp()
        return {
            "format_version": FORMAT_VERSION,
            "tool": TOOL_NAME,
            "batch": {
                "usernames": list(usernames),
                "site_fingerprint": fingerprint,
                "site_count": site_count,
                "sherlock_version": sherlock_version,
                "proxy": proxy,
                "timeout": timeout,
                "completed": False,
                "created_at": now,
                "updated_at": now,
            },
            "results": {},
        }

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_record(self, username: str, site_name: str) -> Optional[dict[str, Any]]:
        """Return the recorded conclusion of one unit, or None.

        A record (including UNKNOWN/WAF outcomes) means the unit already
        reached a conclusion and must not be queried again.
        """
        return self.data.get("results", {}).get(username, {}).get(site_name)

    def completed_unit_count(self) -> int:
        """Number of (username, site) units with a recorded conclusion."""
        return sum(
            len(sites) for sites in self.data.get("results", {}).values()
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(
        self,
        username: str,
        site_name: str,
        *,
        status: str,
        site_url_user: str,
        url_user: str,
        http_status: Any,
        query_time: Optional[float],
        context: Optional[str],
    ) -> None:
        """Register the conclusion of one unit and persist immediately.

        Flushing on every record means an interrupted run loses at most
        the units still in flight, never a reached conclusion.
        """
        if status not in _VALID_STATUSES:
            # Programmer error, not a corrupt-file condition.
            raise ValueError(f"Refusing to persist unknown status {status!r}")

        results = self.data.setdefault("results", {})
        user_results = results.setdefault(username, {})
        user_results[site_name] = {
            "status": status,
            "site_url_user": site_url_user,
            "url_user": url_user,
            "http_status": http_status,
            "query_time": query_time,
            "context": context,
            "completed_at": _timestamp(),
        }
        self._flush()

    def mark_batch_complete(self) -> None:
        """Mark the whole batch as finished and persist."""
        self.data["batch"]["completed"] = True
        self._flush()

    def _flush(self) -> None:
        """Atomically rewrite the state file."""
        self.data["batch"]["updated_at"] = _timestamp()

        directory = os.path.dirname(os.path.abspath(self.path))
        try:
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                dir=directory,
                prefix=".sherlock_checkpoint_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(
                        self.data,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.path)
            except BaseException:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
                raise
        except OSError as error:
            raise CheckpointError(
                f"Could not write checkpoint file '{self.path}': {error}. "
                f"{_RESET_HINT}"
            )
