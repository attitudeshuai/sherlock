"""Sherlock Result Module

This module defines various objects for recording the results of queries.
"""
from enum import Enum


class QueryStatus(Enum):
    """Query Status Enumeration.

    Describes status of query about a given username.
    """
    CLAIMED   = "Claimed"   # Username Detected
    AVAILABLE = "Available" # Username Not Detected
    UNKNOWN   = "Unknown"   # Error Occurred While Trying To Detect Username
    ILLEGAL   = "Illegal"   # Username Not Allowable For This Site
    WAF       = "WAF"       # Request blocked by WAF (i.e. Cloudflare)

    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string to get information about this object.
        """
        return self.value


class RedirectHop:
    """Redirect Hop Object.

    Describes a single hop in a query's redirect chain: the status code
    returned by the server and the address that status pointed at.

    Redirect chains are diagnostic data only.  They never influence the
    verdict (QueryStatus) of a site.
    """
    def __init__(self, status_code, location):
        """Create Redirect Hop Object.

        Keyword Arguments:
        self                   -- This object.
        status_code            -- HTTP status code returned for this hop.
        location               -- Address the response redirected to
                                  (absolute URL).  For a followed hop this is
                                  the URL of the next observed response; for a
                                  refused or truncated final hop this is the
                                  unresolved target taken from the Location
                                  response header, which may be None.

        Return Value:
        Nothing.
        """
        self.status_code = status_code
        self.location    = location

    def to_dict(self):
        """Convert Object To Dictionary.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Dictionary with the status code and target address of this hop.
        """
        return {"status_code": self.status_code, "location": self.location}

    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string describing this hop.
        """
        return f"{self.status_code} -> {self.location}"


class QueryResult():
    """Query Result Object.

    Describes result of query about a given username.
    """
    def __init__(self, username, site_name, site_url_user, status,
                 query_time=None, context=None,
                 redirect_chain=None, final_url=None, redirect_truncated=False):
        """Create Query Result Object.

        Contains information about a specific method of detecting usernames on
        a given type of web sites.

        Keyword Arguments:
        self                   -- This object.
        username               -- String indicating username that query result
                                  was about.
        site_name              -- String which identifies site.
        site_url_user          -- String containing URL for username on site.
                                  NOTE:  The site may or may not exist:  this
                                         just indicates what the name would
                                         be, if it existed.
        status                 -- Enumeration of type QueryStatus() indicating
                                  the status of the query.
        query_time             -- Time (in seconds) required to perform query.
                                  Default of None.
        context                -- String indicating any additional context
                                  about the query.  For example, if there was
                                  an error, this might indicate the type of
                                  error that occurred.
                                  Default of None.
        redirect_chain         -- List of RedirectHop() objects describing
                                  every redirect observed for the query, in
                                  order.  Empty when no redirect occurred.
                                  Default of None (stored as empty list).
        final_url              -- String with the final landing URL after
                                  following redirects.  None when redirects
                                  were not followed or no landing was reached.
        redirect_truncated     -- Boolean indicating that the redirect chain
                                  was cut off at the hop limit (e.g. a
                                  redirect loop).  The recorded chain then
                                  ends with the hop that would have continued.

        Return Value:
        Nothing.
        """

        self.username           = username
        self.site_name          = site_name
        self.site_url_user      = site_url_user
        self.status             = status
        self.query_time         = query_time
        self.context            = context
        self.redirect_chain     = redirect_chain if redirect_chain is not None else []
        self.final_url          = final_url
        self.redirect_truncated = redirect_truncated


    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string to get information about this object.
        """
        status = str(self.status)
        if self.context is not None:
            # There is extra context information available about the results.
            # Append it to the normal response text.
            status += f" ({self.context})"

        return status
