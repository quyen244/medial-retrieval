"""Stop hammering a host that has started refusing us.

Without this, a domain that begins answering 403 keeps receiving the full
crawl: every remaining document for that host is attempted, each with its
retries and backoff. That wastes the run's time and, worse, it is exactly the
behaviour that turns a temporary rate-limit into a lasting block.

This happened. Sustained crawling of nhathuoclongchau.com.vn, baohaiphong.vn,
baonghean.vn, baodanang.vn and hanoimoi.vn took them from working to refusing
a single slow request, and the crawler spent roughly a thousand further
requests confirming it.

The rule is deliberately blunt: after enough consecutive failures from one
host, stop asking. Any success resets the count, so a host that is merely
flaky keeps going.
"""

from collections import defaultdict


# Consecutive failures from one host before it is dropped for this run.
# Low enough to stop quickly, high enough that a handful of bad URLs in a row
# does not retire a working site.
DEFAULT_THRESHOLD = 25


class CircuitBreaker:
    """Tracks consecutive failures per host and trips when they pile up."""

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._consecutive: dict[str, int] = defaultdict(int)
        self._tripped: dict[str, int] = {}

    def record_success(self, host: str) -> None:
        self._consecutive[host] = 0

    def record_failure(self, host: str) -> bool:
        """Count a failure. Returns True if this is the one that trips it."""
        if host in self._tripped:
            return False
        self._consecutive[host] += 1
        if self._consecutive[host] >= self.threshold:
            self._tripped[host] = self._consecutive[host]
            return True
        return False

    def is_open(self, host: str) -> bool:
        """True when this host should not be contacted again this run."""
        return host in self._tripped

    @property
    def tripped_hosts(self) -> dict[str, int]:
        return dict(self._tripped)
