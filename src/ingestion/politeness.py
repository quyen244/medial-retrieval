"""Per-host rate limiting.

A global concurrency cap is not politeness. The corpus is grouped by domain -
a thousand consecutive rows usually belong to one site - so a global cap of 8
means eight simultaneous connections to the *same* server, sustained for
hundreds of thousands of rows. That is the access pattern that got this
client blocked by NCBI.

This module caps what one host sees: how many requests may be in flight to it
at once, and how closely together they may arrive. The global cap still limits
total load; this limits load per victim.

Pair it with interleaving the work across domains (see crawler.load_samples),
otherwise a per-host limit simply makes the whole run as slow as one host.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class HostLimits:
    """How hard a single host may be hit.

    Of the two knobs, max_concurrent is usually the one that matters. Real
    latency to these sites measured 1.5s at p50, so a host capped at 2 in
    flight delivers 2/1.5 = 1.3 docs per second and never gets anywhere near
    a 0.5s min_interval. Raising concurrency to 6 moved the whole crawl from
    15.5 to 40.2 docs/s with p50 latency rising only 0.66s to 0.83s and no
    change in the error count.
    """
    max_concurrent: int = 6
    min_interval: float = 0.2   # seconds between request starts

    @property
    def requests_per_second(self) -> float:
        return 1.0 / self.min_interval if self.min_interval > 0 else float("inf")


# Hosts that need gentler treatment than the default, usually because they
# have already shown they will block a client that pushes.
HOST_OVERRIDES: dict[str, HostLimits] = {
    # NCBI documents 3 requests/second without an API key. The PubMed crawler
    # also rate limits itself; this is the backstop.
    "eutils.ncbi.nlm.nih.gov": HostLimits(max_concurrent=1, min_interval=0.34),
    "pubmed.ncbi.nlm.nih.gov": HostLimits(max_concurrent=1, min_interval=1.0),

    # The three largest crawlable domains. A short probe said they tolerate
    # 10 requests per second, and that probe was misleading: a burst of ten
    # is not the same as hours of steady load. Five other domains were tuned
    # from exactly that kind of evidence and ended up blocking this client
    # outright. These get 5/s, which is half the probed figure, because they
    # are the ones the corpus cannot afford to lose.
    "suckhoecongdongonline.vn": HostLimits(max_concurrent=5, min_interval=0.2),
    "suckhoedoisong.vn": HostLimits(max_concurrent=5, min_interval=0.2),
    "thanhnien.vn": HostLimits(max_concurrent=5, min_interval=0.2),

    # Currently refusing this client. Sustained crawling took all five from
    # working to answering 403 even to a single slow request, so they are held
    # at a trickle for whenever the block lifts. The circuit breaker stops the
    # crawl from spending requests on them in the meantime.
    "nhathuoclongchau.com.vn": HostLimits(max_concurrent=1, min_interval=2.0),
    "baohaiphong.vn": HostLimits(max_concurrent=1, min_interval=2.0),
    "baonghean.vn": HostLimits(max_concurrent=1, min_interval=2.0),
    "baodanang.vn": HostLimits(max_concurrent=1, min_interval=2.0),
    "hanoimoi.vn": HostLimits(max_concurrent=1, min_interval=2.0),
    "mega.vietnamplus.vn": HostLimits(max_concurrent=1, min_interval=2.0),
}


@dataclass
class _HostState:
    """Everything the policy tracks about one host."""
    semaphore: asyncio.Semaphore
    min_interval: float
    next_allowed: float = 0.0

    def reserve_turn(self) -> float:
        """Claim this host's next start slot, returning how long to wait for it."""
        if self.min_interval <= 0:
            return 0.0
        now = time.monotonic()
        start_at = max(now, self.next_allowed)
        self.next_allowed = start_at + self.min_interval
        return start_at - now


class PolitenessPolicy:
    """Hands out permission to make a request, one host at a time.

    Usage:
        async with policy.request_slot("example.com"):
            await client.get(url)

    The slot is held for the duration of the request, so `max_concurrent`
    bounds real connections rather than just call sites.
    """

    def __init__(
        self,
        default: HostLimits | None = None,
        overrides: dict[str, HostLimits] | None = None,
    ):
        self.default = default or HostLimits()
        self.overrides = dict(HOST_OVERRIDES)
        if overrides:
            self.overrides.update(overrides)

        self._states: dict[str, _HostState] = {}

    def limits_for(self, host: str) -> HostLimits:
        return self.overrides.get(host.lower(), self.default)

    def _state_for(self, host: str) -> "_HostState":
        """Per-host state, created on first sight.

        Deliberately synchronous and lock-free. An earlier version guarded
        this with one asyncio.Lock shared by every host, taken on every single
        request: with hundreds of workers that one lock became the crawl's
        narrowest point, and profiling showed only 42 of 320 workers ever
        busy. Nothing here awaits between the lookup and the insert, so on a
        single-threaded event loop the pair cannot be interleaved and no lock
        is needed.
        """
        state = self._states.get(host)
        if state is None:
            limits = self.limits_for(host)
            state = _HostState(
                semaphore=asyncio.Semaphore(limits.max_concurrent),
                min_interval=limits.min_interval,
            )
            self._states[host] = state
        return state

    @asynccontextmanager
    async def request_slot(self, host: str):
        """Hold a slot for one request to `host`."""
        state = self._state_for(host.lower())

        async with state.semaphore:
            # Reserve this host's next start time synchronously, then sleep
            # until it arrives. Reserving before sleeping is what keeps two
            # workers from claiming the same slot.
            delay = state.reserve_turn()
            if delay > 0:
                await asyncio.sleep(delay)
            yield

    def describe(self) -> str:
        lines = [
            f"  default              {self.default.max_concurrent} concurrent, "
            f"{self.default.min_interval:.2f}s apart "
            f"(~{self.default.requests_per_second:.1f} req/s per host)"
        ]
        for host, limits in sorted(self.overrides.items()):
            lines.append(
                f"  {host:20s} {limits.max_concurrent} concurrent, "
                f"{limits.min_interval:.2f}s apart"
            )
        return "\n".join(lines)
