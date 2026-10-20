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
import os
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

# NCBI rate limits: 3 req/s anonymous, 10 req/s with API key.
# Detected once at import time; PubMedCrawler also has its own RateLimiter
# that enforces the same ceiling — this is the backstop for the pipeline.
_NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
_ncbi_eutils_limits = (
    HostLimits(max_concurrent=3, min_interval=0.10)  # 10 req/s with key
    if _NCBI_API_KEY
    else HostLimits(max_concurrent=1, min_interval=0.34)  # 3 req/s anonymous
)

# Hostnames that are really one site wearing many names. Most of these 20
# specialty subdomains resolve to 183.60.248.221; a couple (care, nk) sit
# behind a CDN (wswebpic.com) that rotates between two other IPs - so DNS
# does not cleanly prove "one server," but the anti-bot behaviour does.
# Limiting by hostname let the crawl open up to 6 concurrent connections *per
# subdomain*, so the aggregate hitting 39.net's shared anti-bot layer could
# run far past any single domain's intended budget. Logs show exactly that:
# all 20 subdomains started serving the slide-puzzle challenge in the same
# minute, on two different days, while a single plain request to any one of
# them succeeds cleanly right now. That is a volume-triggered WAF reacting to
# aggregate traffic across the family, not a per-domain block - so they share
# one HostLimits bucket, keyed by the group name below, instead of each
# getting its own. jbk.39.net is excluded: it is already fully crawled
# (2,033/2,033) with no anti-bot failures, so there is no evidence it shares
# this fate and no reason to slow it down.
_39NET_SHARED_IP = "39net-specialty-shared-ip"
HOST_GROUPS: dict[str, str] = {
    host: _39NET_SHARED_IP
    for host in (
        "baby.39.net", "care.39.net", "heart.39.net", "fk.39.net",
        "nk.39.net", "cancer.39.net", "tnb.39.net", "woman.39.net", "food.39.net",
        "ek.39.net", "man.39.net", "shen.39.net", "gan.39.net", "wei.39.net",
        "sj.39.net", "fitness.39.net", "gk.39.net", "pf.39.net", "gc.39.net",
        "byby.39.net",
    )
}

HOST_OVERRIDES: dict[str, HostLimits] = {
    # The PubMed crawler's internal RateLimiter is the primary throttle;
    # this backstop matches it so the pipeline never adds extra waiting.
    "eutils.ncbi.nlm.nih.gov": _ncbi_eutils_limits,
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

    # The default 6-concurrent policy drove this host into 99 connect_errors
    # and 139 timeouts in a single run (2026-10-19), which tripped the circuit
    # breaker and abandoned the remaining 56,079 attempts. Unlike the other
    # 39.net subdomains - which fail on an anti-bot interstitial, a content
    # problem no rate change fixes - ask.39.net fails at the network level,
    # and probing it a day later at 0.5 req/s got clean 200s every time. So
    # the server itself is fine; the crawl was simply too fast for it.
    # max_concurrent raised alongside min_interval: at ~4-9s observed response
    # time, 1 concurrent slot could not sustain 3 req/s no matter how low
    # min_interval went - the single in-flight request becomes the real
    # bottleneck. Needs enough slots in flight to keep a new one starting
    # every 0.33s even while earlier ones are still waiting on a reply.
    "ask.39.net": HostLimits(max_concurrent=5, min_interval=0.33),

    # One shared budget for the whole 39.net specialty family (see
    # HOST_GROUPS above) instead of 6 concurrent per subdomain x 20.
    _39NET_SHARED_IP: HostLimits(max_concurrent=6, min_interval=0.33),
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

    def _bucket_key(self, host: str) -> str:
        """The key a host's rate limit is actually tracked under.

        Usually the host itself. A few hosts are known aliases of one real
        backend (HOST_GROUPS) and share a single bucket instead, so pacing
        reflects the load the backend actually sees rather than what any one
        hostname sees.
        """
        return HOST_GROUPS.get(host, host)

    def limits_for(self, host: str) -> HostLimits:
        return self.overrides.get(self._bucket_key(host.lower()), self.default)

    def _state_for(self, host: str) -> "_HostState":
        """Per-host (or per-group) state, created on first sight.

        Deliberately synchronous and lock-free. An earlier version guarded
        this with one asyncio.Lock shared by every host, taken on every single
        request: with hundreds of workers that one lock became the crawl's
        narrowest point, and profiling showed only 42 of 320 workers ever
        busy. Nothing here awaits between the lookup and the insert, so on a
        single-threaded event loop the pair cannot be interleaved and no lock
        is needed.
        """
        key = self._bucket_key(host)
        state = self._states.get(key)
        if state is None:
            limits = self.limits_for(host)
            state = _HostState(
                semaphore=asyncio.Semaphore(limits.max_concurrent),
                min_interval=limits.min_interval,
            )
            self._states[key] = state
        return state

    @asynccontextmanager
    async def request_slot(self, host: str):
        """Hold a slot for one request to `host` (or its shared group)."""
        state = self._state_for(host.lower())

        async with state.semaphore:
            # Reserve this host's next start time synchronously, then sleep
            # until it arrives. Reserving before sleeping is what keeps two
            # workers from claiming the same slot.
            delay = state.reserve_turn()
            if delay > 0:
                await asyncio.sleep(delay)
            yield

    def describe(self, hosts: set[str] | None = None) -> str:
        active_hosts = set(self.overrides) if hosts is None else hosts
        lines = []
        if hosts is None or any(host not in self.overrides for host in hosts):
            lines.append(
                f"  default              {self.default.max_concurrent} concurrent, "
                f"{self.default.min_interval:.2f}s apart "
                f"(~{self.default.requests_per_second:.1f} req/s per host)"
            )
        for host in sorted(active_hosts & self.overrides.keys()):
            limits = self.overrides[host]
            lines.append(
                f"  {host:20s} {limits.max_concurrent} concurrent, "
                f"{limits.min_interval:.2f}s apart"
            )
        return "\n".join(lines) if lines else "  no active hosts"
