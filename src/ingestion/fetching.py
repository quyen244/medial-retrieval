"""HTTP layer for the crawler.

The job of this module is to return the HTML of a real article, or raise.
It never returns a page that merely *looks* like a success, because the
expensive failure mode is a 2xx response whose body is an anti-bot
interstitial: that body parses fine, saves fine, and silently poisons the
dataset.
"""

import asyncio
from urllib.parse import urlparse

import httpx

from src.ingestion.politeness import PolitenessPolicy


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MedicalKBResearch/0.1; "
        "+https://example.org/bot)"
    )
}

# Some WAFs reject a request not because of the User-Agent but because the
# rest of the header set does not look like a browser. Three provincial news
# sites in this corpus answer 403 to the honest bot header above and 200 to
# this one - the difference is the Sec-Fetch-*, sec-ch-ua and
# Upgrade-Insecure-Requests headers a real Chrome always sends.
#
# Accept-Encoding deliberately omits "br": httpx only decodes brotli when the
# optional dependency is installed, and advertising it otherwise returns a
# body that looks like binary noise.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().split("@")[-1].split(":")[0]

# Interstitials answer with a 2xx status and an HTML body, so neither
# raise_for_status() nor a Content-Type check rejects them. PubMed serves one
# under HTTP 203; Cloudflare serves one under 403 but can also use 200.
BLOCK_PAGE_MARKERS = (
    "cookies must be enabled",
    "enable cookies for",
    "attention required! | cloudflare",
    "just a moment...",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "verifying you are human",
    "请开启cookie",
)

# An interstitial is always a small page. Requiring that stops the markers
# from matching an article that merely discusses cookies or browser security.
BLOCK_PAGE_MAX_BYTES = 30_000


class FetchError(RuntimeError):
    """Base class for every reason a page is not usable content."""


class UnexpectedStatusError(FetchError):
    """A 2xx status other than 200: the body is not the requested document."""


class NotHtmlError(FetchError):
    """The response carried a Content-Type that cannot hold an article."""


class BlockedPageError(FetchError):
    """The server answered 2xx with an anti-bot interstitial."""


def detect_block_page(html: str) -> str | None:
    """Return the marker identifying an interstitial, or None if the page looks real."""
    if len(html) > BLOCK_PAGE_MAX_BYTES:
        return None
    head = html[:8000].lower()
    for marker in BLOCK_PAGE_MARKERS:
        if marker in head:
            return marker
    return None


def classify_failure(exc: Exception) -> str:
    """A short, groupable reason a page could not be fetched.

    Used both by the measurement report and by the failure ledger, so that
    "what is wrong with this domain" reads the same in both places.
    """
    message = str(exc)
    for needle, label in (
        ("got 203", "http_203_block"),
        ("interstitial", "anti_bot_page"),
        ("403", "http_403"),
        ("404", "http_404"),
        ("429", "http_429"),
        ("500", "http_500"),
        ("503", "http_503"),
        ("521", "http_521"),
        ("Content-Type", "not_html"),
        ("Timeout", "timeout"),
        ("ConnectError", "connect_error"),
    ):
        if needle in message:
            return label
    return type(exc).__name__


def build_client(
    timeout: float, max_connections: int = 100
) -> httpx.AsyncClient:
    """One shared client per run, so connections are pooled across requests.

    `max_connections` has to be raised in step with the worker count. httpx
    defaults it to 100, and that default silently caps the whole crawl: with
    320 workers, 220 of them sit in the connection queue rather than on the
    network. Profiling showed a fetch taking 8.8 seconds against a measured
    network latency of 1.5 seconds, with only 37 of 320 slots ever busy -
    all of it queueing for a connection that did not exist.
    """
    return httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=max_connections,
            # Keeping connections alive matters here: the crawl returns to the
            # same 50 hosts a million times, and a fresh TLS handshake per
            # document would cost more than the request.
            max_keepalive_connections=max_connections // 2,
            keepalive_expiry=30.0,
        ),
    )


def validate_response(response: httpx.Response) -> str:
    """Return the HTML body, or raise explaining why it is not content.

    Public because site crawlers that issue their own follow-up requests -
    answering a cookie challenge, say - must apply the same checks.
    """
    response.raise_for_status()  # rejects 4xx and 5xx only

    if response.status_code != 200:
        raise UnexpectedStatusError(f"Expected HTTP 200, got {response.status_code}")

    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and "xhtml" not in content_type:
        raise NotHtmlError(f"Expected HTML, got Content-Type={content_type!r}")

    html = response.text
    marker = detect_block_page(html)
    if marker is not None:
        raise BlockedPageError(f"Anti-bot interstitial, matched {marker!r}")

    return html


async def polite_get(
    client: httpx.AsyncClient,
    url: str,
    policy: PolitenessPolicy | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET a URL, holding the target host's rate-limit slot for the request.

    Every request in the pipeline goes through here, so no code path can
    accidentally bypass the per-host limits.
    """
    if policy is None:
        return await client.get(url, headers=headers)

    async with policy.request_slot(host_of(url)):
        return await client.get(url, headers=headers)


def _is_permanent(exc: Exception) -> bool:
    """True when retrying cannot possibly change the outcome.

    A 404 or a wrong Content-Type describes the URL itself, not a transient
    condition - the tenth attempt fails for exactly the same reason as the
    first. Retrying these burns a worker slot on backoff sleeps (up to 7s
    across the default 3 retries) for nothing, and at scale that adds up: in
    one run, thousands of permanently-404 and image-typed URLs each cost
    their worker several idle seconds, which is what made the whole crawl
    look stalled despite still making progress.

    Anti-bot blocks (UnexpectedStatusError, BlockedPageError) and 5xx/429 stay
    retryable - those genuinely can resolve on a later attempt, as measured
    earlier where 2 of 3 identical PubMed requests returned a block page and
    one did not.
    """
    if isinstance(exc, NotHtmlError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    return False


async def fetch_html(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 3,
    policy: PolitenessPolicy | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[str, str, int]:
    """Fetch one article.

    Returns (html, final_url, status_code). Raises FetchError when every
    attempt failed, including when the server answered but the body was not
    an article. Retries are worth it for blocks - anti-bot pages are often
    served intermittently - but not for errors the URL itself guarantees,
    which fail the same way every time; see _is_permanent.
    """
    last_error: Exception | None = None
    attempts_made = 0

    for attempt in range(retries + 1):
        attempts_made += 1
        try:
            response = await polite_get(client, url, policy, headers)
            html = validate_response(response)
            return html, str(response.url), response.status_code
        except (httpx.HTTPError, FetchError) as exc:
            last_error = exc
            if _is_permanent(exc):
                break
            if attempt < retries:
                await asyncio.sleep(min(2**attempt, 8))

    raise FetchError(
        f"Failed after {attempts_made} attempt(s): "
        f"{type(last_error).__name__}: {last_error}"
    )
