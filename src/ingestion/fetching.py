"""HTTP layer for the crawler.

The job of this module is to return the HTML of a real article, or raise.
It never returns a page that merely *looks* like a success, because the
expensive failure mode is a 2xx response whose body is an anti-bot
interstitial: that body parses fine, saves fine, and silently poisons the
dataset.
"""

import asyncio

import httpx


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MedicalKBResearch/0.1; "
        "+https://example.org/bot)"
    )
}

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


def build_client(timeout: float) -> httpx.AsyncClient:
    """One shared client per run, so connections are pooled across requests."""
    return httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    )


def _validate(response: httpx.Response) -> str:
    """Return the HTML body, or raise explaining why it is not content."""
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


async def fetch_html(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 3,
) -> tuple[str, str, int]:
    """Fetch one article.

    Returns (html, final_url, status_code). Raises FetchError when every
    attempt failed, including when the server answered but the body was not
    an article. Retries are worth it even for blocks: anti-bot pages are
    often served intermittently rather than permanently.
    """
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await client.get(url)
            html = _validate(response)
            return html, str(response.url), response.status_code
        except (httpx.HTTPError, FetchError) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(2**attempt, 8))

    raise FetchError(
        f"Failed after {retries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )
