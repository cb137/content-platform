"""An HTTP client that retries when a server is busy or failing."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

MAX_ATTEMPTS = 4  # 1 try + 3 retries, waiting 1, 2 and 4 seconds between them
TIMEOUT_SECONDS = 10.0
GITHUB_API = "https://api.github.com"

Sleep = Callable[[float], None]


def is_retryable(error: BaseException) -> bool:
    """Retry on 429 (too many requests) and 5xx (server errors) only.

    Other errors, like 404 (not found), would fail again, so don't retry them.
    """
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    code = error.response.status_code
    return code == 429 or code >= 500


@contextmanager
def _open_client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Use the client given, or open (and afterwards close) a new one."""
    if client is not None:
        yield client
    else:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as new_client:
            yield new_client


def get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
) -> httpx.Response:
    """GET a URL, retrying on 429 and 5xx. Raises httpx.HTTPStatusError on failure.

    `sleep` is how the waits happen; tests pass a fake one so nothing really waits.
    """
    retrying = Retrying(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1),  # 1, 2, 4 seconds
        stop=stop_after_attempt(MAX_ATTEMPTS),
        sleep=sleep,
        reraise=True,  # give the caller the real HTTP error, not tenacity's wrapper
    )

    def attempt(http: httpx.Client) -> httpx.Response:
        response = http.get(url, params=params, headers=headers)
        response.raise_for_status()  # turns 4xx/5xx into an HTTPStatusError
        return response

    with _open_client(client) as http:
        return retrying(attempt, http)


def fetch_json(
    url: str, *, client: httpx.Client | None = None, sleep: Sleep = time.sleep
) -> Any:
    """GET a URL (with retries) and return the decoded JSON."""
    return get_with_retry(url, client=client, sleep=sleep).json()


def fetch_all_issues(
    owner: str,
    repo: str,
    *,
    state: Literal["open", "closed", "all"] = "open",
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
) -> list[dict[str, Any]]:
    """Fetch every page of a public GitHub repo's issues (pull requests left out)."""
    url: str | None = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    params: dict[str, Any] | None = {"state": state, "per_page": 100}
    headers = {"Accept": "application/vnd.github+json"}
    issues: list[dict[str, Any]] = []

    with _open_client(client) as http:
        while url is not None:
            response = get_with_retry(
                url, params=params, headers=headers, client=http, sleep=sleep
            )
            # GitHub's issues list also includes pull requests; they have this key.
            issues.extend(
                item for item in response.json() if "pull_request" not in item
            )
            # GitHub puts the next page's full URL in the Link header.
            url = response.links.get("next", {}).get("url")
            params = None  # the next URL already contains the query
    return issues
