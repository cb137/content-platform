"""Client tests. respx fakes the network: any request a test hasn't set up fails."""

import httpx
import pytest
import respx
from content_api.http_client import GITHUB_API, fetch_all_issues, fetch_json

URL = "https://example.test/data"
ISSUES_URL = f"{GITHUB_API}/repos/octo/demo/issues"


class FakeSleep:
    """Records how long the client asked to wait, without waiting."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@respx.mock
def test_429_then_200_succeeds() -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"ok": True})]
    )
    sleep = FakeSleep()
    assert fetch_json(URL, sleep=sleep) == {"ok": True}
    assert route.call_count == 2
    assert sleep.waits == [1]


@respx.mock
def test_404_fails_at_once() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    sleep = FakeSleep()
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        fetch_json(URL, sleep=sleep)
    assert exc_info.value.response.status_code == 404
    assert route.call_count == 1
    assert sleep.waits == []


@respx.mock
def test_four_500s_give_up() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(500))
    sleep = FakeSleep()
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        fetch_json(URL, sleep=sleep)
    assert exc_info.value.response.status_code == 500
    assert route.call_count == 4
    assert sleep.waits == [1, 2, 4]


@respx.mock
def test_other_5xx_are_retried_too() -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=[1, 2])]
    )
    assert fetch_json(URL, sleep=FakeSleep()) == [1, 2]
    assert route.call_count == 2


@respx.mock
def test_uses_the_client_given() -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    with httpx.Client() as client:
        assert fetch_json(URL, client=client, sleep=FakeSleep()) == {"ok": True}


@respx.mock
def test_fetch_all_issues_follows_every_page_and_skips_pull_requests() -> None:
    page_2 = f"{ISSUES_URL}?state=open&per_page=100&page=2"
    route = respx.get(url__startswith=ISSUES_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=[
                    {"number": 1},
                    {"number": 2, "pull_request": {"url": "..."}},
                ],
                headers={"Link": f'<{page_2}>; rel="next"'},
            ),
            httpx.Response(200, json=[{"number": 3}]),
        ]
    )
    issues = fetch_all_issues("octo", "demo", sleep=FakeSleep())

    assert [issue["number"] for issue in issues] == [1, 3]
    assert route.call_count == 2
    first, second = (call.request.url for call in route.calls)
    assert first.params["state"] == "open"
    assert first.params["per_page"] == "100"
    assert second.params["page"] == "2"


@respx.mock
def test_fetch_all_issues_retries_a_busy_page() -> None:
    respx.get(url__startswith=ISSUES_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=[{"number": 7}])]
    )
    sleep = FakeSleep()
    assert fetch_all_issues("octo", "demo", sleep=sleep) == [{"number": 7}]
    assert sleep.waits == [1]


@respx.mock
def test_fetch_all_issues_can_ask_for_all_states() -> None:
    route = respx.get(url__startswith=ISSUES_URL).mock(
        return_value=httpx.Response(200, json=[])
    )
    assert fetch_all_issues("octo", "demo", state="all", sleep=FakeSleep()) == []
    assert route.calls[0].request.url.params["state"] == "all"
