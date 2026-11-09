from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from content_api.main import app, get_store
from content_api.store import PostStore


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A test client whose store starts empty for every test."""
    store = PostStore()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def post_json(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "niche": "ai-engineering",
        "platform": "linkedin",
        "hook": "Most RAG demos fail in production.",
        "pain_points": ["Wrong answers", "Slow retrieval"],
        "body": "Test the retrieval step before you trust the model.",
        "cta": "Comment RAG for my checklist",
        "hashtags": ["#RAG", "#LLM", "#AI"],
    }
    data.update(changes)
    return data


def create(client: TestClient, **changes: Any) -> dict[str, Any]:
    response = client.post("/posts", json=post_json(**changes))
    assert response.status_code == 201
    result: dict[str, Any] = response.json()
    return result


# health
def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# create
def test_create_returns_post_with_id(client: TestClient) -> None:
    created = create(client)
    assert created == {"id": 1, **post_json()}


def test_create_gives_new_ids(client: TestClient) -> None:
    assert create(client)["id"] == 1
    assert create(client)["id"] == 2


def test_create_breaking_a_rule_is_422_in_plain_english(client: TestClient) -> None:
    response = client.post("/posts", json=post_json(hook=" ".join(["w"] * 20)))
    assert response.status_code == 422
    assert response.json() == {
        "detail": ["hook: has 20 words, but must be 15 words or fewer"]
    }


def test_create_missing_field_is_422(client: TestClient) -> None:
    data = post_json()
    del data["cta"]
    response = client.post("/posts", json=data)
    assert response.status_code == 422
    assert response.json() == {"detail": ["cta: is required"]}


def test_create_lists_every_broken_rule(client: TestClient) -> None:
    response = client.post("/posts", json=post_json(niche="", hashtags=["#a"]))
    assert response.status_code == 422
    assert response.json()["detail"] == [
        "niche: is required and cannot be blank",
        "hashtags: has 1 hashtags, but must have 3 to 5",
    ]


def test_create_does_not_save_a_bad_post(client: TestClient) -> None:
    client.post("/posts", json=post_json(cta=""))
    assert client.get("/posts").json() == []


# list
def test_list_starts_empty(client: TestClient) -> None:
    response = client.get("/posts")
    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_all_posts(client: TestClient) -> None:
    create(client)
    create(client, niche="career")
    assert [post["id"] for post in client.get("/posts").json()] == [1, 2]


def test_list_filters_by_niche(client: TestClient) -> None:
    create(client, niche="career")
    create(client, niche="marketing")
    found = client.get("/posts", params={"niche": "career"}).json()
    assert [post["niche"] for post in found] == ["career"]


def test_list_filters_by_platform(client: TestClient) -> None:
    create(client, platform="linkedin")
    create(client, platform="tiktok")
    found = client.get("/posts", params={"platform": "tiktok"}).json()
    assert [post["platform"] for post in found] == ["tiktok"]


def test_list_filters_by_niche_and_platform(client: TestClient) -> None:
    create(client, niche="career", platform="linkedin")
    create(client, niche="career", platform="tiktok")
    create(client, niche="marketing", platform="tiktok")
    found = client.get("/posts", params={"niche": "career", "platform": "tiktok"})
    assert [post["id"] for post in found.json()] == [2]


def test_list_unknown_platform_filter_is_422(client: TestClient) -> None:
    response = client.get("/posts", params={"platform": "facebook"})
    assert response.status_code == 422
    [message] = response.json()["detail"]
    assert message.startswith("platform: ")


# get one
def test_get_one_post(client: TestClient) -> None:
    create(client)
    response = client.get("/posts/1")
    assert response.status_code == 200
    assert response.json()["hook"] == "Most RAG demos fail in production."


def test_get_missing_post_is_404(client: TestClient) -> None:
    response = client.get("/posts/99")
    assert response.status_code == 404
    assert response.json() == {"detail": "post 99 not found"}


def test_get_with_non_number_id_is_422(client: TestClient) -> None:
    response = client.get("/posts/abc")
    assert response.status_code == 422
    assert response.json()["detail"][0].startswith("post_id: ")


# update
def test_update_changes_only_the_fields_sent(client: TestClient) -> None:
    create(client)
    response = client.patch("/posts/1", json={"cta": "Follow for more"})
    assert response.status_code == 200
    assert response.json() == {"id": 1, **post_json(cta="Follow for more")}
    assert client.get("/posts/1").json()["cta"] == "Follow for more"


def test_update_missing_post_is_404(client: TestClient) -> None:
    response = client.patch("/posts/99", json={"cta": "Follow"})
    assert response.status_code == 404
    assert response.json() == {"detail": "post 99 not found"}


def test_update_breaking_a_rule_is_422_and_keeps_old_post(client: TestClient) -> None:
    create(client)
    response = client.patch("/posts/1", json={"hook": " ".join(["w"] * 20)})
    assert response.status_code == 422
    assert response.json() == {
        "detail": ["hook: has 20 words, but must be 15 words or fewer"]
    }
    assert client.get("/posts/1").json()["hook"] == post_json()["hook"]


def test_update_rechecks_rules_across_fields(client: TestClient) -> None:
    # A long body is fine on YouTube, but not once the platform becomes LinkedIn.
    create(client, platform="youtube", body="x" * 4000)
    response = client.patch("/posts/1", json={"platform": "linkedin"})
    assert response.status_code == 422
    assert response.json()["detail"][0].startswith("body: is 4,000 characters")


def test_update_unknown_field_is_422(client: TestClient) -> None:
    create(client)
    response = client.patch("/posts/1", json={"hok": "typo"})
    assert response.status_code == 422
    assert response.json()["detail"][0].startswith("hok: ")


# delete
def test_delete_post(client: TestClient) -> None:
    create(client)
    response = client.delete("/posts/1")
    assert response.status_code == 204
    assert client.get("/posts/1").status_code == 404


def test_delete_missing_post_is_404(client: TestClient) -> None:
    response = client.delete("/posts/99")
    assert response.status_code == 404
    assert response.json() == {"detail": "post 99 not found"}


def test_deleted_ids_are_not_reused(client: TestClient) -> None:
    create(client)
    client.delete("/posts/1")
    assert create(client)["id"] == 2
