"""The posts web API. Run it from api/ with:

    uv run uvicorn content_api.main:app --reload

then open http://localhost:8000/docs
"""

from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from content_api.models import Platform, Post, describe_error
from content_api.store import PostStore

app = FastAPI(title="content-api", description="Create and manage social posts.")

_store = PostStore()


def get_store() -> PostStore:
    """Give each route the store. Tests swap in a fresh one."""
    return _store


StoreDep = Annotated[PostStore, Depends(get_store)]


class PostOut(Post):
    """A post as the API returns it: the post fields plus its id."""

    id: int


class PostUpdate(BaseModel):
    """The fields a PATCH may change. Leave a field out to keep it as it is."""

    model_config = ConfigDict(extra="forbid")  # a typo like "hok" is an error

    niche: str | None = None
    platform: str | None = None
    hook: str | None = None
    pain_points: list[str] | None = None
    body: str | None = None
    cta: str | None = None
    hashtags: list[str] | None = None


def plain_english(errors: Iterable[Mapping[str, Any]]) -> list[str]:
    """Turn validation errors into lines like "hook: has 20 words, ..."."""
    messages = []
    for detail in errors:
        location = [str(part) for part in detail["loc"]]
        # FastAPI starts the location with where the value came from; drop that.
        if location and location[0] in ("body", "query", "path"):
            location = location[1:]
        field = ".".join(location) or "body"
        messages.append(f"{field}: {describe_error(detail)}")
    return messages


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    """Send every 422 in the same plain-English format."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": plain_english(error.errors())},
    )


def with_id(post_id: int, post: Post) -> PostOut:
    return PostOut.model_validate({"id": post_id, **post.model_dump()})


def get_or_404(store: PostStore, post_id: int) -> Post:
    post = store.get(post_id)
    if post is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"post {post_id} not found")
    return post


@app.get("/health")
def health() -> dict[str, str]:
    """Return status ok while the server is running."""
    return {"status": "ok"}


@app.post("/posts", status_code=status.HTTP_201_CREATED)
def create_post(post: Post, store: StoreDep) -> PostOut:
    """Save a new post. 422 if it breaks a post rule."""
    return with_id(store.add(post), post)


@app.get("/posts")
def list_posts(
    store: StoreDep, niche: str | None = None, platform: Platform | None = None
) -> list[PostOut]:
    """List posts, optionally only one niche and/or platform."""
    found = store.find(niche=niche, platform=platform)
    return [with_id(post_id, post) for post_id, post in found.items()]


@app.get("/posts/{post_id}")
def get_post(post_id: int, store: StoreDep) -> PostOut:
    """Get one post. 404 if there is no post with that id."""
    return with_id(post_id, get_or_404(store, post_id))


@app.patch("/posts/{post_id}")
def update_post(post_id: int, changes: PostUpdate, store: StoreDep) -> PostOut:
    """Change some fields of a post. 404 if missing, 422 if the result breaks a rule."""
    current = get_or_404(store, post_id)
    merged = current.model_dump() | changes.model_dump(exclude_unset=True)
    try:
        # Check the whole post again: a change to one field (e.g. platform)
        # can break a rule that depends on another (e.g. body length).
        updated = Post.model_validate(merged)
    except ValidationError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, plain_english(error.errors())
        ) from error
    store.replace(post_id, updated)
    return with_id(post_id, updated)


@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int, store: StoreDep) -> None:
    """Delete a post. 404 if there is no post with that id."""
    if not store.delete(post_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"post {post_id} not found")
