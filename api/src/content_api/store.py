"""An in-memory store for posts. Everything is lost when the server stops."""

from content_api.models import Post


class PostStore:
    """Keeps posts in a dict, giving each new post the next id (1, 2, 3 ...)."""

    def __init__(self) -> None:
        self._posts: dict[int, Post] = {}
        self._next_id = 1

    def add(self, post: Post) -> int:
        """Save a new post and return its id."""
        post_id = self._next_id
        self._posts[post_id] = post
        self._next_id += 1
        return post_id

    def get(self, post_id: int) -> Post | None:
        """Return the post with this id, or None if there isn't one."""
        return self._posts.get(post_id)

    def find(
        self, niche: str | None = None, platform: str | None = None
    ) -> dict[int, Post]:
        """Return the posts that match every filter given (all posts if none)."""
        return {
            post_id: post
            for post_id, post in self._posts.items()
            if (niche is None or post.niche == niche)
            and (platform is None or post.platform == platform)
        }

    def replace(self, post_id: int, post: Post) -> None:
        """Overwrite an existing post."""
        self._posts[post_id] = post

    def delete(self, post_id: int) -> bool:
        """Remove a post. Return False if there was no post with that id."""
        return self._posts.pop(post_id, None) is not None
