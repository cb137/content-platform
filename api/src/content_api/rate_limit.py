"""A token bucket rate limiter."""

import time
from collections.abc import Callable


class TokenBucket:
    """Allow bursts up to `capacity` requests, refilling over time.

    The bucket starts full. Each allowed request uses one token. Tokens come
    back at `refill_per_second`, but the bucket never holds more than `capacity`.
    """

    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be more than 0, but got {capacity}")
        if refill_per_second <= 0:
            raise ValueError(
                f"refill_per_second must be more than 0, but got {refill_per_second}"
            )
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._clock = clock
        self._tokens = float(capacity)
        self._last_refill = clock()

    @property
    def tokens(self) -> float:
        """How many tokens are in the bucket right now."""
        self._refill()
        return self._tokens

    def allow(self) -> bool:
        """Use one token and return True, or return False if the bucket is empty."""
        self._refill()
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False

    def _refill(self) -> None:
        """Add the tokens earned since the last refill, up to capacity."""
        now = self._clock()
        earned = (now - self._last_refill) * self.refill_per_second
        self._tokens = min(self.capacity, self._tokens + earned)
        self._last_refill = now
