import pytest
from content_api.rate_limit import TokenBucket


class FakeClock:
    """A clock that only moves when the test says so. No real waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_bucket_of_5_allows_5_then_refuses_the_6th() -> None:
    bucket = TokenBucket(capacity=5, refill_per_second=1, clock=FakeClock())
    assert [bucket.allow() for _ in range(5)] == [True] * 5
    assert bucket.allow() is False


def test_one_more_request_passes_after_1_second() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_per_second=1, clock=clock)
    for _ in range(5):
        bucket.allow()
    assert bucket.allow() is False

    clock.advance(1.0)
    assert bucket.allow() is True
    assert bucket.allow() is False  # only one token came back


def test_never_holds_more_than_capacity_after_an_hour() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_per_second=1, clock=clock)
    bucket.allow()

    clock.advance(3600)
    assert bucket.tokens == 5
    assert [bucket.allow() for _ in range(5)] == [True] * 5
    assert bucket.allow() is False


def test_half_a_second_is_not_enough_for_a_token() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_per_second=1, clock=clock)
    bucket.allow()

    clock.advance(0.5)
    assert bucket.allow() is False
    clock.advance(0.5)
    assert bucket.allow() is True


def test_starts_full() -> None:
    bucket = TokenBucket(capacity=3, refill_per_second=1, clock=FakeClock())
    assert bucket.tokens == 3


@pytest.mark.parametrize(
    ("capacity", "rate", "field"),
    [(0, 1, "capacity"), (-1, 1, "capacity"), (5, 0, "refill_per_second")],
)
def test_bad_settings_raise_clear_error(capacity: int, rate: float, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        TokenBucket(capacity=capacity, refill_per_second=rate, clock=FakeClock())
