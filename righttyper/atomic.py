import itertools
import sys

class AtomicCounter:
    """Implements a lockless atomic counter.
       It assumes that itertools.count is implemented in C, and that the GIL makes
       each call into itertools.count, which are single python expressions, atomic.
       Based on a GitHub gist by Phil Marsh.
    """

    def __init__(self) -> None:
        assert not hasattr(sys, '_is_gil_enabled') or sys._is_gil_enabled(), "Not using the GIL"
        assert not hasattr(itertools.count, "__code__"), "Itertools.count() not written in C"

        self._count = itertools.count()
        self._last = 0

    def inc(self) -> None:
        next(self._count)

    def count_and_clear(self) -> int:
        # This operation isn't, by itself, atomic, but safe as long as only one thread
        # ever calls it
        current = next(self._count)
        value = current - self._last
        self._last = current + 1    # reading 'current' increments
        return value
