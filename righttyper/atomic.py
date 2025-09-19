import itertools

class AtomicCounter:
    """Implements a lockless atomic counter.
       It assumes that itertools.count is implemented in C, and that the GIL makes
       each call into itertools.count, which are single python expressions, atomic.
       Based on a GitHub gist by Phil Marsh.
    """

    def __init__(self):
        assert not hasattr(itertools.count, "__code__"), "Itertools.count() not written in C"

        self._incs = itertools.count()
        self._decs = itertools.count()

    def inc(self):
        next(self._incs)

    def dec(self):
        next(self._decs)

    def count(self):
        return next(self._incs) - next(self._decs)

