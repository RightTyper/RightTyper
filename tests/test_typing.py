from righttyper.righttyper_runtime import get_full_type
from collections.abc import Iterable


class IterableClass(Iterable):
    def __iter__(self):
        return None


def test_get_full_type():
    assert "bool" == get_full_type(True)
    assert "bool" == get_full_type(False)

    assert "int" == get_full_type(10)
    assert "float" == get_full_type(0.0)
    assert "str" == get_full_type('foo')

    assert "List[str]" == get_full_type(['a', 'b'])
    assert "List[int]" == get_full_type([0, 1])

    assert "Set[str]" == get_full_type({'a', 'b'})
    assert "Set[int]" == get_full_type({0, 1})

    assert "Dict[str, str]" == get_full_type({'a': 'b'})

    #assert "List[int]" == get_full_type([0, 'a'])

    o = range(10)
    assert "Iterable[int]" == get_full_type(o)
    assert 0 == next(iter(o)), "changed state"

    o = iter(range(10))
    assert "Iterator[int]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter([0,1])
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0, 1})
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1})
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "Iterator[Any]" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = (i for i in range(10))
    assert "Generator[Any, None, None]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    assert "IterableClass" == get_full_type(IterableClass())

    async def async_range(start):
        for i in range(start):
            yield y

    assert "AsyncGenerator[Any, None, None]" == get_full_type(async_range(10))
    assert "AsyncGenerator[Any, None, None]" == get_full_type(aiter(async_range(10)))
