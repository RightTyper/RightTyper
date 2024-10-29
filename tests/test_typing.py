from righttyper.righttyper_runtime import get_full_type, get_adjusted_full_type
from collections.abc import Iterable
from typing import Any


class IterableClass(Iterable):
    def __iter__(self):
        return None


def test_get_full_type():
    assert "bool" == get_full_type(True)
    assert "bool" == get_full_type(False)
    assert "int" == get_full_type(10)
    assert "float" == get_full_type(0.0)
    assert "str" == get_full_type('foo')

    assert "str" == get_full_type(bin(0))
    assert "bool" == get_full_type(bool(0))

    assert "bytearray" == get_full_type(bytearray(b'0000'))
    assert "bytes" == get_full_type(bytes(b'0000'))
    assert "complex" == get_full_type(complex(1, 1))
    assert "List[str]" == get_full_type(dir())

    assert "List[str]" == get_full_type(['a', 'b'])
    assert "List[int]" == get_full_type([0, 1])

    assert "List[int]" == get_full_type([0, 1][:1])
    assert "int" == get_full_type([0, 1][0])

    #assert "List[int]" == get_full_type([0, 'a'])

    assert "Set[str]" == get_full_type({'a', 'b'})
    assert "Set[int]" == get_full_type({0, 1})

    # FIXME use Set instead?  specify element type?
    assert "frozenset" == get_full_type(frozenset({'a', 'b'}))
    assert "frozenset" == get_full_type(frozenset({0, 1}))
    assert "frozenset" == get_full_type(frozenset())

    assert "Dict[str, str]" == get_full_type({'a': 'b'})

    assert "Iterable[str]" == get_full_type({'a':0, 'b':1}.keys())
    assert "Iterable[int]" == get_full_type({'a':0, 'b':1}.values())
    assert "Iterable[Tuple[str, int]]" == get_full_type({'a':0, 'b':1}.items())

    # FIXME is it useful to have 'Never' here? Or better simply 'Iterable' ?
    assert "Iterable[Never]" == get_full_type(dict().keys())
    assert "Iterable[Never]" == get_full_type(dict().values())
    assert "Iterable[Tuple[Never, Never]]" == get_full_type(dict().items())

    assert "Set[str]" == get_full_type({'a', 'b'})

    o : Any = range(10)
    assert "Iterable[int]" == get_full_type(o)
    assert 0 == next(iter(o)), "changed state"

    o = iter(range(10))
    assert "Iterator[int]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter([0,1])
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = enumerate([0,1])
    assert "Iterator[Tuple[int, Any]]" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = filter(lambda x:True, [0,1])
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = reversed([0,1])
    assert "Iterator[Any]" == get_full_type(o)
    assert 1 == next(o), "changed state"

    o = zip([0,1], ['a','b'])
    assert "Iterator[Any]" == get_full_type(o)
    assert (0,'a') == next(o), "changed state"

    o = map(lambda x:x, [0,1])
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

    o = iter({0:0, 1:1}.values())
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.keys())
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "Iterator[Any]" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = iter({0:0, 1:1}.values())
    assert "Iterator[Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = (i for i in range(10))
    assert "Generator[Any, None, None]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    assert "IterableClass" == get_full_type(IterableClass())
    assert "super" == get_full_type(super(IterableClass))

    assert "slice" == get_full_type(slice(0, 5, 1))
    assert "type" == get_full_type(type(str))
    assert "super" == get_full_type(super(str))

    async def async_range(start):
        for i in range(start):
            yield i

    assert "AsyncGenerator[Any, None, None]" == get_full_type(async_range(10))
    assert "AsyncGenerator[Any, None, None]" == get_full_type(aiter(async_range(10)))


class Foo:
    pass

def test_adjusted_full_type():
    # these types used to be special cased... ensure they still work
    assert "None" == get_adjusted_full_type(None)
    assert "bool" == get_adjusted_full_type(True)
    assert "float" == get_adjusted_full_type(.0)
    assert "int" == get_adjusted_full_type(0)

    # get_adjusted_full_type's main function is to translate to 'Self'

    class Bar:
        pass

    assert "Self" == get_adjusted_full_type(Foo(), f"{Foo.__module__}.{Foo.__qualname__}")
    assert f"{Foo.__module__}.{Foo.__qualname__}" == get_adjusted_full_type(Foo(), f"{Foo.__module__}.Baz")

    assert "Self" == get_adjusted_full_type(Bar(), f"{Bar.__module__}.{Bar.__qualname__}")
    assert f"{Bar.__module__}.{Bar.__qualname__}" == get_adjusted_full_type(Bar(), f"{Bar.__module__}.Baz")
