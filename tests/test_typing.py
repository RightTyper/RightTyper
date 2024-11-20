from righttyper.righttyper_runtime import get_full_type, get_adjusted_full_type
from collections.abc import Iterable
from collections import namedtuple
from typing import Any
import pytest
import importlib


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
    assert "list[str]" == get_full_type(dir())

    assert "list[str]" == get_full_type(['a', 'b'])
    assert "list[int]" == get_full_type([0, 1])
    assert "list[tuple[int]]" == get_full_type([(0,), (1,)])

    assert "list[int]" == get_full_type([0, 1][:1])
    assert "int" == get_full_type([0, 1][0])

    #assert "List[int]" == get_full_type([0, 'a'])

    assert "set[str]" == get_full_type({'a', 'b'})
    assert "set[int]" == get_full_type({0, 1})

    # FIXME use Set instead?  specify element type?
    assert "frozenset" == get_full_type(frozenset({'a', 'b'}))
    assert "frozenset" == get_full_type(frozenset({0, 1}))
    assert "frozenset" == get_full_type(frozenset())

    assert "dict[str, str]" == get_full_type({'a': 'b'})

    assert "typing.KeysView[str]" == get_full_type({'a':0, 'b':1}.keys())
    assert "typing.ValuesView[int]" == get_full_type({'a':0, 'b':1}.values())
    assert "typing.ItemsView[str, int]" == get_full_type({'a':0, 'b':1}.items())

    assert "typing.KeysView[typing.Never]" == get_full_type(dict().keys())
    assert "typing.ValuesView[typing.Never]" == get_full_type(dict().values())
    assert "typing.ItemsView[typing.Never, typing.Never]" == get_full_type(dict().items())

    assert "set[str]" == get_full_type({'a', 'b'})
    assert "set[typing.Never]" == get_full_type(set())

    o : Any = range(10)
    assert "range" == get_full_type(o)
    assert 0 == next(iter(o)), "changed state"

    o = iter(range(10))
    assert "typing.Iterator[int]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter([0,1])
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = enumerate([0,1])
    assert "enumerate" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = filter(lambda x:True, [0,1])
    assert "filter" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = reversed([0,1])
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 1 == next(o), "changed state"

    o = zip([0,1], ['a','b'])
    assert "zip" == get_full_type(o)
    assert (0,'a') == next(o), "changed state"

    o = map(lambda x:x, [0,1])
    assert "map" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0, 1})
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1})
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = iter({0:0, 1:1}.values())
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.keys())
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert (0, 0) == next(o), "changed state"

    o = iter({0:0, 1:1}.values())
    assert "typing.Iterator[typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    o = (i for i in range(10))
    assert "typing.Generator[typing.Any, typing.Any, typing.Any]" == get_full_type(o)
    assert 0 == next(o), "changed state"

    Point = namedtuple('Point', ['x', 'y'])
    assert f"{__name__}.Point" == get_full_type(Point(1,1))

    assert f"{__name__}.IterableClass" == get_full_type(IterableClass())
    assert "super" == get_full_type(super(IterableClass))

    assert "slice" == get_full_type(slice(0, 5, 1))
    assert "type" == get_full_type(type(str))
    assert "super" == get_full_type(super(str))

    async def async_range(start):
        for i in range(start):
            yield i

    assert "typing.AsyncGenerator[typing.Any, typing.Any]" == get_full_type(async_range(10))
    assert "typing.AsyncGenerator[typing.Any, typing.Any]" == get_full_type(aiter(async_range(10)))


@pytest.mark.filterwarnings("ignore:coroutine .* never awaited")
def test_get_full_type_coro():
    async def coro():
        import asyncio
        await asyncio.sleep(1)

    assert "typing.Coroutine[typing.Any, typing.Any, typing.Any]" == get_full_type(coro())


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module numpy')
def test_get_full_type_dtype():
    import numpy as np

    assert "numpy.ndarray[typing.Any, numpy.dtypes.Float64DType]" == get_full_type(np.array([], np.float64))


class NonArrayWithDtype:
    def __init__(self):
        self.dtype = 10

def test_non_array_with_dtype():
    # RightTyper used to just check for the presence of a 'dtype' attribute, causing
    # it to generate "mock.MagicMock[Any, mock.MagicMock]" annotations
    assert f"{__name__}.NonArrayWithDtype" == get_full_type(NonArrayWithDtype())


class NamedTupleClass:
    P = namedtuple('P', [])

@pytest.mark.xfail(reason='Not sure how to fix')
def test_get_full_type_namedtuple_in_class():
    # namedtuple's __qualname__ also doesn't contain the enclosing class name...
    assert f"{__name__}.NamedTupleClass.P" == get_full_type(NamedTupleClass.P())


class MyDict(dict):
    def items(self):
        for k, v in super().items():
            yield k, v

def test_get_full_type_dict_with_non_collection_items():
    assert f"{__name__}.MyDict[str, int]" == get_full_type(MyDict({'a': 0}))


class MyList(list):
    pass
class MySet(set):
    pass

def test_get_full_type_custom_collection():
    assert f"{__name__}.MyList[int]" == get_full_type(MyList([0,1]))
    assert f"{__name__}.MySet[int]" == get_full_type(MySet({0,1}))


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

    assert "typing.Self" == get_adjusted_full_type(Foo(), Foo)
    assert f"{__name__}.Foo" == get_adjusted_full_type(Foo())

    assert f"{__name__}.test_adjusted_full_type.<locals>.Bar" == get_adjusted_full_type(Bar())
