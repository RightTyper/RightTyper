from righttyper.righttyper_types import TypeInfo, TypeInfoSet
from righttyper.righttyper_utils import union_typeset_str
from collections.abc import Iterable
from collections import namedtuple
from typing import Any
import pytest
import importlib


def get_full_type(*args, **kwargs) -> str:
    import righttyper.righttyper_runtime as rt
    return str(rt.get_full_type(*args, **kwargs))

def type_from_annotations(*args, **kwargs) -> str:
    import righttyper.righttyper_runtime as rt
    return str(rt.type_from_annotations(*args, **kwargs))


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
    assert "list[typing.Never]" == get_full_type([])
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
    assert "dict[typing.Never, typing.Never]" == get_full_type(dict())

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


@pytest.mark.skipif((importlib.util.find_spec('numpy') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_full_type_numpy_jaxtyping():
    import numpy as np

    assert 'jaxtyping.Float64[numpy.ndarray, "0"]' == get_full_type(np.array([], np.float64), use_jaxtyping=True)
    assert 'jaxtyping.Float16[numpy.ndarray, "1 1 1"]' == \
            get_full_type(np.array([[[1]]], np.float16), use_jaxtyping=True)


@pytest.mark.skipif((importlib.util.find_spec('torch') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_full_type_torch_jaxtyping():
    import torch

    assert 'jaxtyping.Float64[torch.Tensor, "0"]' == \
            get_full_type(torch.tensor([], dtype=torch.float64), use_jaxtyping=True)
    assert 'jaxtyping.Int32[torch.Tensor, "2 1"]' == \
            get_full_type(torch.tensor([[1],[2]], dtype=torch.int32), use_jaxtyping=True)


def test_type_from_annotations():
    def foo(x: int|float, y: list[tuple[bool, ...]]) -> complex|None:
        pass

    assert "typing.Callable[[int | float, list[tuple[bool, ...]]], complex | None]" == type_from_annotations(foo)


def test_typeinfo():
    assert "foo.bar" == str(TypeInfo("foo", "bar"))
    assert "foo.bar[m.baz, \"x y\"]" == str(TypeInfo("foo", "bar", (TypeInfo("m", "baz"), "\"x y\"")))
    assert "int" == str(TypeInfo("", "int"))
    assert "tuple[bool]" == str(TypeInfo("", "tuple", args=('bool',)))


def test_union_typeset():
    assert "None" == union_typeset_str(TypeInfoSet({}))
    assert "bool" == union_typeset_str({TypeInfo("", "bool")})

    assert "bool|int|zoo.bar" == union_typeset_str({
            TypeInfo("", "bool"),
            TypeInfo("", "int"),
            TypeInfo("zoo", "bar"),
        }
    )

    assert "bool|int|None" == union_typeset_str({
            TypeInfo("", "None"),
            TypeInfo("", "bool"),
            TypeInfo("", "int"),
        }
    )


def test_union_typeset_generics():
    assert "list[bool|int]|None" == union_typeset_str({
            TypeInfo("", "list", args=(TypeInfo("", "int"),)),
            TypeInfo("", "list", args=(TypeInfo("", "bool"),)),
            TypeInfo("", "None")
        }
    )

    assert "list[tuple[bool|int, float]]" == union_typeset_str({
            TypeInfo("", "list", args=(
                TypeInfo("", "tuple", args=(
                    TypeInfo("", "bool"),
                    TypeInfo("", "float"),
                )),
            )),
            TypeInfo("", "list", args=(
                TypeInfo("", "tuple", args=(
                    TypeInfo("", "int"),
                    TypeInfo("", "float"),
                )),
            )),
        }
    )

    assert "list[tuple[bool, float]|tuple[float]]" == union_typeset_str({
            TypeInfo("", "list", args=(
                TypeInfo("", "tuple", args=(
                    TypeInfo("", "bool"),
                    TypeInfo("", "float"),
                )),
            )),
            TypeInfo("", "list", args=(
                TypeInfo("", "tuple", args=(
                    TypeInfo("", "float"),
                )),
            )),
        }
    )


def test_union_typeset_generics_str_not_merged():
    assert "Callable[[], None]|Callable[[int], None]" == union_typeset_str({
            TypeInfo("", "Callable", args=(
                "[], None",
            )),
            TypeInfo("", "Callable", args=(
                "[int], None",
            )),
        }
    )


def test_union_typeset_superclass():
    class A: pass
    class B(A): pass
    class C(B): pass
    class D(B): pass

    assert f"{__name__}.{B.__qualname__}" == union_typeset_str({
            TypeInfo.from_type(C),
            TypeInfo.from_type(D)
        }
    )

    assert f"{__name__}.{B.__qualname__}" == union_typeset_str({
            TypeInfo.from_type(B),
            TypeInfo.from_type(D)
        }
    )

    assert f"{__name__}.{A.__qualname__}" == union_typeset_str({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D)
        }
    )


def test_union_typeset_superclass_bare_type():
    # invoking type.mro() raises an exception
    assert "builtins.int|builtins.type" == union_typeset_str({
            TypeInfo.from_type(int),
            TypeInfo.from_type(type)
        }
    )


def test_union_typeset_generics_superclass():
    class A: pass
    class B(A): pass
    class C(B): pass
    class D(B): pass

    assert f"list[{__name__}.{B.__qualname__}]|None" == union_typeset_str({
            TypeInfo("", "list", args=(TypeInfo.from_type(C),)),
            TypeInfo("", "list", args=(TypeInfo.from_type(D),)),
            TypeInfo("", "None")
        }
    )
