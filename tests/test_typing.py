from righttyper.righttyper_types import TypeInfo, NoneTypeInfo, AnyTypeInfo, Sample
from righttyper.typeinfo import merged_types
import righttyper.righttyper_runtime as rt
from collections.abc import Iterable
from collections import namedtuple
from typing import Any, Callable
import pytest
import importlib
import types


def get_value_type(*args, **kwargs) -> str:
    return str(rt.get_value_type(*args, **kwargs))

def type_from_annotations(*args, **kwargs) -> str:
    return str(rt.type_from_annotations(*args, **kwargs))


class IterableClass(Iterable):
    def __iter__(self):
        return None


def test_get_value_type():
    assert NoneTypeInfo is rt.get_value_type(None)

    assert "bool" == get_value_type(True)
    assert "bool" == get_value_type(False)
    assert "int" == get_value_type(10)
    assert "float" == get_value_type(0.0)
    assert "str" == get_value_type('foo')

    assert "str" == get_value_type(bin(0))
    assert "bool" == get_value_type(bool(0))

    assert "bytearray" == get_value_type(bytearray(b'0000'))
    assert "bytes" == get_value_type(bytes(b'0000'))
    assert "complex" == get_value_type(complex(1, 1))
    assert "list[str]" == get_value_type(dir())

    assert "list[str]" == get_value_type(['a', 'b'])
    assert "list[typing.Never]" == get_value_type([])
    assert "list[int]" == get_value_type([0, 1])
    assert "list[tuple[int]]" == get_value_type([(0,), (1,)])

    assert "list[int]" == get_value_type([0, 1][:1])
    assert "int" == get_value_type([0, 1][0])

    #assert "List[int]" == get_value_type([0, 'a'])

    assert "set[str]" == get_value_type({'a', 'b'})
    assert "set[int]" == get_value_type({0, 1})

    # FIXME use Set instead?  specify element type?
    assert "frozenset" == get_value_type(frozenset({'a', 'b'}))
    assert "frozenset" == get_value_type(frozenset({0, 1}))
    assert "frozenset" == get_value_type(frozenset())

    assert "dict[str, str]" == get_value_type({'a': 'b'})
    assert "dict[typing.Never, typing.Never]" == get_value_type(dict())

    assert "typing.KeysView[str]" == get_value_type({'a':0, 'b':1}.keys())
    assert "typing.ValuesView[int]" == get_value_type({'a':0, 'b':1}.values())
    assert "typing.ItemsView[str, int]" == get_value_type({'a':0, 'b':1}.items())

    assert "typing.KeysView[typing.Never]" == get_value_type(dict().keys())
    assert "typing.ValuesView[typing.Never]" == get_value_type(dict().values())
    assert "typing.ItemsView[typing.Never, typing.Never]" == get_value_type(dict().items())

    assert "set[str]" == get_value_type({'a', 'b'})
    assert "set[typing.Never]" == get_value_type(set())

    o : Any = range(10)
    assert "range" == get_value_type(o)
    assert 0 == next(iter(o)), "changed state"

    o = iter(range(10))
    assert "typing.Iterator[int]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter([0,1])
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = enumerate([0,1])
    assert "enumerate" == get_value_type(o)
    assert (0, 0) == next(o), "changed state"

    o = filter(lambda x:True, [0,1])
    assert "filter" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = reversed([0,1])
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 1 == next(o), "changed state"

    o = zip([0,1], ['a','b'])
    assert "zip" == get_value_type(o)
    assert (0,'a') == next(o), "changed state"

    o = map(lambda x:x, [0,1])
    assert "map" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0, 1})
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1})
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert (0, 0) == next(o), "changed state"

    o = iter({0:0, 1:1}.values())
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.keys())
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = iter({0:0, 1:1}.items())
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert (0, 0) == next(o), "changed state"

    o = iter({0:0, 1:1}.values())
    assert "typing.Iterator[typing.Any]" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = (i for i in range(10))
    assert "typing.Generator" == get_value_type(o)
    assert 0 == next(o), "changed state"

    Point = namedtuple('Point', ['x', 'y'])
    assert f"{__name__}.Point" == get_value_type(Point(1,1))

    assert f"{__name__}.IterableClass" == get_value_type(IterableClass())
    assert "super" == get_value_type(super(IterableClass))

    assert "slice" == get_value_type(slice(0, 5, 1))
    assert "type[str]" == get_value_type(str)
    assert "type" == get_value_type(type(str))
    assert "super" == get_value_type(super(str))

    async def async_range(start):
        for i in range(start):
            yield i

    assert "typing.AsyncGenerator" == get_value_type(async_range(10))
    assert "typing.AsyncGenerator" == get_value_type(aiter(async_range(10)))


@pytest.mark.filterwarnings("ignore:coroutine .* never awaited")
def test_get_value_type_coro():
    async def coro():
        import asyncio
        await asyncio.sleep(1)

    assert "typing.Coroutine" == get_value_type(coro())


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module numpy')
def test_get_value_type_dtype():
    import numpy as np

    assert "numpy.ndarray[typing.Any, numpy.dtypes.Float64DType]" == get_value_type(np.array([], np.float64))


class NonArrayWithDtype:
    def __init__(self):
        self.dtype = 10

def test_non_array_with_dtype():
    # RightTyper used to just check for the presence of a 'dtype' attribute, causing
    # it to generate "mock.MagicMock[Any, mock.MagicMock]" annotations
    assert f"{__name__}.NonArrayWithDtype" == get_value_type(NonArrayWithDtype())


class NamedTupleClass:
    P = namedtuple('P', [])

@pytest.mark.xfail(reason='Not sure how to fix')
def test_get_value_type_namedtuple_in_class():
    # namedtuple's __qualname__ also doesn't contain the enclosing class name...
    assert f"{__name__}.NamedTupleClass.P" == get_value_type(NamedTupleClass.P())


class MyDict(dict):
    def items(self):
        for k, v in super().items():
            yield k, v

def test_get_value_type_dict_with_non_collection_items():
    assert f"{__name__}.MyDict[str, int]" == get_value_type(MyDict({'a': 0}))


class MyList(list):
    pass
class MySet(set):
    pass

def test_get_value_type_custom_collection():
    assert f"{__name__}.MyList[int]" == get_value_type(MyList([0,1]))
    assert f"{__name__}.MySet[int]" == get_value_type(MySet({0,1}))


@pytest.mark.skipif((importlib.util.find_spec('numpy') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_numpy_jaxtyping():
    import numpy as np

    assert 'jaxtyping.Float64[numpy.ndarray, "0"]' == get_value_type(np.array([], np.float64), use_jaxtyping=True)
    assert 'jaxtyping.Float16[numpy.ndarray, "1 1 1"]' == \
            get_value_type(np.array([[[1]]], np.float16), use_jaxtyping=True)


@pytest.mark.skipif((importlib.util.find_spec('torch') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_torch_jaxtyping():
    import torch

    assert 'jaxtyping.Float64[torch.Tensor, "0"]' == \
            get_value_type(torch.tensor([], dtype=torch.float64), use_jaxtyping=True)
    assert 'jaxtyping.Int32[torch.Tensor, "2 1"]' == \
            get_value_type(torch.tensor([[1],[2]], dtype=torch.int32), use_jaxtyping=True)


def test_type_from_annotations():
    def foo(x: int|float, y: list[tuple[bool, ...]]) -> complex|None:
        pass

    assert "typing.Callable[[int | float, list[tuple[bool, ...]]], complex | None]" == type_from_annotations(foo)


def test_typeinfo():
    assert "foo.bar" == str(TypeInfo("foo", "bar"))
    assert "foo.bar[m.baz, \"x y\"]" == str(TypeInfo("foo", "bar", (TypeInfo("m", "baz"), "\"x y\"")))
    assert "int" == str(TypeInfo("", "int"))
    assert "tuple[bool]" == str(TypeInfo("", "tuple", args=('bool',)))

    t = TypeInfo.from_type(type(None))
    assert t.module == ''
    assert t.name == 'None'
    assert t.type_obj is type(None)
    assert str(t) == "None"


def test_typeinfo_from_set():
    t = TypeInfo.from_set(set())
    assert t == NoneTypeInfo    # or should this be Never ?

    t = TypeInfo.from_set({TypeInfo.from_type(int)})

    assert str(t) == 'builtins.int'
    assert t.name == 'int'
    assert not t.args

    assert t is not TypeInfo.from_type(int) # should be new object
    assert t.type_obj is int

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(bool)
        })

    assert str(t) == 'builtins.bool|builtins.int'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(type(None)),
            TypeInfo.from_type(bool),
            TypeInfo(module='', name='z')
        })

    assert str(t) == 'builtins.bool|builtins.int|z|None'
    assert isinstance(t.args[-1], TypeInfo)
    assert t.args[-1].name == 'None'


def test_merged_types():
    assert "None" == str(merged_types(set()))
    assert "bool" == str(merged_types({TypeInfo("", "bool")}))

    assert "bool|int|zoo.bar" == str(merged_types({
            TypeInfo("", "bool"),
            TypeInfo("", "int"),
            TypeInfo("zoo", "bar"),
        }
    ))

    assert "bool|int|None" == str(merged_types({
            TypeInfo.from_type(type(None)),
            TypeInfo("", "bool"),
            TypeInfo("", "int"),
        }
    ))


def test_merged_types_generics():
    assert "list[bool|int]|None" == str(merged_types({
            TypeInfo("", "list", args=(TypeInfo("", "int"),)),
            TypeInfo("", "list", args=(TypeInfo("", "bool"),)),
            TypeInfo.from_type(type(None))
        }
    ))

    assert "list[tuple[bool|int, float]]" == str(merged_types({
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
    ))

    assert "list[tuple[bool, float]|tuple[float]]" == str(merged_types({
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
    ))


def test_merged_types_generics_str_not_merged():
    # the [...] parameters in Callable are passed as a string, which we don't merge (yet)
    assert "Callable[[], None]|Callable[[int], None]" == str(merged_types({
            TypeInfo("", "Callable", args=(
                "[], None",
            )),
            TypeInfo("", "Callable", args=(
                "[int], None",
            )),
        }
    ))


def test_merged_types_superclass():
    class A: pass
    class B(A): pass
    class C(B): pass
    class D(B): pass

    assert f"{__name__}.{B.__qualname__}" == str(merged_types({
            TypeInfo.from_type(C),
            TypeInfo.from_type(D)
        }
    ))

    assert f"{__name__}.{B.__qualname__}" == str(merged_types({
            TypeInfo.from_type(B),
            TypeInfo.from_type(D)
        }
    ))

    assert f"{__name__}.{A.__qualname__}" == str(merged_types({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D)
        }
    ))


def test_merged_types_superclass_bare_type():
    # invoking type.mro() raises an exception
    assert "builtins.int|builtins.type" == str(merged_types({
            TypeInfo.from_type(int),
            TypeInfo.from_type(type)
        }
    ))


def test_merged_types_generics_superclass():
    class A: pass
    class B(A): pass
    class C(B): pass
    class D(B): pass

    assert f"list[{__name__}.{B.__qualname__}]|None" == str(merged_types({
            TypeInfo("", "list", args=(TypeInfo.from_type(C),)),
            TypeInfo("", "list", args=(TypeInfo.from_type(D),)),
            TypeInfo.from_type(type(None))
        }
    ))

str_ti = TypeInfo("", "str", type_obj=str)
int_ti = TypeInfo("", "int", type_obj=int)
bool_ti = TypeInfo("", "bool", type_obj=bool)
any_ti = AnyTypeInfo
generator_ti = lambda *a: TypeInfo("typing", "Generator", tuple(a))
iterator_ti = lambda *a: TypeInfo("typing", "Iterator", tuple(a))
union_ti = lambda *a: TypeInfo("types", "UnionType", tuple(a), type_obj=types.UnionType)

def generate_sample(func: Callable, *args) -> Sample:
    import righttyper.righttyper_runtime as rt

    res = func(*args)
    sample = Sample(tuple(rt.get_value_type(arg) for arg in args))
    if type(res).__name__ == "generator":
        try:
            while True:
                nex = next(res) # this can fail
                sample.yields.add(rt.get_value_type(nex))
        except StopIteration as e:
            if e.value is not None:
                sample.returns = rt.get_value_type(e.value)
    else:
        sample.returns = rt.get_value_type(res)

    return sample


def test_sample_process_simple():
    def dog(a):
        return a

    sample = generate_sample(dog, "hi")
    assert sample == Sample((str_ti,), returns=str_ti)
    assert sample.process() == (str_ti, str_ti)


def test_sample_process_generator():
    def dog(a, b):
        yield a
        return b

    sample = generate_sample(dog, 1, "hi")
    assert sample == Sample((int_ti, str_ti,), {int_ti}, str_ti)
    assert sample.process() == (int_ti, str_ti, generator_ti(int_ti, any_ti, str_ti))


def test_sample_process_iterator_union():
    def dog(a, b):
        yield a
        yield b

    sample = generate_sample(dog, 1, "hi")
    assert sample == Sample((int_ti, str_ti,), yields={int_ti, str_ti})
    assert sample.process() == (int_ti, str_ti, iterator_ti(union_ti(int_ti, str_ti)))


def test_sample_process_iterator():
    def dog(a):
        yield a

    sample = generate_sample(dog, "hi")
    assert sample == Sample((str_ti,), yields={str_ti})
    assert sample.process() == (str_ti, iterator_ti((str_ti)))


def test_sample_process_generator_union():
    def dog(a, b, c):
        yield a
        yield b
        return c

    sample = generate_sample(dog, 1, "hi", True)
    assert sample == Sample((int_ti, str_ti, bool_ti,), {int_ti, str_ti}, bool_ti)
    assert sample.process() == (int_ti, str_ti, bool_ti, generator_ti(union_ti(int_ti, str_ti), any_ti, bool_ti))
