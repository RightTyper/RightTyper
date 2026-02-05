from righttyper.typeinfo import TypeInfo, NoneTypeInfo, AnyTypeInfo, UnknownTypeInfo
from righttyper.generalize import merged_types, generalize
import righttyper.type_id as t_id
import collections.abc as abc
from collections import namedtuple
import typing
from typing import Any, Callable, get_type_hints, Union, Optional, List, Literal, cast, Self, Never
import pytest
import importlib
import types
from righttyper.options import run_options
from enum import Enum
import sys
from righttyper.typemap import TypeMap, AdjustTypeNamesT, CheckTypeNamesT

rt_get_value_type = t_id.get_value_type

# This plugin is needed to avoid caching results that depend upon 'run_options'
assert importlib.util.find_spec('pytest_antilru') is not None, "pytest-antilru missing"


def get_value_type(v, **kwargs) -> str:
    return str(rt_get_value_type(v, **kwargs))


def get_type_name(t) -> TypeInfo:
    ti = t_id.get_type_name(cast(type, t))
    typemap = TypeMap(sys.modules['__main__'].__dict__)
    if run_options.adjust_type_names:
        return AdjustTypeNamesT(typemap).visit(ti)
    else:
        return CheckTypeNamesT(typemap).visit(ti)
    return ti
 

class IterableClass(abc.Iterable):
    def __iter__(self):
        return None


class MyGeneric[A, B](dict): pass


T = typing.TypeVar("T")
class MyOldGeneric(typing.Generic[T]): pass


def test_get_value_type():
    assert NoneTypeInfo is rt_get_value_type(None)

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

    assert "set[str]" == get_value_type({'a', 'b'})
    assert "set[int]" == get_value_type({0, 1})
    assert "set[typing.Never]" == get_value_type(set())

    assert "frozenset[str]" == get_value_type(frozenset({'a', 'b'}))
    assert "frozenset[int]" == get_value_type(frozenset({0, 1}))
    assert "frozenset[typing.Never]" == get_value_type(frozenset())

    assert "dict[str, str]" == get_value_type({'a': 'b'})
    assert "dict[typing.Never, typing.Never]" == get_value_type(dict())

    assert "types.MappingProxyType[str, int]" == get_value_type(types.MappingProxyType({'a': 1}))
    assert "types.MappingProxyType[typing.Never, typing.Never]" == get_value_type(types.MappingProxyType(dict()))

    assert "tuple[()]" == get_value_type(tuple())
    assert "tuple[int, str]" == get_value_type((1, "foo"))

    assert "collections.abc.KeysView[str]" == get_value_type({'a':0, 'b':1}.keys())
    assert "collections.abc.ValuesView[int]" == get_value_type({'a':0, 'b':1}.values())
    assert "collections.abc.ItemsView[str, int]" == get_value_type({'a':0, 'b':1}.items())

    assert "collections.abc.KeysView[typing.Never]" == get_value_type(dict().keys())
    assert "collections.abc.ValuesView[typing.Never]" == get_value_type(dict().values())
    assert "collections.abc.ItemsView[typing.Never, typing.Never]" == get_value_type(dict().items())

    o : Any = range(10)
    assert "range" == get_value_type(o)
    assert 0 == next(iter(o)), "changed state"

    o = filter(lambda x:True, [0,1])
    assert "filter" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = map(lambda x:x, [0,1])
    assert "map" == get_value_type(o)
    assert 0 == next(o), "changed state"

    o = (i for i in range(10))
    assert get_value_type(o) == "collections.abc.Generator"
    assert 0 == next(o), "changed state"

    assert f"{__name__}.IterableClass" == get_value_type(IterableClass())
    assert "super" == get_value_type(super(IterableClass))

    assert "slice" == get_value_type(slice(0, 5, 1))
    assert "type[str]" == get_value_type(str)
    assert "type" == get_value_type(type(str))
    assert "super" == get_value_type(super(str))

    async def async_range(start):
        for i in range(start):
            yield i

    assert get_value_type(async_range(10)) == "collections.abc.AsyncGenerator"
    assert get_value_type(aiter(async_range(10))) == "collections.abc.AsyncGenerator"

    assert f"{__name__}.MyGeneric[int, str]" == get_value_type(MyGeneric[int, str]())
    assert f"{__name__}.MyOldGeneric[int]" == get_value_type(MyOldGeneric[int]())

@pytest.mark.parametrize("init, name, nextv", [
    ["iter(b'0')", "collections.abc.Iterator[int]", b'0'[0]],
    ["iter(bytearray(b'0'))", "collections.abc.Iterator[int]", bytearray(b'0')[0]],
    ["iter({'a': 0})", "collections.abc.Iterator[str]", 'a'],
    ["iter({'a': 0}.values())", "collections.abc.Iterator[int]", 0],
    ["iter({'a': 0}.items())", "collections.abc.Iterator[tuple[str, int]]", ('a', 0)],
    ["iter([0, 1])", "collections.abc.Iterator[int]", 0],
    ["iter(reversed([0, 1]))", "collections.abc.Iterator[int]", 1],
    ["iter(range(1))", "collections.abc.Iterator[int]", 0],
    ["iter(range(1 << 1000))", "collections.abc.Iterator[int]", 0],
    ["iter({'a'})", "collections.abc.Iterator[str]", 'a'],
    ["iter('ab')", "collections.abc.Iterator[str]", 'a'],
    ["iter(('a', 0))", "collections.abc.Iterator[int|str]", 'a'],
    ["iter(tuple(c for c in ('a', 0)))", "collections.abc.Iterator[int|str]", 'a'],
    ["zip([0], ('a',))", "collections.abc.Iterator[tuple[int, str]]", (0, 'a')],
    ["iter(zip([0], ('a',)))", "collections.abc.Iterator[tuple[int, str]]", (0, 'a')],
    ["enumerate(('a', 'b'))", "enumerate[str]", (0, 'a')],
    ["iter(enumerate(('a', 'b')))", "enumerate[str]", (0, 'a')],
    # The generator in these cases needs to be observed to fully type... see integration test.
#    ["iter(zip([0], (c for c in ('a',))))", "collections.abc.Iterator[tuple[int, str]]", (0, 'a')],
#    ["enumerate(c for c in ('a', 'b'))", "enumerate[str]", (0, 'a')],
])
def test_value_type_iterator(init, name, nextv):
    obj = eval(init)
    assert name == get_value_type(obj)
    assert nextv == next(obj), "changed state"


@pytest.mark.parametrize("init, name", [
    ["iter(b'0')", "collections.abc.Iterator[int]"],
    ["iter(bytearray(b'0'))", "collections.abc.Iterator[int]"],
    ["iter({'a': 0})", "collections.abc.Iterator"],
    ["iter({'a': 0}.values())", "collections.abc.Iterator"],
    ["iter({'a': 0}.items())", "collections.abc.Iterator"],
    ["iter([0, 1])", "collections.abc.Iterator"],
    ["iter(reversed([0, 1]))", "collections.abc.Iterator"],
    ["iter(range(1))", "collections.abc.Iterator[int]"],
    ["iter(range(1 << 1000))", "collections.abc.Iterator[int]"],
    ["iter({'a'})", "collections.abc.Iterator"],
    ["iter('ab')", "collections.abc.Iterator[str]"],
    ["iter(('a', 'b'))", "collections.abc.Iterator"],
    ["zip([0], ('a',))", "zip"],
    ["iter(zip([0], ('a',)))", "zip"],
    ["enumerate(('a', 'b'))", "enumerate"],
    ["iter(enumerate(('a', 'b')))", "enumerate"],
])
def test_type_name_iterator(init, name):
    assert name == str(get_type_name(type(eval(init))))


@pytest.mark.parametrize("adjust_type_names", [False, True])
def test_dynamic_type(monkeypatch, adjust_type_names):
    monkeypatch.setattr(run_options, 'adjust_type_names', adjust_type_names)

    t = type('myType', (object,), dict())
    assert get_type_name(t) is UnknownTypeInfo


@pytest.mark.parametrize("adjust_type_names", [False, True])
def test_local_type_module_invalid(monkeypatch, adjust_type_names):
    monkeypatch.setattr(run_options, 'adjust_type_names', adjust_type_names)

    class Invalid: pass
    Invalid.__module__ = 'does_not_exist'

    assert get_type_name(Invalid).module == "does_not_exist"


def test_items_from_typing():
    assert TypeInfo("typing", "Any") == get_type_name(Any)
    assert TypeInfo("typing", "Self") == get_type_name(Self)
    assert TypeInfo("typing", "List") == get_type_name(List)
    assert TypeInfo("typing", "Callable") == get_type_name(Callable)
    assert TypeInfo("typing", "Sequence") == get_type_name(typing.Sequence)
    assert TypeInfo("typing", "Never") == get_type_name(typing.Never)

    # the concrete "List[int]" isn't defined anywhere
    assert UnknownTypeInfo is get_type_name(List[int])


@pytest.mark.filterwarnings("ignore:coroutine .* never awaited")
def test_get_value_type_coro():
    async def coro():
        import asyncio
        await asyncio.sleep(1)

    assert "collections.abc.Coroutine[None, None, typing.Any]" == get_value_type(coro())


@pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='missing module numpy')
def test_get_value_type_dtype():
    import numpy as np

    assert "numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]" == get_value_type(np.array([], np.float64))


class NonArrayWithDtype:
    def __init__(self):
        self.dtype = 10

def test_non_array_with_dtype():
    # RightTyper used to just check for the presence of a 'dtype' attribute, causing
    # it to generate "mock.MagicMock[Any, mock.MagicMock]" annotations
    assert f"{__name__}.NonArrayWithDtype" == get_value_type(NonArrayWithDtype())


class NamedTupleClass:
    P = namedtuple('P', [])

@pytest.mark.parametrize("adjust_type_names", [False, True])
def test_get_type_name_namedtuple_nonlocal(monkeypatch, adjust_type_names):
    monkeypatch.setattr(run_options, 'adjust_type_names', adjust_type_names)

    if adjust_type_names:
        # namedtuple's __qualname__ also doesn't contain the enclosing class name...
        assert f"{__name__}.NamedTupleClass.P" == get_type_name(NamedTupleClass.P).fullname()
    else:
        assert UnknownTypeInfo is get_type_name(NamedTupleClass.P)


class Decision(Enum):
    NO = 0
    MAYBE = 1
    YES = 2

def test_get_value_type_enum():
    assert f"{__name__}.Decision" == get_value_type(Decision.MAYBE)


@pytest.mark.xfail(reason="How to best solve this?")
def test_get_value_type_namedtuple_local():
    # namedtuple's __qualname__ lacks context
    P = namedtuple('P', ['x', 'y'])
    assert f"{__name__}.test_get_value_type_namedtuple_local.<locals>.P" == get_value_type(P(1,1))


@pytest.mark.skipif((importlib.util.find_spec('numpy') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_numpy_jaxtyping(monkeypatch):
    monkeypatch.setattr(run_options, 'infer_shapes', True)
    import numpy as np

    assert 'jaxtyping.Float64[numpy.ndarray, "0"]' == get_value_type(np.array([], np.float64))
    assert 'jaxtyping.Float16[numpy.ndarray, "1 1 1"]' == \
            get_value_type(np.array([[[1]]], np.float16))


@pytest.mark.skipif((importlib.util.find_spec('torch') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_torch_jaxtyping(monkeypatch):
    monkeypatch.setattr(run_options, 'infer_shapes', True)
    import torch

    assert 'jaxtyping.Float64[torch.Tensor, "0"]' == \
            get_value_type(torch.tensor([], dtype=torch.float64))
    assert 'jaxtyping.Int32[torch.Tensor, "2 1"]' == \
            get_value_type(torch.tensor([[1],[2]], dtype=torch.int32))


def test_type_from_annotations():
    def foo(x: int|float, y: list[tuple[bool, ...]], z: Callable[[], None]) -> complex|None:
        pass

    assert "collections.abc.Callable[[float|int, list[tuple[bool, ...]], collections.abc.Callable[[], None]], complex|None]" == \
            get_value_type(foo)


@pytest.mark.skipif((importlib.util.find_spec('jaxtyping') is None or
                     importlib.util.find_spec('numpy') is None),
                    reason='missing modules')
def test_hint2type():
    import jaxtyping
    import jax

    def foo(
        x: int | MyGeneric[str, Callable[[], None]],
        y: tuple[int, ...],
        z: jaxtyping.Float[jax.Array, "10 20"],
        t: tuple[()],
        u: tuple,
    ): pass

    hints = get_type_hints(foo)

    assert f"int|{__name__}.MyGeneric[str, collections.abc.Callable[[], None]]" == str(t_id.hint2type(hints['x']))
    assert f"tuple[int, ...]" == str(t_id.hint2type(hints['y']))
    assert f"tuple[()]" == str(t_id.hint2type(hints['t']))
    assert f"tuple" == str(t_id.hint2type(hints['u']))
    assert """jaxtyping.Float[jax.Array, "10 20"]""" == str(t_id.hint2type(hints['z']))


def test_hint2type_pre_3_10():
    def foo(
        x: Optional[Union[int, str]],
        y: Optional[bool]
    ): pass

    hints = get_type_hints(foo)

    assert "int|str|None" == str(t_id.hint2type(hints['x']))
    assert "bool|None" == str(t_id.hint2type(hints['y']))


def test_typeinfo():
    assert "foo.bar" == str(TypeInfo("foo", "bar"))
    assert "foo.bar[m.baz, \"x y\"]" == str(TypeInfo("foo", "bar", (TypeInfo("m", "baz"), "x y")))
    assert "int" == str(TypeInfo.from_type(int))
    assert "tuple[bool]" == str(TypeInfo.from_type(tuple, args=(TypeInfo.from_type(bool),)))

    t = TypeInfo.from_type(type(None))
    assert t.module == ''
    assert t.name == 'None'
    assert t.type_obj is type(None)
    assert str(t) == "None"


def test_typeinfo_from_set():
    t = TypeInfo.from_set(set())
    assert str(t) == "typing.Never"

    t = TypeInfo.from_set(set(), empty_is_none=True)
    assert t is NoneTypeInfo

    t = TypeInfo.from_set({TypeInfo.from_type(int)})

    assert str(t) == 'int'
    assert t.name == 'int'
    assert not t.args

    assert t is not TypeInfo.from_type(int) # should be new object
    assert t.type_obj is int

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            NoneTypeInfo
        })

    assert str(t) == 'int|None'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(bool)
        })

    assert str(t) == 'bool|int'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo(module='', name='X', args=(
                TypeInfo.from_set({
                    TypeInfo.from_type(bool),
                    NoneTypeInfo
                }),
            ))
        })

    assert str(t) == 'X[bool|None]|int'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(type(None)),
            TypeInfo.from_type(bool),
            TypeInfo(module='', name='z')
        })

    assert str(t) == 'bool|int|z|None'
    assert isinstance(t.args[-1], TypeInfo)
    assert t.args[-1].name == 'None'


def test_merged_types():
    assert "typing.Never" == str(merged_types(set()))
    assert "bool" == str(merged_types({TypeInfo.from_type(bool)}))

    assert "int|str|zoo.bar" == str(merged_types({
            TypeInfo.from_type(str),
            TypeInfo.from_type(int),
            TypeInfo("zoo", "bar"),
        }
    ))

    assert "bool|str|None" == str(merged_types({
            TypeInfo.from_type(type(None)),
            TypeInfo.from_type(bool),
            TypeInfo.from_type(str),
        }
    ))


def test_merged_types_generics():
    assert "list[bool]|list[int]" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(list, args=(TypeInfo.from_type(bool),))
        }
    ))

    assert "list" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(list),
        }
    ))

    assert "tuple" == str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int), ...)),
            TypeInfo.from_type(tuple),
        }
    ))

    assert "tuple" == str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int), ...)),
            TypeInfo.from_type(tuple),
        }
    ))

    assert "collections.abc.Callable" == str(merged_types({
            t_id.hint2type(abc.Callable[[], None]),
            t_id.hint2type(abc.Callable[[int], None]),
            t_id.hint2type(abc.Callable),
        }
    ))


def test_merged_types_generics_never():
    assert "list[int]" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(typing.Never),)),
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
        }
    ))

    assert "dict[str, int]" == str(merged_types({
            TypeInfo.from_type(dict, args=(TypeInfo.from_type(typing.Never),TypeInfo.from_type(typing.Never),)),
            TypeInfo.from_type(dict, args=(TypeInfo.from_type(str),TypeInfo.from_type(int),)),
        }
    ))

    # tuple is covariant, so tuple[Never, ...] | tuple[int, ...] merges to
    # tuple[int, ...] (Never is subsumed, and tuple[int, ...] includes empty tuples)
    assert "tuple[int, ...]" == str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(typing.Never), ...)),
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int), ...))
        }
    ))

    # Empty tuple is subsumed by variable-length tuple (zero or more ints)
    assert "tuple[int, ...]" == str(merged_types({
            TypeInfo.from_type(tuple, args=((),)),
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int), ...))
        }
    ))


def test_merged_types_superclass():
    class A: pass
    class B(A): pass
    class C(B): pass
    class D(B): pass

    assert f"{name(B)}" == str(merged_types({
            TypeInfo.from_type(C),
            TypeInfo.from_type(D)
        }
    ))

    assert f"{name(B)}" == str(merged_types({
            TypeInfo.from_type(B),
            TypeInfo.from_type(D)
        }
    ))

    assert f"{name(A)}" == str(merged_types({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D)
        }
    ))

    assert f"int|{name(A)}" == str(merged_types({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D),
            get_type_name(int),
        }
    ))

    assert f"foo.bar|{name(A)}" == str(merged_types({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D),
            TypeInfo("foo", "bar")
        }
    ))

    assert f"list[int]|{name(A)}" == str(merged_types({
            TypeInfo.from_type(A),
            TypeInfo.from_type(D),
            rt_get_value_type([1]),
            rt_get_value_type([])
        }
    ))


def name(t: type):
    return f"{t.__module__}.{t.__qualname__}"


def test_merged_types_superclass_checks_attributes():
    class A: pass
    class B(A):
        def foo(self): pass
    class C(B):
        def _shouldnt_matter(self): pass
        def bar(self): pass
    class D(B):
        def bar(self): pass
    class E(B):
        def _shouldnt_matter(self): pass

    assert f"{name(C)}|{name(D)}" == str(merged_types({
            TypeInfo.from_type(C),
            TypeInfo.from_type(D)
        }
    ))

    assert f"{name(B)}" == str(merged_types({
            TypeInfo.from_type(C),
            TypeInfo.from_type(E)
        }
    ))


def test_merged_types_superclass_dunder_matters():
    class A: pass
    class B(A):
        pass
    class C(B):
        def __foo__(self): pass
    class D(B):
        def __foo__(self): pass

    assert f"{name(C)}|{name(D)}" == str(merged_types({
            TypeInfo.from_type(C),
            TypeInfo.from_type(D)
        }
    ))


def test_merged_types_superclass_bare_type():
    # invoking type.mro() raises an exception
    assert "int|type" == str(merged_types({
            get_type_name(int),
            get_type_name(type)
        }
    ))


def test_merged_types_superclass_multiple_superclasses():
    class A: pass
    class B(A):
        def foo(self): pass
    class C(B):
        pass

    class D: pass
    class E(D):
        def foo(self): pass
    class F(E):
        pass

    assert f"{name(B)}|{name(E)}" == str(merged_types({
            TypeInfo.from_type(B),
            TypeInfo.from_type(C),
            TypeInfo.from_type(E),
            TypeInfo.from_type(F),
        }
    ))


str_ti = TypeInfo.from_type(str)
int_ti = TypeInfo.from_type(int)


def test_hint2type_typevar():
    t = t_id.hint2type(T)
    assert t == TypeInfo(module=T.__module__, name=T.__name__)

    t = t_id.hint2type(List[T])   # type: ignore[valid-type]
    assert t == TypeInfo.from_type(list, args=(
            TypeInfo(module=T.__module__, name=T.__name__),
        )
    )


def test_hint2type_none():
    t = t_id.hint2type(None)
    assert t is NoneTypeInfo

    t = t_id.hint2type(abc.Generator[int|str, None, None])
    assert t == TypeInfo.from_type(abc.Generator, args=(
        TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(str),
        }),
        NoneTypeInfo,
        NoneTypeInfo
    ))


def test_hint2type_ellipsis():
    t = t_id.hint2type(tuple[str, ...])
    assert t == TypeInfo.from_type(tuple, module="", args=(
        TypeInfo.from_type(str, module=""),
        ...
    ))


def test_hint2type_list():
    t = t_id.hint2type(abc.Callable[[], None]) 
    assert t == TypeInfo.from_type(cast(type, abc.Callable), args=(
        TypeInfo.list([]),
        NoneTypeInfo
    ))


def test_hint2type_literal():
    def foo(
        x: Literal[10, 20],
        y: Literal[True, "lies"],
    ): pass

    hints = get_type_hints(foo)
    assert "int" == str(t_id.hint2type(hints['x']))
    assert "bool|str" == str(t_id.hint2type(hints['y']))


def test_hint2type_unions():
    t = t_id.hint2type(Union[int, str])
    assert t.is_union()
    assert t.args == (
        TypeInfo.from_type(int),
        TypeInfo.from_type(str),
    )
    assert str(t) == "int|str"

    t = t_id.hint2type(Optional[str])
    assert t.is_union()
    assert t.args == (
        TypeInfo.from_type(str),
        NoneTypeInfo
    )
    assert str(t) == "str|None"

    t = t_id.hint2type(str|int)
    assert t.is_union()
    assert t.args == (
        TypeInfo.from_type(int),
        TypeInfo.from_type(str),
    )
    assert str(t) == "int|str"


@pytest.mark.skipif((importlib.util.find_spec('numpy') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_hint2type_jaxtyping():
    import jaxtyping
    import numpy

    t = t_id.hint2type(jaxtyping.Float64[numpy.ndarray, "0"])
    assert t == TypeInfo("jaxtyping", "Float64", args=(
        TypeInfo.from_type(numpy.ndarray),
        "0"
    ))

    assert str(t) == "jaxtyping.Float64[numpy.ndarray, \"0\"]"


def test_from_set_with_unions():
    t = TypeInfo.from_set({
            TypeInfo.from_set({
                TypeInfo.from_type(str),
                TypeInfo.from_type(int)
            })
        })

    assert t.is_union()
    assert t.args == (
        TypeInfo.from_type(int),
        TypeInfo.from_type(str),
    )
    assert str(t) == "int|str"


def test_from_set_with_never():
    t = TypeInfo.from_set({
            TypeInfo.from_type(Never),
            TypeInfo.from_type(int),
            TypeInfo.from_type(str)
        })

    assert t.is_union()
    assert t.args == (
        TypeInfo.from_type(int),
        TypeInfo.from_type(str),
    )
    assert str(t) == "int|str"


def test_from_set_with_any():
    t = TypeInfo.from_set({
            TypeInfo.from_type(Any),
            TypeInfo.from_type(int),
            TypeInfo.from_type(str)
        })

    assert not t.is_union()
    assert t.fullname() == "typing.Any"
    assert t.args == ()


def test_merged_types_for_variable_simple():
    """Test that for_variable=True merges similar generics."""
    # Without for_variable, should keep separate
    assert "list[bool]|list[int]" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(list, args=(TypeInfo.from_type(bool),))
        }
    ))

    # With for_variable=True, should merge type arguments.
    # bool is a subtype of int, so simplify() further reduces to list[int]
    assert "list[int]" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(list, args=(TypeInfo.from_type(bool),))
        },
        for_variable=True
    ))


def test_merged_types_for_variable_with_none():
    """Test for_variable with None in the union."""
    # list[int] | list[bool] | None -> list[int] | None (bool simplified to int)
    assert "list[int]|None" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(list, args=(TypeInfo.from_type(bool),)),
            TypeInfo.from_type(type(None))
        },
        for_variable=True
    ))


def test_merged_types_for_variable_dict():
    """Test for_variable with dict types."""
    # dict[str, int] | dict[str, float] -> dict[str, float] (int simplified to float via numeric tower)
    assert "dict[str, float]" == str(merged_types({
            TypeInfo.from_type(dict, args=(TypeInfo.from_type(str), TypeInfo.from_type(int))),
            TypeInfo.from_type(dict, args=(TypeInfo.from_type(str), TypeInfo.from_type(float)))
        },
        for_variable=True
    ))


def test_merged_types_for_variable_nested():
    """Test for_variable with nested generics."""
    # list[tuple[int, float]] | list[tuple[bool, float]] -> list[tuple[int, float]] (bool simplified to int)
    assert "list[tuple[int, float]]" == str(merged_types({
            TypeInfo.from_type(list, args=(
                TypeInfo.from_type(tuple, args=(
                    TypeInfo.from_type(bool),
                    TypeInfo.from_type(float),
                )),
            )),
            TypeInfo.from_type(list, args=(
                TypeInfo.from_type(tuple, args=(
                    TypeInfo.from_type(int),
                    TypeInfo.from_type(float),
                )),
            )),
        },
        for_variable=True
    ))


def test_merged_types_for_variable_different_containers():
    """Test that different container types are not merged."""
    # list[int] | set[int] should stay separate even with for_variable=True
    assert "list[int]|set[int]" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(set, args=(TypeInfo.from_type(int),))
        },
        for_variable=True
    ))


def test_merged_types_for_variable_different_arity():
    """Test that generics with different number of args are not merged."""
    # tuple[int] | tuple[int, str] should stay separate
    result = str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int),)),
            TypeInfo.from_type(tuple, args=(TypeInfo.from_type(int), TypeInfo.from_type(str)))
        },
        for_variable=True
    ))
    assert "tuple[int, str]" in result
    assert "tuple[int]" in result


# =============================================================================
# Container sampling tests
# =============================================================================

from collections import Counter


def test_container_cache_same_object():
    """Cache returns same entry for same container object."""
    from righttyper.type_id import ContainerTypeCache

    cache = ContainerTypeCache(100)
    data = [1, 2, 3]

    entry1 = cache.get(data, 1)
    entry2 = cache.get(data, 1)
    assert entry1 is entry2


def test_container_cache_different_objects():
    """Cache returns different entries for different objects."""
    from righttyper.type_id import ContainerTypeCache

    cache = ContainerTypeCache(100)
    data1 = [1, 2, 3]
    data2 = [1, 2, 3]  # Same content, different object

    entry1 = cache.get(data1, 1)
    entry2 = cache.get(data2, 1)
    assert entry1 is not entry2


def test_container_cache_lru_eviction():
    """Cache evicts least recently used entries."""
    from righttyper.type_id import ContainerTypeCache

    cache = ContainerTypeCache(2)  # Small capacity
    data1, data2, data3 = [1], [2], [3]

    cache.get(data1, 1)
    cache.get(data2, 1)
    cache.get(data3, 1)  # Should evict data1

    assert len(cache._cache) == 2


def test_large_container_is_sampled():
    """Containers larger than the small threshold are sampled, not fully scanned."""
    from righttyper.type_id import _cache

    _cache._cache.clear()

    # Create large list
    data = list(range(1000))
    assert len(data) > run_options.container_small_threshold  # Test precondition
    t = get_value_type(data)

    assert 'int' in t
    # Verify we didn't scan all 1000 elements
    entry = _cache.get(data, 1)
    assert entry.all_samples[0].total() <= run_options.container_max_samples


def test_small_container_is_fully_scanned():
    """Containers at or below the small threshold are fully scanned."""
    from righttyper.type_id import _cache

    _cache._cache.clear()

    # Small list
    data = [1, 'a', 2.0]
    assert len(data) <= run_options.container_small_threshold  # Test precondition
    t = get_value_type(data)

    # Should see all types
    assert 'int' in t
    assert 'str' in t
    assert 'float' in t


def test_dict_samples_keys_and_values():
    """Dict sampling tracks both key and value types."""
    from righttyper.type_id import _cache

    _cache._cache.clear()

    data = {i: str(i) for i in range(100)}
    assert len(data) > run_options.container_small_threshold  # Test precondition
    t = get_value_type(data)

    assert t == 'dict[int, str]'


# =============================================================================
# Sliding window container sampling tests (new behavior)
# =============================================================================

from collections import deque


def test_container_sliding_window_detects_changes(monkeypatch):
    """Container that changes types should be resampled and include all types."""
    from righttyper.type_id import _cache

    monkeypatch.setattr(run_options, 'container_check_probability', 1.0)
    _cache._cache.clear()

    # First observation: list of ints (must be large enough to trigger sampling)
    data: list[Any] = list(range(100))
    assert len(data) > run_options.container_small_threshold  # Test precondition
    t1 = get_value_type(data)
    assert 'int' in t1

    # Mutate to strings
    data.clear()
    data.extend(['a'] * 100)

    # Second observation should detect the change and resample
    t2 = get_value_type(data)
    # Should include both int and str from full history
    assert 'int' in t2 and 'str' in t2


def test_sample_until_stable_uniform_types():
    """Sampling stops quickly when all samples have the same type."""
    from righttyper.type_id import ContainerSamples

    # Container with uniform type
    data = [42] * 100
    samples = ContainerSamples(o=data, n_counters=1)
    sampler = lambda: (data[0],)  # Always returns an int

    samples.sample_until_stable(sampler, depth=0)

    # Should have stopped near the minimum samples (all same type = low singleton ratio)
    total_samples = samples.all_samples[0].total()
    assert total_samples <= run_options.container_min_samples + 5


def test_sample_until_stable_diverse_types():
    """Sampling continues longer when types are highly diverse."""
    from righttyper.type_id import ContainerSamples

    # Create 30 distinct types dynamically - each sample will have a unique type
    distinct_types = [type(f'Type{i}', (), {}) for i in range(30)]
    distinct_values = [t() for t in distinct_types]
    idx = [0]
    samples = ContainerSamples(o=None, n_counters=1)

    def counting_sampler() -> tuple[Any, ...]:
        val = distinct_values[idx[0] % len(distinct_values)]
        idx[0] += 1
        return (val,)

    samples.sample_until_stable(counting_sampler, depth=0)

    # With 30 distinct types and min_samples=25, singleton ratio stays high
    # (each type seen only ~once), so sampling should continue beyond min_samples.
    total_samples = samples.all_samples[0].total()
    assert total_samples > run_options.container_min_samples


def test_container_full_history_preserved():
    """All types ever seen appear in annotation, even if evicted from window."""
    from righttyper.type_id import _cache

    _cache._cache.clear()

    data: list[Any] = list(range(100))
    assert len(data) > run_options.container_small_threshold  # Test precondition

    # First: ints
    get_value_type(data)

    # Mutate and observe many times to push ints out of window
    for _ in range(run_options.container_min_samples + 10):
        data.clear()
        data.extend(['a'] * 100)
        get_value_type(data)

    # Final type should still include int from history
    t = get_value_type(data)
    assert 'int' in t


# Hybrid sampling tests: Good-Turing for stopping, spot-check for change detection
# =============================================================================


def test_hybrid_spot_check_no_new_type_no_resample(monkeypatch):
    """When spot-check finds an existing type, no full resampling should occur."""
    from righttyper.type_id import _cache

    monkeypatch.setattr(run_options, 'container_check_probability', 1.0)
    _cache._cache.clear()

    # Create a uniform list (all ints) large enough to trigger sampling
    data = list(range(100))
    assert len(data) > run_options.container_small_threshold

    # First observation: triggers Good-Turing sampling until stable
    get_value_type(data)
    entry = _cache.get(data, 1)
    initial_sample_count = entry.all_samples[0].total()

    # Second observation: spot-check should find int (existing type)
    # With hybrid approach: should NOT trigger full resampling
    get_value_type(data)

    # Sample count should increase by at most 1 (the spot-check sample)
    new_sample_count = entry.all_samples[0].total()
    samples_added = new_sample_count - initial_sample_count
    assert samples_added <= 1, f"Expected at most 1 sample (spot-check), got {samples_added}"


def test_hybrid_spot_check_new_type_triggers_resample(monkeypatch):
    """When spot-check finds a new type, full Good-Turing resampling should occur."""
    from righttyper.type_id import _cache

    monkeypatch.setattr(run_options, 'container_check_probability', 1.0)
    _cache._cache.clear()

    # First observation with ints
    data: list[Any] = list(range(100))
    assert len(data) > run_options.container_small_threshold
    get_value_type(data)

    entry = _cache.get(data, 1)
    initial_sample_count = entry.all_samples[0].total()

    # Mutate to all strings - spot-check will find new type
    data.clear()
    data.extend(['a'] * 100)

    get_value_type(data)

    # Should have triggered full resampling (many more samples)
    new_sample_count = entry.all_samples[0].total()
    samples_added = new_sample_count - initial_sample_count
    assert samples_added > 1, f"Expected full resampling (>1 sample), got {samples_added}"


def test_hybrid_spot_check_sample_preserved(monkeypatch):
    """The spot-check sample that triggers resampling should be included in results."""
    from righttyper.type_id import _cache

    monkeypatch.setattr(run_options, 'container_check_probability', 1.0)
    _cache._cache.clear()

    # First: all ints
    data: list[Any] = list(range(100))
    assert len(data) > run_options.container_small_threshold
    t1 = get_value_type(data)
    assert 'int' in t1

    # Mutate to all strings
    data.clear()
    data.extend(['hello'] * 100)

    t2 = get_value_type(data)

    # Both types should be in the result (int from history, str from new sampling)
    assert 'int' in t2 and 'str' in t2
