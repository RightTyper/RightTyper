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
from righttyper.typemap import AdjustTypeNamesT

rt_get_value_type = t_id.get_value_type

# This plugin is needed to avoid caching results that depend upon 'run_options'
assert importlib.util.find_spec('pytest_antilru') is not None, "pytest-antilru missing"


def get_value_type(v, **kwargs) -> str:
    return str(rt_get_value_type(v, **kwargs))


def get_type_name(t) -> TypeInfo:
    transformer = AdjustTypeNamesT(sys.modules['__main__'].__dict__)
    ti = t_id.get_type_name(cast(type, t))
    if run_options.adjust_type_names:
        return transformer.visit(ti)
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

    assert "tuple" == get_value_type(tuple())
    assert "tuple[int, str]" == get_value_type((1, "foo"))
#    assert "tuple[()]" == get_value_type(tuple())  # FIXME
#    assert "tuple[int, ...]" == get_value_type((1, 2, 3, 4))

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
    ["iter(('a', 'b'))", "collections.abc.Iterator[str]", 'a'],
    ["iter(tuple(c for c in ('a', 'b')))", "collections.abc.Iterator[str]", 'a'],
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

    assert "collections.abc.Callable[[int|float, list[tuple[bool, ...]], collections.abc.Callable[[], None]], complex|None]" == \
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
        z: jaxtyping.Float[jax.Array, "10 20"]
    ): pass

    hints = get_type_hints(foo)

    assert f"int|{__name__}.MyGeneric[str, collections.abc.Callable[[], None]]" == str(t_id.hint2type(hints['x']))
    assert f"tuple[int, ...]" == str(t_id.hint2type(hints['y']))
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
    assert "int" == str(TypeInfo("", "int"))
    assert "tuple[bool]" == str(TypeInfo("", "tuple", args=(TypeInfo('', 'bool'),)))

    t = TypeInfo.from_type(type(None))
    assert t.module == ''
    assert t.name == 'None'
    assert t.type_obj is type(None)
    assert str(t) == "None"


def test_typeinfo_from_set():
    t = TypeInfo.from_set(set())
    assert t == NoneTypeInfo    # or should this be Never ?

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
    assert "list[bool]|list[int]" == str(merged_types({
            TypeInfo("", "list", type_obj=list, args=(TypeInfo("", "int"),)),
            TypeInfo("", "list", type_obj=list, args=(TypeInfo("", "bool"),))
        }
    ))

    assert "list" == str(merged_types({
            TypeInfo.from_type(list, args=(TypeInfo("", "int"),)),
            TypeInfo.from_type(list),
        }
    ))

    assert "tuple" == str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo("", "int"), ...)),
            TypeInfo.from_type(tuple),
        }
    ))

    assert "tuple" == str(merged_types({
            TypeInfo.from_type(tuple, args=(TypeInfo("", "int"), ...)),
            TypeInfo.from_type(tuple),
        }
    ))

    assert "collections.abc.Callable" == str(merged_types({
            t_id.hint2type(abc.Callable[[], None]),
            t_id.hint2type(abc.Callable[[int], None]),
            t_id.hint2type(abc.Callable),
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

    assert f"list[int]|list[typing.Never]|{name(A)}" == str(merged_types({
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


str_ti = TypeInfo("", "str", type_obj=str)
int_ti = TypeInfo("", "int", type_obj=int)
union_ti = lambda *a: TypeInfo("types", "UnionType", tuple(a), type_obj=types.UnionType)


def test_hint2type_typevar():
    t = t_id.hint2type(T)
    assert t == TypeInfo(module=T.__module__, name=T.__name__)

    t = t_id.hint2type(List[T])   # type: ignore[valid-type]
    assert t == TypeInfo.from_type(list, module='', args=(
            TypeInfo(module=T.__module__, name=T.__name__),
        )
    )


def test_hint2type_none():
    t = t_id.hint2type(None)
    assert t is NoneTypeInfo

    t = t_id.hint2type(abc.Generator[int|str, None, None])
    assert t == TypeInfo.from_type(abc.Generator, args=(
        TypeInfo.from_set({
            TypeInfo.from_type(int, module=''),
            TypeInfo.from_type(str, module=''),
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
    assert t.fullname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(str, module=''),
    )

    t = t_id.hint2type(Optional[str])
    assert t.fullname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(str, module=''),
        NoneTypeInfo
    )


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
                TypeInfo.from_type(str, module=''),
                TypeInfo.from_type(int, module='')
            })
        })

    assert t.fullname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(str, module=''),
    )


def test_from_set_with_never():
    t = TypeInfo.from_set({
            TypeInfo.from_type(Never),
            TypeInfo.from_type(int, module=''),
            TypeInfo.from_type(str, module='')
        })

    assert t.fullname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(str, module=''),
    )


def test_from_set_with_any():
    t = TypeInfo.from_set({
            TypeInfo.from_type(Any),
            TypeInfo.from_type(int, module=''),
            TypeInfo.from_type(str, module='')
        })

    assert t.fullname() == "typing.Any"
    assert t.args == ()


def test_uniontype():
    assert "int|str" == str(union_ti(int_ti, str_ti))
    assert "types.UnionType" == str(union_ti())
