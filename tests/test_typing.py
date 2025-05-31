from righttyper.righttyper_types import TypeInfo, NoneTypeInfo, AnyTypeInfo, PendingCallTrace, UnknownTypeInfo
from righttyper.typeinfo import merged_types, generalize
import righttyper.righttyper_runtime as rt
import collections.abc as abc
from collections import namedtuple
from typing import Any, Callable, get_type_hints, Union, Optional, TypeVar, List, Literal, cast, Self
import pytest
import importlib
import types
import righttyper.options as options

rt_get_value_type = rt.get_value_type

def get_value_type(v, **kwargs) -> str:
    return str(rt_get_value_type(v, **kwargs))

def type_from_annotations(*args, **kwargs) -> str:
    return str(rt.type_from_annotations(*args, **kwargs))


@pytest.fixture
def save_options():
    saved = options.options
    yield
    options.options = saved


class IterableClass(abc.Iterable):
    def __iter__(self):
        return None


class MyGeneric[A, B](dict): pass


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

    assert "typing.KeysView[str]" == get_value_type({'a':0, 'b':1}.keys())
    assert "typing.ValuesView[int]" == get_value_type({'a':0, 'b':1}.values())
    assert "typing.ItemsView[str, int]" == get_value_type({'a':0, 'b':1}.items())

    assert "typing.KeysView[typing.Never]" == get_value_type(dict().keys())
    assert "typing.ValuesView[typing.Never]" == get_value_type(dict().values())
    assert "typing.ItemsView[typing.Never, typing.Never]" == get_value_type(dict().items())

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
    assert "typing.Generator" == get_value_type(o)
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

    assert "typing.AsyncGenerator" == get_value_type(async_range(10))
    assert "typing.AsyncGenerator" == get_value_type(aiter(async_range(10)))

    assert f"{__name__}.MyGeneric[int, str]" == \
            get_value_type(MyGeneric[int, str]())


@pytest.mark.parametrize("init, name, nextv", [
    ["iter(b'0')", "typing.Iterator[int]", b'0'[0]],
    ["iter(bytearray(b'0'))", "typing.Iterator[int]", bytearray(b'0')[0]],
    ["iter({'a': 0})", "typing.Iterator[str]", 'a'],
    ["iter({'a': 0}.values())", "typing.Iterator[int]", 0],
    ["iter({'a': 0}.items())", "typing.Iterator[tuple[str, int]]", ('a', 0)],
    ["iter([0, 1])", "typing.Iterator[int]", 0],
    ["iter(reversed([0, 1]))", "typing.Iterator[int]", 1],
    ["iter(range(1))", "typing.Iterator[int]", 0],
    ["iter(range(1 << 1000))", "typing.Iterator[int]", 0],
    ["iter({'a'})", "typing.Iterator[str]", 'a'],
    ["iter('ab')", "typing.Iterator[str]", 'a'],
    ["iter(('a', 'b'))", "typing.Iterator[str]", 'a'],
    ["iter(tuple(c for c in ('a', 'b')))", "typing.Iterator[str]", 'a'],
    ["zip([0], ('a',))", "typing.Iterator[tuple[int, str]]", (0, 'a')],
    ["iter(zip([0], ('a',)))", "typing.Iterator[tuple[int, str]]", (0, 'a')],
    ["enumerate(('a', 'b'))", "enumerate[str]", (0, 'a')],
    ["iter(enumerate(('a', 'b')))", "enumerate[str]", (0, 'a')],
#    ["iter(zip([0], (c for c in ('a',))))", "typing.Iterator[tuple[int, str]]", (0, 'a')],
#    ["enumerate(c for c in ('a', 'b'))", "enumerate[str]", (0, 'a')],
])
def test_value_type_iterator(init, name, nextv):
    obj = eval(init)
    assert name == get_value_type(obj)
    assert nextv == next(obj), "changed state"


@pytest.mark.parametrize("init, name", [
    ["iter(b'0')", "typing.Iterator[int]"],
    ["iter(bytearray(b'0'))", "typing.Iterator[int]"],
    ["iter({'a': 0})", "typing.Iterator"],
    ["iter({'a': 0}.values())", "typing.Iterator"],
    ["iter({'a': 0}.items())", "typing.Iterator"],
    ["iter([0, 1])", "typing.Iterator"],
    ["iter(reversed([0, 1]))", "typing.Iterator"],
    ["iter(range(1))", "typing.Iterator[int]"],
    ["iter(range(1 << 1000))", "typing.Iterator[int]"],
    ["iter({'a'})", "typing.Iterator"],
    ["iter('ab')", "typing.Iterator[str]"],
    ["iter(('a', 'b'))", "typing.Iterator"],
    ["zip([0], ('a',))", "typing.Iterator"],
    ["iter(zip([0], ('a',)))", "typing.Iterator"],
    ["enumerate(('a', 'b'))", "enumerate"],
    ["iter(enumerate(('a', 'b')))", "enumerate"],
])
def test_type_name_iterator(init, name):
    assert name == str(rt.get_type_name(type(eval(init))))


def test_type_name_not_in_sys_modules():
    t = type('myType', (object,), dict())
    t.__module__ = 'doesntexist'

    assert rt.get_type_name(t) is UnknownTypeInfo


def test_items_from_typing():
    import typing

    def get_type_name(obj):
        # mypy ignore non-'type' asks for typing.*
        return rt.get_type_name(cast(type, obj))

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

def test_get_value_type_namedtuple_nonlocal():
    # namedtuple's __qualname__ also doesn't contain the enclosing class name...
    assert f"{__name__}.NamedTupleClass.P" == get_value_type(NamedTupleClass.P())


@pytest.mark.xfail(reason="How to best solve this?")
def test_get_value_type_namedtuple_local():
    # namedtuple's __qualname__ lacks context
    P = namedtuple('P', ['x', 'y'])
    assert f"{__name__}.test_get_value_type_namedtuple_local.<locals>.P" == get_value_type(P(1,1))


@pytest.mark.skipif((importlib.util.find_spec('numpy') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_numpy_jaxtyping(save_options):
    import numpy as np

    options.options.infer_shapes=True

    assert 'jaxtyping.Float64[numpy.ndarray, "0"]' == get_value_type(np.array([], np.float64))
    assert 'jaxtyping.Float16[numpy.ndarray, "1 1 1"]' == \
            get_value_type(np.array([[[1]]], np.float16))


@pytest.mark.skipif((importlib.util.find_spec('torch') is None or
                     importlib.util.find_spec('jaxtyping') is None),
                    reason='missing modules')
def test_get_value_type_torch_jaxtyping(save_options):
    import torch

    options.options.infer_shapes=True

    assert 'jaxtyping.Float64[torch.Tensor, "0"]' == \
            get_value_type(torch.tensor([], dtype=torch.float64))
    assert 'jaxtyping.Int32[torch.Tensor, "2 1"]' == \
            get_value_type(torch.tensor([[1],[2]], dtype=torch.int32))


def test_type_from_annotations():
    def foo(x: int|float, y: list[tuple[bool, ...]], z: Callable[[], None]) -> complex|None:
        pass

    assert "typing.Callable[[int|float, list[tuple[bool, ...]], collections.abc.Callable[[], None]], complex|None]" == \
            type_from_annotations(foo)


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

    assert f"int|{__name__}.MyGeneric[str, collections.abc.Callable[[], None]]" == str(rt.hint2type(hints['x']))
    assert f"tuple[int, ...]" == str(rt.hint2type(hints['y']))
    assert """jaxtyping.Float[jax.Array, "10 20"]""" == str(rt.hint2type(hints['z']))


def test_hint2type_pre_3_10():
    def foo(
        x: Optional[Union[int, str]],
        y: Optional[bool]
    ): pass

    hints = get_type_hints(foo)

    assert "int|str|None" == str(rt.hint2type(hints['x']))
    assert "bool|None" == str(rt.hint2type(hints['y']))


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

    assert str(t) == 'builtins.int'
    assert t.name == 'int'
    assert not t.args

    assert t is not TypeInfo.from_type(int) # should be new object
    assert t.type_obj is int

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            NoneTypeInfo
        })

    assert str(t) == 'builtins.int|None'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo.from_type(bool)
        })

    assert str(t) == 'builtins.bool|builtins.int'

    t = TypeInfo.from_set({
            TypeInfo.from_type(int),
            TypeInfo(module='', name='X', args=(
                TypeInfo.from_set({
                    TypeInfo.from_type(bool),
                    NoneTypeInfo
                }),
            ))
        })

    assert str(t) == 'X[builtins.bool|None]|builtins.int'

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
    assert "list[bool]|list[int]" == str(merged_types({
            TypeInfo("", "list", args=(TypeInfo("", "int"),)),
            TypeInfo("", "list", args=(TypeInfo("", "bool"),))
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
            rt.get_type_name(int),
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
        pass

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
            rt.get_type_name(int),
            rt.get_type_name(type)
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
bool_ti = TypeInfo("", "bool", type_obj=bool)
generator_ti = lambda *a: TypeInfo.from_type(abc.Generator, module="typing", args=tuple(a))
iterator_ti = lambda *a: TypeInfo("typing", "Iterator", tuple(a))
union_ti = lambda *a: TypeInfo("types", "UnionType", tuple(a), type_obj=types.UnionType)

def generate_sample(func: Callable, *args) -> PendingCallTrace:
    import righttyper.righttyper_runtime as rt

    res = func(*args)
    tr = PendingCallTrace(tuple(rt_get_value_type(arg) for arg in args))
    if type(res).__name__ == "generator":
        tr.is_generator = True
        try:
            while True:
                nex = next(res) # this can fail
                tr.yields.add(rt_get_value_type(nex))
        except StopIteration as e:
            if e.value is not None:
                tr.returns = rt_get_value_type(e.value)
    else:
        tr.returns = rt_get_value_type(res)

    return tr


def test_sample_process_simple():
    def dog(a):
        return a

    tr = generate_sample(dog, "hi")
    assert tr == PendingCallTrace((str_ti,), returns=str_ti)
    assert generalize([tr.process()]) == [str_ti, str_ti]


def test_sample_process_generator():
    def dog(a, b):
        yield a
        return b

    tr = generate_sample(dog, 1, "hi")
    assert tr == PendingCallTrace((int_ti, str_ti,), {int_ti}, returns=str_ti, is_generator=True)
    assert generalize([tr.process()]) == [int_ti, str_ti, generator_ti(int_ti, NoneTypeInfo, str_ti)]


def test_sample_process_generator_noyield():
    def dog(a, b):
        return b
        yield 

    tr = generate_sample(dog, 1, "hi")
    assert tr == PendingCallTrace((int_ti, str_ti,), returns=str_ti, is_generator=True)
    assert generalize([tr.process()]) == [int_ti, str_ti, generator_ti(NoneTypeInfo, NoneTypeInfo, str_ti)]


def test_sample_process_iterator_union():
    def dog(a, b):
        yield a
        yield b

    tr = generate_sample(dog, 1, "hi")
    assert tr == PendingCallTrace((int_ti, str_ti,), yields={int_ti, str_ti}, is_generator=True)
    assert generalize([tr.process()]) == [int_ti, str_ti, iterator_ti(union_ti(int_ti, str_ti))]


def test_sample_process_iterator():
    def dog(a):
        yield a

    tr = generate_sample(dog, "hi")
    assert tr == PendingCallTrace((str_ti,), yields={str_ti}, is_generator=True)
    assert generalize([tr.process()]) == [str_ti, iterator_ti((str_ti))]


def test_sample_process_generator_union():
    def dog(a, b, c):
        yield a
        yield b
        return c

    tr = generate_sample(dog, 1, "hi", True)
    assert tr == PendingCallTrace((int_ti, str_ti, bool_ti,), yields={int_ti, str_ti}, returns=bool_ti, is_generator=True)
    assert generalize([tr.process()]) == [int_ti, str_ti, bool_ti, generator_ti(union_ti(int_ti, str_ti), NoneTypeInfo, bool_ti)]


T = TypeVar("T")

def test_hint2type_typevar():
    t = rt.hint2type(T)
    assert t == TypeInfo(module=__name__, name='T')

    t = rt.hint2type(List[T])   # type: ignore[valid-type]
    assert t == TypeInfo.from_type(list, module='', args=(TypeInfo(module=__name__, name='T'),))


def test_hint2type_none():
    t = rt.hint2type(None)
    assert t is NoneTypeInfo

    t = rt.hint2type(abc.Generator[int|str, None, None])
    assert t == TypeInfo.from_type(abc.Generator, args=(
        TypeInfo.from_set({
            TypeInfo.from_type(int, module=''),
            TypeInfo.from_type(str, module=''),
        }),
        NoneTypeInfo,
        NoneTypeInfo
    ))


def test_hint2type_ellipsis():
    t = rt.hint2type(tuple[str, ...])
    assert t == TypeInfo.from_type(tuple, module="", args=(
        TypeInfo.from_type(str, module=""),
        ...
    ))


def test_hint2type_list():
    t = rt.hint2type(abc.Callable[[], None]) 
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
    assert "int" == str(rt.hint2type(hints['x']))
    assert "bool|str" == str(rt.hint2type(hints['y']))


def test_hint2type_unions():
    t = rt.hint2type(Union[int, str])
    assert t.qualname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(str, module=''),
    )

    t = rt.hint2type(Optional[str])
    assert t.qualname() == "types.UnionType"
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

    t = rt.hint2type(jaxtyping.Float64[numpy.ndarray, "0"])
    assert t == TypeInfo("jaxtyping", "Float64", args=(
        TypeInfo.from_type(numpy.ndarray),
        "0"
    ))

    assert str(t) == "jaxtyping.Float64[numpy.ndarray, \"0\"]"


def test_from_set_with_unions():
    t = merged_types({
            TypeInfo.from_set({
                TypeInfo.from_type(str, module=''),
                TypeInfo.from_set({
                    TypeInfo.from_type(int, module='')
                })
            }),
            TypeInfo.from_type(str, module='')
        })

    assert t.qualname() == "types.UnionType"
    assert t.args == (
        TypeInfo.from_type(int, module=''),
        TypeInfo.from_type(str, module=''),
    )

