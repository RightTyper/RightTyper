import inspect
import random
import re
import sys

import collections
from collections import Counter, OrderedDict
from dataclasses import dataclass
import collections.abc as abc
from abc import ABCMeta
from functools import cache
import itertools
from types import (
    CodeType,
    FrameType,
    FunctionType,
    MethodType,
    GeneratorType,
    AsyncGeneratorType,
    CoroutineType,
    GenericAlias,
    ModuleType,
    MappingProxyType,
    UnionType
)
import builtins
import types
from typing import Any, cast, get_type_hints, get_origin, get_args, TypeGuard
import typing

from righttyper.random_dict import RandomDict
from righttyper.typeinfo import TypeInfo, TypeInfoArg, NoneTypeInfo, AnyTypeInfo, UnknownTypeInfo
from righttyper.righttyper_types import Filename, FunctionName, CodeId
from righttyper.righttyper_utils import is_test_module, normalize_module_name
from righttyper.options import run_options, output_options
from righttyper.logger import logger


@cache
def get_jaxtyping() -> ModuleType|None:
    try:
        # we lazy load jaxtyping to avoid "PytestAssertRewriteWarning"s
        import jaxtyping
        return jaxtyping
    except ImportError:
        return None


def jx_dtype(value: Any) -> type|None:
    if jx := get_jaxtyping():
        t = type(value)
        for dtype in (
            jx.UInt4, jx.UInt8, jx.UInt16, jx.UInt32, jx.UInt64,
            jx.Int4, jx.Int8, jx.Int16, jx.Int32, jx.Int64,
            jx.BFloat16, jx.Float16, jx.Float32, jx.Float64,
            jx.Complex64, jx.Complex128,
            jx.Bool, jx.UInt, jx.Int, jx.Integer,
            jx.Float, jx.Complex, jx.Inexact, jx.Real,
            jx.Num, jx.Shaped, jx.Key
        ):
            if isinstance(value, dtype[t, "..."]):
                assert isinstance(dtype, type)
                return dtype

    return None


@cache
def get_numpy() -> ModuleType|None:
    try:
        import numpy
        return numpy
    except ImportError:
        return None


def hint2type(hint: object) -> TypeInfo:
    import typing

    def hint2type_arg(hint: object) -> TypeInfoArg:
        if isinstance(hint, list):
            return TypeInfo.list([hint2type_arg(el) for el in hint])

        if hint is Ellipsis or type(hint) is str or (isinstance(hint, tuple) and hint == ()):
            return hint

        return hint2type(hint)

    if (origin := get_origin(hint)):
        args = get_args(hint)

        if origin in (types.UnionType, typing.Union):
            return TypeInfo.from_set({hint2type(a) for a in args})

        # FIXME TypeInfo args can't hold non-str Literal values; for now,
        # convert to a union of their types
        if origin is typing.Literal:
            return TypeInfo.from_set({get_value_type(a) for a in args})

        # Somehow get_args(tuple[()]) yields no arguments, rather than the expected [()]
        if origin is tuple and not args:
            return TypeInfo.from_type(tuple, args=((),))

        return TypeInfo.from_type(origin, args=tuple(hint2type_arg(a) for a in args))


    if hint is None:
        return NoneTypeInfo

    if not hasattr(hint, "__module__"):
        return UnknownTypeInfo

    if (
        hint.__module__ == 'jaxtyping'
        and (array_type := getattr(hint, "array_type", None))
        and isinstance(name := getattr(hint, '__name__'), str)
        and (base_type := getattr(get_jaxtyping(), name.split('[')[0], None))
        and hasattr(hint, "dim_str")
    ):
        # jaxtyping arrays are really dynamic types formed by indexing on their
        # base type.  I am not quite sure whether type_obj should be base_type here:
        # it may yield incorrect results in generalize.simplify()
        return TypeInfo.from_type(base_type, args=(
            get_type_name(array_type), hint.dim_str
        ))

    if not hasattr(hint, "__qualname__"): # e.g., typing.TypeVar
        # The type object is likely (dynamic and) not a 'type'...
        return TypeInfo(normalize_module_name(hint.__module__), getattr(hint, "__name__"))

    return get_type_name(cast(type, hint))  # requires __module__ and __qualname__


@cache
def _type_for_callable(func: abc.Callable) -> TypeInfo:
    if (orig_func := unwrap(func)): # in case this is a wrapper
        func = orig_func

    args: "tuple[TypeInfo|str|ellipsis, ...]" = tuple()
    if not output_options.ignore_annotations and hasattr(func, "__annotations__"): # FIXME this should be a run option
        try:
            signature = inspect.signature(func)
            hints = get_type_hints(func)
        except (ValueError, NameError) as e:
            logger.info(f"Error getting type hints for {func} " +
                        f"({func.__annotations__}): {e}.\n")
            signature = None
            hints = None

        # any type hints?
        if signature and hints:
            arg_types = TypeInfo.list([
                hint2type(hints[arg_name]) if arg_name in hints else UnknownTypeInfo
                for arg_name in signature.parameters
            ])

            return_type = (
                hint2type(hints['return']) if 'return' in hints else UnknownTypeInfo
            )

            args = (arg_types, return_type)

    code = cast(CodeType|None, getattr(func, "__code__", None)) # native callables may lack this

    return TypeInfo.from_type(abc.Callable,
        args=args,
        code_id=CodeId.from_code(code) if code is not None else None,
        is_bound=isinstance(func, MethodType)
    )


def _is_function(f: object) -> TypeGuard[FunctionType|MethodType]:
    return type(f) in (FunctionType, MethodType)


def _retval_of(f: object) -> TypeInfo|None:
    """Returns a TypeInfo for the return value of the given object, if a function or method.
       If unknown (unannotated or ignoring annotations), the TypeInfo is linked to the object's
       code, so that it may be obtained from what it is observed to return.
    """

    if _is_function(f):
        if not output_options.ignore_annotations: # FIXME should be a run option
            try:
                if (retval_hint := get_type_hints(f).get('return')):
                    return hint2type(retval_hint)
            except:
                pass

        return UnknownTypeInfo.replace(code_id=CodeId.from_code(f.__code__))

    return None


# CPython 3.12 returns specialized objects for each one of the following iterators,
# but does not name their types publicly.  We here give them names to keep the
# introspection code more readable. Note that these types may not be exhaustive
# and may overlap as across Python versions and implementations.
# TODO add checks to verify that they appear unique
BYTES_ITER: type = type(iter(b''))
BYTEARRAY_ITER: type = type(iter(bytearray()))
DICT_KEYITER: type = type(iter({}))
DICT_VALUEITER: type = type(iter({}.values()))
DICT_ITEMITER: type = type(iter({}.items()))
LIST_ITER: type = type(iter([]))
LIST_REVERSED_ITER: type = type(iter(reversed([])))
RANGE_ITER: type = type(iter(range(1)))
LONGRANGE_ITER: type = type(iter(range(1 << 1000)))
SET_ITER: type = type(iter(set()))
STR_ITER: type = type(iter(""))
TUPLE_ITER: type = type(iter(()))

class _GetItemDummy:
    def __getitem__(self, n: Any) -> Any:
        return n
GETITEM_ITER: type = type(iter(_GetItemDummy()))


T = typing.TypeVar("T")

# Build map of well-known (mostly built-in) types to their names. Some builtin types
# are available from the "builtins" module, some from the "types" module, but others
# still, such as "list_iterator", aren't known by any particular name.
#
_BUILTINS: typing.Final[dict[type, TypeInfo]] = {
    # first get what we can from 'builtins'...
    t: TypeInfo('', n, type_obj=t)  # note we use '' as the module name
    for n, t in builtins.__dict__.items()
    if type(t) is type
} | {
    # then add types from 'types'...
    t: TypeInfo('types', n, type_obj=t)
    for n in types.__all__
    if type(t := types.__dict__.get(n)) is type
} | {
    # then add some overrides.

    # types.NoneType is annotated as 'None'
    types.NoneType: NoneTypeInfo,

    # These iterators are implemented natively, so we can't observe their execution
    # to infer their type arguments.
    RANGE_ITER:     TypeInfo.from_type(abc.Iterator, args=(TypeInfo.from_type(int),)),
    LONGRANGE_ITER: TypeInfo.from_type(abc.Iterator, args=(TypeInfo.from_type(int),)),
    BYTES_ITER:     TypeInfo.from_type(abc.Iterator, args=(TypeInfo.from_type(int),)),
    BYTEARRAY_ITER: TypeInfo.from_type(abc.Iterator, args=(TypeInfo.from_type(int),)),
    STR_ITER:       TypeInfo.from_type(abc.Iterator, args=(TypeInfo.from_type(str),)),

    # Special type constructs are often private types. Map them to 'type' to make them
    # easy to identify.
    type(typing.Generic[T]):         TypeInfo.from_type(type), # type: ignore[index]
    type(typing.Union[int, str]):    TypeInfo.from_type(type),
    type(typing.Callable[[], None]): TypeInfo.from_type(type),
    type(abc.Callable[[], None]):    TypeInfo.from_type(type),
    GenericAlias:                    TypeInfo.from_type(type),
    UnionType:                       TypeInfo.from_type(type),
}


class ABCFinder:
    """Finds the collections.abc protocol implemented by 't' that matches the most callable attributes."""

    # Reverse order in collections.abc to go from most specific to most general
    _ABCs: list[type] = list(reversed([
        obj
        for obj in vars(abc).values()
        if isinstance(obj, ABCMeta)
    ]))


    @cache
    @staticmethod
    def find_abc(t: type) -> type|None:
        matching = [
            g
            for g in ABCFinder._ABCs
            if issubclass(t, g)
        ]

        if not matching:
            return None

        def methods(o: object) -> set[str]:
            return {
                name
                for name in dir(o)
                if callable(getattr(o, name, None))
            }

        t_methods = methods(t)
        return max(matching, key=lambda it: len(methods(it) & t_methods))


def get_type_name(t: type, depth: int = 0) -> TypeInfo:
    """Returns a type's name as a TypeInfo."""

    if depth > 255:
        # We have likely fallen into an infinite recursion; fail gracefully
        logger.error(f"RightTyper failed to compute the type of {t}.")
        return UnknownTypeInfo

    if (ti := _BUILTINS.get(t)):
        return ti

    if t.__module__ == "builtins":
        # the type didn't have a name in "builtins" or "types" modules, so use protocol
        if (g := ABCFinder.find_abc(t)):
            return TypeInfo.from_type(g)

        return UnknownTypeInfo

    return TypeInfo.from_type(t)


def unwrap(method: abc.Callable|None) -> abc.Callable|None:
    """Follows a chain of `__wrapped__` attributes to find the original function."""

    # Remember objects by id to work around unhashable items, but point to object so
    # that the object can't go away (possibly reusing the id)
    visited = {}
    while hasattr(method, "__wrapped__"):
        if id(method) in visited: return None
        visited[id(method)] = method

        method = getattr(method, "__wrapped__")

    return method


@cache
def src2module(src: str) -> ModuleType|None:
    """Maps a module's source file to the module."""
    return next(
        (
            m
            for m in list(sys.modules.values()) # sys.modules may change during iteration
            if getattr(m, "__file__", None) == src
        ),
        None
    )


def find_function(
    caller_frame: FrameType,
    code: CodeType
) -> abc.Callable|None:
    """Attempts to map back from a code object to the function that uses it."""

    parts = code.co_qualname.split('.')

    def find_in(namespace: dict[str, Any]|MappingProxyType[str, Any], index: int=0) -> abc.Callable|None:
        if index < len(parts):
            name = parts[index]
            if (
                name.startswith("__")
                and not name.endswith("__")
                and index > 0
                and parts[index-1] != '<locals>'
            ):
                name = f"_{parts[index-1]}{name}"   # private method/attribute

            if obj := namespace.get(name):
                if (
                    # don't use isinstance(obj, Callable), as it relies on __class__, which may be overridden
                    (hasattr(obj, "__call__") or type(obj) is classmethod)
                    and (obj := unwrap(obj))
                    and getattr(obj, "__code__", None) is code
                ):
                    return obj

                if type(obj) is dict:
                    return find_in(obj, index+1)
                elif isinstance(obj, type):
                    return find_in(obj.__dict__, index+1)

        return None


    if '<locals>' in parts:
        # Python re-creates the function object dynamically with each invocation;
        # look for it on the stack.
        if caller_frame and caller_frame.f_back:
            after_locals = len(parts) - parts[::-1].index('<locals>')
            parts = parts[after_locals:]
            return find_in(caller_frame.f_back.f_locals)

    else:
        # look for it in its module
        if (m := src2module(code.co_filename)):
            return find_in(m.__dict__)

    return None


def _type_for_generator(
    type_obj: type,
    frame: FrameType,
    code: CodeType,
) -> TypeInfo:

    retval: TypeInfo|None = None
    # FIXME: using find_function here prevents us from using annotations from <locals> functions
    if not output_options.ignore_annotations and (f := find_function(frame, code)): # FIXME should be a run option
        try:
            if (retval_hint := get_type_hints(f).get('return')):
                retval = hint2type(retval_hint)
        except:
            pass

    if type_obj is abc.Coroutine:
        if not retval:
            retval = UnknownTypeInfo.replace(code_id=CodeId.from_code(code))

        return TypeInfo.from_type(
            type_obj,
            args=(
                NoneTypeInfo, NoneTypeInfo, retval
            )
        )

    return retval if retval else TypeInfo.from_type(type_obj, code_id=CodeId.from_code(code))


def _random_item[T](container: abc.Collection[T]) -> T:
    """Randomly samples from a container."""
    # Unbounded, islice's running time seems to be O(N);
    # options.container_sample_limit provides an optional bound
    limit = len(container)-1
    if run_options.container_sample_limit is not None:
        limit = min(limit, run_options.container_sample_limit)
    n = random.randint(0, limit)
    return next(itertools.islice(container, n, None))


def _first_referent(value: Any) -> object|None:
    """Returns the first object 'value' refers to, if any."""
    import gc
    ref = gc.get_referents(value)
    return ref[0] if len(ref) else None


def _needs_more_samples(counters: tuple[Counter[TypeInfo], ...]) -> bool:
    if (n := counters[0].total()) < run_options.container_min_samples:
        return True

    if n >= run_options.container_max_samples:
        return False

    # Good-Turing estimator based heuristic: if we're likely to see a new type
    # for any of the counters, take another sample
    if any(
        (sum(c == 1 for c in counter.values()) / n) > run_options.container_type_threshold
        for counter in counters
    ):
        return True

    return False


@dataclass
class Entry:
    o: object
    counters: tuple[Counter[TypeInfo], ...]

class ContainerTypeCache:
    """LRU cache of type information about a container."""
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._cache: OrderedDict[int, Entry] = OrderedDict()

    def get(self, o: object, n_counters: int) -> tuple[Counter[TypeInfo], ...]:
        o_id = id(o)    # accommodate non-hashable objects
        if not (e := self._cache.get(o_id, None)) or e.o is not o:
            e = Entry(o, tuple(Counter() for _ in range(n_counters)))
            self._cache[o_id] = e
            self._cache.move_to_end(o_id)
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
            return e.counters

        self._cache.move_to_end(o_id)
        return e.counters

_cache = ContainerTypeCache(1024)

def _get_container_args(
    container: Any,
    n_counters: typing.Literal[1,2],
    sampler: abc.Callable[[], tuple[TypeInfo, ...]],
    depth: int
) -> tuple[TypeInfo, ...]:

    counters = _cache.get(container, n_counters)

    if container:
        # In the first time, if it's a small container, just scan it
        if counters[0].total() == 0 and len(container) <= run_options.container_min_samples:
            if n_counters == 1:
                for v in container:
                    counters[0].update((get_value_type(v, depth+1),))
            else:
                for it in container.items():
                    for c, v in zip(counters, it):
                        c.update((get_value_type(v, depth+1),))
        else:
            # Take as many samples as G-T suggests we need, but at least one to
            # try and accommodate changing data
            while True:
                sample = sampler()
                for c, v in zip(counters, sample):
                    c.update((get_value_type(v, depth+1),))

                if not _needs_more_samples(counters):
                    break

    return tuple(TypeInfo.from_set(set(c)) for c in counters)


def _handle_tuple(value: Any, depth: int) -> TypeInfo:
    args: tuple[TypeInfoArg, ...]

    if value:
        args = tuple(get_value_type(fld, depth+1) for fld in value)
        if (
            run_options.generalize_tuples
            and len(args) >= run_options.generalize_tuples
            and all(a == args[0] for a in args[1:])
        ):
            args = (args[0], ...)
    else:
        args = ((),)
    return TypeInfo.from_type(tuple, args=args)


def _handle_dict(value: Any, depth: int) -> TypeInfo:
    sampler = lambda: ((el := _random_item(value)), value[el])
    args = _get_container_args(value, 2, sampler, depth)

    t: type = type(value)
    if (ti := _BUILTINS.get(t)):
        return ti.replace(args=args)

    return TypeInfo.from_type(t, args=args)


def _handle_randomdict(value: Any, depth: int) -> TypeInfo:
    sampler = lambda: ((el := value.random_item())[0], el[1])

    try:
        args = _get_container_args(value, 2, sampler, depth)
    except Exception as e:
        return TypeInfo.from_type(dict)
    else:
        return TypeInfo.from_type(dict, args=args)


def _handle_list(value: Any, depth: int) -> TypeInfo:
    sampler = lambda: (value[random.randint(0, len(value)-1)],) # this is O(1), much faster than islice()
    return TypeInfo.from_type(list, args=_get_container_args(value, 1, sampler, depth))


def _handle_set(value: Any, depth: int) -> TypeInfo:
    sampler = lambda: (_random_item(value),)
    return TypeInfo.from_type(type(value), args=_get_container_args(value, 1, sampler, depth))


def _handle_dict_keyiter(value: Any, depth: int) -> TypeInfo|None:
    if type(d := _first_referent(value)) is dict:
        return TypeInfo.from_type(abc.Iterator, args=(get_value_type(d, depth+1).args[0],))
    return None


def _handle_dict_valueiter(value: Any, depth: int) -> TypeInfo|None:
    if type(d := _first_referent(value)) is dict:
        return TypeInfo.from_type(abc.Iterator, args=(get_value_type(d, depth+1).args[1],))
    return None


def _handle_dict_itemiter(value: Any, depth: int) -> TypeInfo|None:
    if type(d := _first_referent(value)) is dict:
        return TypeInfo.from_type(abc.Iterator, args=(
                   TypeInfo.from_type(tuple, args=get_value_type(d, depth+1).args),)
               )
    return None


def _handle_list_iter(value: Any, depth: int) -> TypeInfo|None:
    if type(l := _first_referent(value)) is list:
        return TypeInfo.from_type(abc.Iterator, args=get_value_type(l, depth+1).args)
    return None


def _handle_set_iter(value: Any, depth: int) -> TypeInfo|None:
    if type(s := _first_referent(value)) is set:
        return TypeInfo.from_type(abc.Iterator, args=get_value_type(s, depth+1).args)
    return None


def _handle_tuple_iter(value: Any, depth: int) -> TypeInfo|None:
    if type(t := _first_referent(value)) is tuple:
        sampler = lambda: (t[random.randint(0, len(t)-1)],) # this is O(1), much faster than islice()
        return TypeInfo.from_type(abc.Iterator, args=_get_container_args(t, 1, sampler, depth))
    return None


def _handle_getitem_iter(value: Any, depth: int) -> TypeInfo|None:
    if (
        (obj := _first_referent(value)) is not None
        and (getitem := getattr(type(obj), "__getitem__", None)) is not None
    ):
        if (
            (obj_t := type(obj)).__module__ == 'numpy'
            and obj_t.__qualname__ == 'ndarray'
            and getattr(obj, "size", 0) > 0
        ):
            # Use __getitem__, as obj.dtype contains classes from numpy.dtypes;
            # check size, as using typing.Never for size=0 leads to mypy errors
            return TypeInfo.from_type(abc.Iterator, args=(get_value_type(getitem(obj, 0), depth+1),))
        
        if (retval := _retval_of(getitem)):
            return TypeInfo.from_type(abc.Iterator, args=(retval,))

        return TypeInfo.from_type(abc.Iterator)
    return None


class PostponedArg0:
    """Type used to postpone extracting the first (yield) argument of
       generators and iterators.  Their types may not be fully known
       until observed at runtime and merged using their code_id.
       Their code's annotated return value is Iterator[Y] or Generator[Y, S, R];
       we wrap these in a TypeInfo.from_type(PostponedArg0) to extract Y
       after resolution.
    """


def _handle_zip(value: Any, depth: int) -> TypeInfo|None:
    if (
        type(t := _first_referent(value)) is tuple
        and all(isinstance(s, abc.Iterator) for s in t)  # note a generator also IS-A abc.Iterator
    ):
        zip_sources = tuple(
            get_value_type(s, depth+1)
            for s in t
        )

        args = (
            TypeInfo.from_type(tuple, args=tuple(
                TypeInfo.from_type(PostponedArg0, args=(src,)) if any(t.code_id for t in src.walk()) else (
                    src.args[0] if src.args else UnknownTypeInfo
                )
                for src in zip_sources
            )),
        )
        return TypeInfo.from_type(abc.Iterator, args=args)
    return None


def _handle_enumerate(value: Any, depth: int) -> TypeInfo|None:
    if (
        (l := _first_referent(value)) is not None
        and isinstance(l, abc.Iterator)
    ):
        src = get_value_type(l, depth+1)
        return TypeInfo.from_type(enumerate,
            args=(TypeInfo.from_type(PostponedArg0, args=(src,)) if any(t.code_id for t in src.walk()) else (
                    src.args[0] if src.args else UnknownTypeInfo
                ),
            )
        )
    return None


# Build a map of well-known types to handlers that knows how to sample them, emitting
# a parametrized generic.
#
_type2handler: dict[type, abc.Callable[[Any, int], TypeInfo|None]] = {
    tuple: _handle_tuple,
    list: _handle_list,
    RandomDict: _handle_randomdict,
    dict: _handle_dict,
    collections.defaultdict: _handle_dict,
    collections.OrderedDict: _handle_dict,
    collections.ChainMap: _handle_dict,
    MappingProxyType: _handle_dict,
    set: _handle_set,
    frozenset: _handle_set,
    collections.Counter: _handle_set,
    # note that deque is-a Sequence, but its integer indexing is O(N)
    collections.deque: _handle_set,
    re.Pattern: lambda value, depth: TypeInfo.from_type(type(value), args=(get_value_type(value.pattern, depth+1),)),
    re.Match: lambda value, depth: TypeInfo.from_type(type(value), args=(get_value_type(value.group(), depth+1),)),
    DICT_KEYITER: _handle_dict_keyiter,
    DICT_VALUEITER: _handle_dict_valueiter,
    DICT_ITEMITER: _handle_dict_itemiter,
    LIST_ITER: _handle_list_iter,
    LIST_REVERSED_ITER: _handle_list_iter,
    SET_ITER: _handle_set_iter,
    TUPLE_ITER: _handle_tuple_iter,
    GETITEM_ITER: _handle_getitem_iter,
    zip: _handle_zip,
    enumerate: _handle_enumerate,

    type: lambda v, d: _BUILTINS[type] if v is type else _BUILTINS[type].replace(args=(get_type_name(v, d+1),)),

    FunctionType: lambda v, d: _type_for_callable(v),
    MethodType: lambda v, d: _type_for_callable(v),
    GeneratorType: lambda v, d: _type_for_generator(abc.Generator, v.gi_frame, v.gi_code),
    AsyncGeneratorType: lambda v, d: _type_for_generator(abc.AsyncGenerator, v.ag_frame, v.ag_code),
    CoroutineType: lambda v, d: _type_for_generator(abc.Coroutine, v.cr_frame, v.cr_code)
}


def _safe_getattr(obj: object, attr: str) -> Any|None:
    """Retrieves the given attribute statically, if possible.
       Using getattr or hasattr can lead to problems when __getattr__ is overridden;
       but even inspect.getattr_static may raise TypeError for objects that lack __mro__
       such as scipy.linalg.lapack.dpotrs (a fortran object)
    """

    try:
        return inspect.getattr_static(obj, attr, None)
    except:
        try:
            obj_name = str(obj)
        except:
            obj_name = "(object lacking __str__)"   # really?... just in case.

        logger.error(f"getattr_static({obj_name}, \'{attr}\', None) raised exception", exc_info=True)

    return None


def get_value_type(
    value: Any,
    depth: int = 0
) -> TypeInfo:
    """
    get_value_type takes a value (an instance) as input and returns a TypeInfo representing its type.

    If the value is a collection, it randomly selects an element (or key-value pair) and determines their types.
    If the value is a tuple, it determines the types of all elements in the tuple.
    """
    if depth > 255:
        # We have likely fallen into an infinite recursion; fail gracefully
        logger.error(f"RightTyper failed to compute the type of {value}.")
        return UnknownTypeInfo

    t: type = type(value)
    args: tuple[TypeInfo|str|ellipsis, ...]

    if (h := _type2handler.get(t)):
        if (ti := h(value, depth)) is not None:
            return ti

    if ti := _BUILTINS.get(t):
        return ti

    # Is this a spec-based mock?
    if (
        run_options.resolve_mocks
        and is_test_module(t.__module__)
        and (mock_spec := _safe_getattr(value, "_spec_class"))
    ):
        ti = get_type_name(mock_spec, depth+1)
        logger.debug(f"Resolved spec mock {t.__module__}.{t.__qualname__} -> {str(ti)}")
        return ti

    if (orig := _safe_getattr(value, "__orig_class__")):
        assert type(orig) in (GenericAlias, type(typing.Generic[T])), f"{orig=} {type(orig)=}" # type: ignore[index]
        return hint2type(orig)

    if isinstance(value, type) and value is not type:
        # isinstance() based search is needed, for example, for enum.EnumType
        return TypeInfo.from_type(type, args=(get_type_name(value, depth+1),))
    elif t.__module__ == "builtins":
        if t.__name__ == 'async_generator_wrapped_value':
            # workaround for https://github.com/python/cpython/issues/129013
            if (l := _first_referent(value)) is not None:
                return get_value_type(l, depth+1)

        # the type didn't have a name in "builtins" or "types" modules, so use protocol
        if (g := ABCFinder.find_abc(t)):
            if issubclass(g, (abc.Mapping, abc.ItemsView)):
                if value:
                    el = _random_item(value)
                    args = (get_value_type(el[0], depth+1), get_value_type(el[1], depth+1))
                else:
                    args = (TypeInfo.from_type(typing.Never), TypeInfo.from_type(typing.Never))
            elif issubclass(g, abc.Collection):
                if value:
                    el = _random_item(value)
                    args = (get_value_type(el, depth+1),)
                else:
                    args = (TypeInfo.from_type(typing.Never),)
# not usable for 'builtins': all objects are implemented in C, so lack __code__
#           elif issubclass(g, abc.Iterator):
#                if (retval := _retval_of(getattr(t, "__next__", None))):
#                    args = (retval,)
            else:
                args = ()

            return TypeInfo.from_type(g, args=args)

        return UnknownTypeInfo
    elif (
        t.__module__ == "functools"
        and isinstance(value, abc.Callable) # type: ignore[arg-type]
        and unwrap(value) is not value
    ):
        return _type_for_callable(value) # a function wrapper such as @cache

    if run_options.infer_shapes and hasattr(value, "dtype") and hasattr(value, "shape"):
        if (dtype := jx_dtype(value)) is not None:
            shape = " ".join(str(d) for d in value.shape)
            return TypeInfo.from_type(dtype, args=(get_type_name(t, depth+1), f"{shape}"))

    if t.__module__.startswith('numpy') and (numpy := get_numpy()):
        if t is numpy.ndarray:
            return TypeInfo.from_type(numpy.ndarray, args=(
                AnyTypeInfo,
                get_value_type(value.dtype, depth+1),
            ))
        elif issubclass(t, numpy.dtype):
            return TypeInfo.from_type(numpy.dtype, args=(
                get_type_name(value.type, depth+1) if value.type else UnknownTypeInfo,
            ))

    return get_type_name(t, depth+1)
