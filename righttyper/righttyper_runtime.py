import inspect
import random
import re
import sys

import collections
import collections.abc as abc
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
from typing import Any, cast, get_type_hints, get_origin, get_args, Callable
import typing
from pathlib import Path

from righttyper.random_dict import RandomDict
from righttyper.typeinfo import TypeInfo, TypeInfoArg, NoneTypeInfo, AnyTypeInfo, UnknownTypeInfo
from righttyper.righttyper_types import Filename, FunctionName, CodeId
from righttyper.righttyper_utils import is_test_module, normalize_module_name
from righttyper.options import options
from righttyper.logger import logger


@cache
def get_jaxtyping():
    try:
        # we lazy load jaxtyping to avoid "PytestAssertRewriteWarning"s
        import jaxtyping
        return jaxtyping
    except ImportError:
        return None


def jx_dtype(value: Any) -> str|None:
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
                return dtype.__qualname__

    return None


def hint2type(hint) -> TypeInfo:
    import typing

    def hint2type_arg(hint):
        if isinstance(hint, list):
            return TypeInfo.list([hint2type_arg(el) for el in hint])

        if hint is Ellipsis or type(hint) is str:
            return hint

        return hint2type(hint)

    if (origin := get_origin(hint)):
        if origin is typing.Union:
            return TypeInfo.from_set({hint2type(a) for a in get_args(hint)})

        # FIXME TypeInfo args can't hold non-str Literal values; for now,
        # convert to a union of their types
        if origin is typing.Literal:
            return TypeInfo.from_set({get_value_type(a) for a in get_args(hint)})

        return TypeInfo.from_type(
                    origin,
                    module=normalize_module_name(origin.__module__),
                    args=tuple(
                        hint2type_arg(a) for a in get_args(hint)
                    )
                )

    if hint is None:
        return NoneTypeInfo

    if not hasattr(hint, "__module__"):
        return UnknownTypeInfo

    if (
        hint.__module__ == 'jaxtyping'
        and (array_type := getattr(hint, "array_type", None))
    ):
        return TypeInfo(hint.__module__, hint.__name__.split('[')[0], args=(
            get_type_name(array_type), hint.dim_str
        ))

    if not hasattr(hint, "__qualname__"): # e.g., typing.TypeVar
        return TypeInfo(normalize_module_name(hint.__module__), hint.__name__)

    return get_type_name(hint)  # requires __module__ and __qualname__


def _type_for_callable(func: abc.Callable) -> TypeInfo:
    if (orig_func := unwrap(func)): # in case this is a wrapper
        func = orig_func

    args: "tuple[TypeInfo|str|ellipsis, ...]" = tuple()
    if not options.ignore_annotations:
        try:
            signature = inspect.signature(func)
            hints = get_type_hints(func)
        except (ValueError, NameError):
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


def _is_defined_in(t: type, target: type|ModuleType, name_parts: list[str], name_index: int=0) -> bool:
    """Checks whether a name, given split into name_parts, is defined in a class or module."""
    if name_index<len(name_parts) and (obj := target.__dict__.get(name_parts[name_index])):
        if obj is t:
            return True

        if type(obj) in (type, ModuleType):
            return _is_defined_in(t, obj, name_parts, name_index+1)

    return False


@cache
def search_type(t: type) -> tuple[str, str] | None:
    """Searches for a given type in its __module__ and any submodules,
       returning the module and qualified name under which it exists, if any.
    """
    if not options.adjust_type_names:
        # If we we can, (i.e., not '<locals>', not from '__main__',) check the type is there
        # The problem with __main__ is that if runpy is done running the module/script,
        # sys.modules['__main__'] points back to RightTyper's __main__.
        if (
            t.__module__ != '__main__'
            and '<locals>' not in (name_parts := t.__qualname__.split('.'))
            and (
                not (m := sys.modules.get(t.__module__))
                or not _is_defined_in(t, m, name_parts)
            )
        ):
            return None

    return normalize_module_name(t.__module__), t.__qualname__


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
    def __getitem__(self, n):
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


def get_type_name(obj: type, depth: int = 0) -> TypeInfo:
    """Returns a type's name as a TypeInfo."""

    if depth > 255:
        # We have likely fallen into an infinite recursion; fail gracefully
        logger.error(f"RightTyper failed to compute the type of {obj}.")
        return UnknownTypeInfo

    if (ti := _BUILTINS.get(obj)):
        return ti

    if obj.__module__ == "builtins":
        # the type didn't have a name in "builtins" or "types" modules, so use protocol
        if issubclass(obj, abc.Iterator):
            return TypeInfo.from_type(abc.Iterator)
        else:
            return UnknownTypeInfo

    # Certain dtype types' __qualname__ doesn't include a fully qualified name of their inner type
    if obj.__module__ == 'numpy' and 'dtype[' in obj.__name__ and hasattr(obj, "type"):
        t_name = obj.__qualname__.split('[')[0]
        return TypeInfo(obj.__module__, t_name, args=(get_type_name(obj.type, depth+1),))

    if (module_and_name := search_type(obj)):
        return TypeInfo(*module_and_name, type_obj=obj)

    return UnknownTypeInfo


def _is_instance(obj: object, types: tuple[type, ...]) -> type|None:
    """Like isinstance(), but returns the type matched."""
    for t in types:
        if isinstance(obj, t):
            return t

    return None


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

    def find_in(namespace: dict|MappingProxyType, index: int=0) -> abc.Callable|None:
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
    obj: GeneratorType|AsyncGeneratorType|CoroutineType,
    type_obj: type,
    frame: FrameType,
    code: CodeType,
) -> TypeInfo:

    retval: TypeInfo|None = None
    # FIXME: using find_function here prevents us from using annotations from <locals> functions
    if not options.ignore_annotations and (f := find_function(frame, code)):
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
    if options.container_sample_limit is not None:
        limit = min(limit, options.container_sample_limit)
    n = random.randint(0, limit)
    return next(itertools.islice(container, n, None))


def _first_referent(value: Any) -> object|None:
    """Returns the first object 'value' refers to, if any."""
    import gc
    ref = gc.get_referents(value)
    return ref[0] if len(ref) else None


def _handle_tuple(value: Any, depth: int) -> TypeInfo:
    if value:
        args = tuple(get_value_type(fld, depth+1) for fld in value)
    else:
        args = tuple()  # FIXME this should yield "tuple[()]"
    return TypeInfo.from_type(tuple, args=args)


def _handle_dict(value: Any, depth: int) -> TypeInfo:
    t: type = type(value)
    if value:
        # it's more efficient to sample a key and then use it than to build .items()
        el = _random_item(value)
        args = (get_value_type(el, depth+1), get_value_type(value[el], depth+1))
    else:
        args = (TypeInfo.from_type(typing.Never), TypeInfo.from_type(typing.Never))

    if (ti := _BUILTINS.get(t)):
        return ti.replace(args=args)

    return TypeInfo.from_type(t, args=args)


def _handle_randomdict(value: Any, depth: int) -> TypeInfo:
    args: tuple[TypeInfo, ...] = ()
    try:
        if value:
            el = value.random_item()
            args = tuple(get_value_type(fld, depth+1) for fld in el)
        else:
            args = (TypeInfo.from_type(typing.Never), TypeInfo.from_type(typing.Never))
    except Exception:
        pass
    return TypeInfo.from_type(dict, args=args)


def _handle_list(value: Any, depth: int) -> TypeInfo:
    if value:
        el = value[random.randint(0, len(value)-1)] # this is O(1), much faster than islice()
        args = (get_value_type(el, depth+1),)
    else:
        args = (TypeInfo.from_type(typing.Never),)
    return TypeInfo.from_type(list, args=args)


def _handle_set(value: Any, depth: int) -> TypeInfo:
    t = type(value)
    # note that deque is-a Sequence, but its integer indexing is O(N)
    if value:
        el = _random_item(value)
        args = (get_value_type(el, depth+1),)
    else:
        args = (TypeInfo.from_type(typing.Never),)

    return TypeInfo.from_type(t, args=args)


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
        if t:
            el = t[random.randint(0, len(t)-1)] # this is O(1), much faster than islice()
            args = (get_value_type(el, depth+1),)
        else:
            args = (TypeInfo.from_type(typing.Never),)
        return TypeInfo.from_type(abc.Iterator, args=args)
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
            return TypeInfo.from_type(abc.Iterator, args=(get_type_name(type(getitem(obj, 0)), depth+1),))
        
        if type(getitem) in (FunctionType, MethodType): # get full type from runtime observations
            callable_type = _type_for_callable(getitem)
            retval: TypeInfoArg|None = callable_type.args[-1] if callable_type.args else None

            if not retval:
                retval = UnknownTypeInfo.replace(code_id=CodeId.from_code(getitem.__code__))

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
_type2handler: dict[type, Callable[[Any, int], TypeInfo|None]] = {
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
    GeneratorType: lambda v, d: _type_for_generator(v, abc.Generator, v.gi_frame, v.gi_code),
    AsyncGeneratorType: lambda v, d: _type_for_generator(v, abc.AsyncGenerator, v.ag_frame, v.ag_code),
    CoroutineType: lambda v, d: _type_for_generator(v, abc.Coroutine, v.cr_frame, v.cr_code)
}


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
    if mock_spec := inspect.getattr_static(value, "_spec_class", None):
        if options.resolve_mocks and is_test_module(t.__module__):
            ti = get_type_name(mock_spec, depth+1)
            logger.debug(f"Resolved spec mock {t.__module__}.{t.__qualname__} -> {str(ti)}")
            return ti

    # using getattr or hasattr here can lead to problems when __getattr__ is overridden
    if (orig := inspect.getattr_static(value, "__orig_class__", None)):
        assert type(orig) in (GenericAlias, type(typing.Generic[T])), f"{orig=} {type(orig)=}" # type: ignore[index]
        return hint2type(orig)

    if isinstance(value, type) and value is not type:
        # isinstance() based search is needed, for example, for enum.EnumType
        return TypeInfo.from_type(type, args=(get_type_name(value, depth+1),))
    elif t.__module__ == "builtins":
        # the type didn't have a name in "builtins" or "types" modules, so use protocol
        if (view := _is_instance(value, (abc.KeysView, abc.ValuesView))):
            if value:
                el = _random_item(value)
                args = (get_value_type(el, depth+1),)
            else:
                args = (TypeInfo.from_type(typing.Never),)
            return TypeInfo.from_type(view, args=args)
        elif isinstance(value, abc.ItemsView):
            if value:
                el = _random_item(value)
                args = (get_value_type(el[0], depth+1), get_value_type(el[1], depth+1))
            else:
                args = (TypeInfo.from_type(typing.Never), TypeInfo.from_type(typing.Never))
            return TypeInfo.from_type(abc.ItemsView, args=args)
        elif t.__name__ == 'async_generator_wrapped_value':
            if (l := _first_referent(value)) is not None:
                return get_value_type(l, depth+1)

            return UnknownTypeInfo
    elif (
        t.__module__ == "functools"
        and isinstance(value, abc.Callable) # type: ignore[arg-type]
        and unwrap(value) is not value
    ):
        return _type_for_callable(value) # a function wrapper such as @cache

    if options.infer_shapes and hasattr(value, "dtype") and hasattr(value, "shape"):
        if (dtype := jx_dtype(value)) is not None:
            shape = " ".join(str(d) for d in value.shape)
            return TypeInfo("jaxtyping", dtype, args=(
                get_type_name(t, depth+1),
                f"{shape}"
            ))

    if t.__module__ == 'numpy' and t.__qualname__ == 'ndarray':
        return TypeInfo("numpy", "ndarray", args=(
            AnyTypeInfo,
            get_type_name(type(value.dtype), depth+1)
        ))

    return get_type_name(t, depth+1)
