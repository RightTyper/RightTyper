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
    NoneType,
    FunctionType,
    MethodType,
    GeneratorType,
    AsyncGeneratorType,
    CoroutineType,
    GenericAlias,
    ModuleType,
    MappingProxyType
)
from typing import Any, Iterator, cast, TypeAlias, get_type_hints, get_origin, get_args
import typing
from pathlib import Path

from righttyper.random_dict import RandomDict
from righttyper.righttyper_types import (
    CodeId,
    Filename,
    FunctionName,
    FuncId,
    FuncContext,
    T,
    TypeInfo,
    NoneTypeInfo,
    AnyTypeInfo,
    UnknownTypeInfo
)
from righttyper.righttyper_types import types
from righttyper.righttyper_utils import skip_this_file, get_main_module_fqn


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

        return TypeInfo.from_type(
                    origin,
                    module=origin.__module__ if origin.__module__ != 'builtins' else '',
                    args=tuple(
                        hint2type_arg(a) for a in get_args(hint)
                    )
                )

    if type(hint) is NoneType:
        return NoneTypeInfo

    if (
        hint.__module__ == 'jaxtyping' and
        (array_type := getattr(hint, "array_type", None))
    ):
        return TypeInfo(hint.__module__, hint.__name__.split('[')[0], args=(
            get_type_name(array_type), hint.dim_str
        ))

    if not hasattr(hint, "__qualname__"): # e.g., typing.TypeVar
        return TypeInfo(name=hint.__name__, module=hint.__module__)

    return get_type_name(hint)


def type_from_annotations(func: abc.Callable) -> TypeInfo:
    try:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
    except (ValueError, NameError):
        signature = None
        hints = None

    args: tuple = tuple() # default to just "Callable"

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

    return TypeInfo("typing", "Callable",
        args=args,
        code_id=CodeId(id(func.__code__)),
        type_obj=cast(type, abc.Callable),
        is_bound=isinstance(func, MethodType)
    )


@cache
def should_skip_function(
    code: CodeType,
    script_dir: str,
    include_all: bool,
    include_files_pattern: str,
    include_functions_pattern: tuple[str, ...]
) -> bool:
    skip_file = skip_this_file(
        code.co_filename,
        script_dir,
        include_all,
        include_files_pattern,
    )
    included_in_pattern = include_functions_pattern and \
        all([not re.search(pattern, code.co_name) \
             for pattern in include_functions_pattern])
    if (
        skip_file
        or included_in_pattern
    ):
        return True
    if not (code.co_flags & 0x2):
        import dis

        assert dis.COMPILER_FLAG_NAMES[0x2] == "NEWLOCALS"
        return True
    return False


def find_caller_frame() -> FrameType|None:
    """Attempts to find the stack frame which from which we were called. A bit brittle!"""
    pkg_path = Path(find_caller_frame.__code__.co_filename).parent

    frame: FrameType|None = sys._getframe(1)

    while (frame is not None and frame.f_code is not None and
           pkg_path in Path(frame.f_code.co_filename).parents):
        frame = frame.f_back

    return frame


@cache
def from_types_import(t: type) -> str | None:
    # TODO we could also simply reverse types.__dict__ ...
    import types

    for k in types.__all__:
        if (v := types.__dict__.get(k)) and t is v:
            return k

    return None


@cache
def in_builtins_import(t: type) -> bool:
    import builtins

    if bt := builtins.__dict__.get(t.__name__):
        return t == bt

    return False


def normalize_module_name(module_name: str) -> str:
    """Applies common substitutions to a type's module's name."""
    if module_name == "__main__":
        # "__main__" isn't generally usable for typing, and only unique in this execution
        return get_main_module_fqn()

    if module_name == "builtins":
        return ""   # we consider these "well-known" and, for brevity, omit the module name

    return module_name


class TypeFinder:
    """Maps type objects to the module and qualified name under which they are exported."""

    def __init__(self):
        self._sys_modules_len = 0
        self._known_modules: set[str] = set()
        self._map: dict[type, tuple[list[str], list[str]]] = dict()
        self._private_map: dict[type, tuple[list[str], list[str]]] = dict()

    @staticmethod
    def _to_strings(r: tuple[list[str], list[str]]) -> tuple[str, str]:
        return ".".join(r[0]), ".".join(r[1])


    def find(self, t: type) -> tuple[str, str]|None:
        """Given a type object, return its module and qualified name as strings."""

        self._update_map()

        if t.__module__ == '__main__' and t not in self._map and t not in self._private_map:
            # TODO if running in __main__'s top-level code, the code may not yet have
            # reached the location where the type is defined.  We could handle this much
            # better by doing this mapping after execution and using __main__'s dictionary
            # returned by runpy.
            # TODO this is invoked after runpy is done running the module/script,
            # sys.modules['__main__'] may point back to RightTyper's main...
            # the same change above would resolve this as well.
            self._add_types_from(sys.modules['__main__'], ['__main__'], [], is_private=False)

        if (r := self._map.get(t)) or (r := self._private_map.get(t)):
            return self._to_strings(r)

        return None


    def _update_map(self) -> None:
        """Updates our internal map of types to modules and names."""
        # We detect changes in sys.modules by changes in its size,
        # assuming it monotonically increases in size.
        # TODO change to doing this mapping after execution, when
        # changes no longer affect us.
        if len(sys.modules) != self._sys_modules_len:
            sys_modules = list(sys.modules) # list() in case it changes while we're working
            for m in sys_modules:
                if (
                    m == '__main__'
                    or m in self._known_modules
                ):
                    continue

                self._add_types_from(
                    sys.modules[m],
                    m.split('.'), [],
                    is_private=(m.startswith("_") or "._" in m)
                )
                self._known_modules.add(m)

            self._sys_modules_len = len(sys_modules)


    def _add_types_from(
        self,
        target: type|ModuleType,
        mod_parts: list[str],
        name_parts: list[str],
        *,
        is_private: bool,
        objs_in_path: set = set()
    ) -> None:
        """Recursively explores a module (or type), updating self._map.
           'target': the object to explore
           'mod_parts': the module name, split on '.'
           'name_parts': the qualified name parts, split on '.'
           'objs_in_path': set of objects being recursed on, for loop avoidance
        """

        dunder_all = (
            set(da)
            if isinstance(target, ModuleType)
            and isinstance((da := target.__dict__.get('__all__')), (list, tuple))
            else None
        )

        for name, obj in target.__dict__.items():
            name_is_private = (
                is_private
                or (dunder_all is not None and name not in dunder_all)
                or name.startswith("_")
            )

            if (
                isinstance(obj, (type, ModuleType))
                # also include typing's special definitions; must be hashable to use as dict key
                or (target is typing and isinstance(obj, abc.Hashable) and hasattr(obj, "__name__"))
            ):
                # Some module objects are really namespaces, like "sys.monitoring"; they
                # don't show up in sys.modules. We want to process any such, but leave others
                # to be processed on their own from sys.modules
                if isinstance(obj, ModuleType) and obj.__name__ in sys.modules:
                    continue

                new_name_parts = name_parts + [name]

                if not isinstance(obj, ModuleType):
                    the_map = self._private_map if name_is_private else self._map
                    if (prev := the_map.get(cast(type, obj))):
                        prev_pkg = prev[0][0]
                        this_pkg = mod_parts[0]
                        prev_len = len(prev[0]) + len(prev[1])
                        this_len = len(mod_parts) + len(new_name_parts)

                    # When a module creates an alias for some other package's type,
                    # it must first import that dependency; that causes the dependency
                    # to show up first in sys.modules. Because of this, if a name already
                    # exists for a type pointing to another package, we don't override it.
                    # However, within a package, modules often load submodules before
                    # creating aliases that give types their "official" names. Within the
                    # same package, we pick the shortest, as shorter names are easier to
                    # use, and thus more likely to be the intended name.
                    # We assume that the package is given by the module name's first
                    # dot-delimited part, which isn't always true (e.g., namespace packages).
                    # TODO figure out package based on __file__ and/or __path__
                    if (
                        not prev
                        or (
                            prev_pkg == this_pkg and (
                                prev_len > this_len
                                or (prev_len == this_len and new_name_parts[-1] == obj.__name__)
                            )
                        )
                    ):
                        the_map[cast(type, obj)] = (mod_parts, new_name_parts)

                if isinstance(obj, (type, ModuleType)) and obj not in objs_in_path:
                    self._add_types_from(
                        obj,
                        mod_parts,
                        new_name_parts,
                        is_private=name_is_private,
                        objs_in_path=objs_in_path | {obj}
                    )


type_finder = TypeFinder()


@cache
def search_type(t: type) -> tuple[str, str] | None:
    """Searches for a given type in its __module__ and any submodules,
       returning the module and qualified name under which it exists, if any.
    """

    # Look up type in map
    if (f := type_finder.find(t)):
        return normalize_module_name(f[0]), f[1]

    # Just trust local scope names... can we do better?
    if '<locals>' in t.__qualname__:
        return normalize_module_name(t.__module__), t.__qualname__

    return None

# CPython 3.12 returns specialized objects for each one of the following iterators,
# but does not name their types publicly.  We here give them names to keep the
# introspection code more readable. Note that these types may not be exhaustive
# and may overlap as across Python versions and implementations.
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


def get_type_name(obj: type, depth: int = 0) -> TypeInfo:
    """Returns a type's name as a TypeInfo."""

    if depth > 255:
        # We have likely fallen into an infinite recursion; fail gracefully
        print(f"Warning: RightTyper failed to compute the type of {obj}.")
        return UnknownTypeInfo

    # Some builtin types are available from the "builtins" module,
    # some from the "types" module, but others still, such as
    # "list_iterator", aren't known by any particular name.
    if obj.__module__ == "builtins":
        if obj is NoneType:
            return NoneTypeInfo
        elif obj is zip:
            return TypeInfo("typing", "Iterator")
        elif in_builtins_import(obj):
            return TypeInfo("", obj.__name__, type_obj=obj) # these are "well known", so no module name needed
        elif (name := from_types_import(obj)):
            return TypeInfo("types", name, type_obj=obj)
        elif obj in (RANGE_ITER, LONGRANGE_ITER, BYTES_ITER, BYTEARRAY_ITER):
            return TypeInfo("typing", "Iterator", args=(TypeInfo.from_type(int, module=""),))
        elif obj is STR_ITER:
            return TypeInfo("typing", "Iterator", args=(TypeInfo.from_type(str, module=""),))
        elif issubclass(obj, abc.Iterator):
            return TypeInfo("typing", "Iterator")
        else:
            return UnknownTypeInfo

    # Certain dtype types' __qualname__ doesn't include a fully qualified name of their inner type
    if obj.__module__ == 'numpy' and 'dtype[' in obj.__name__ and hasattr(obj, "type"):
        t_name = obj.__qualname__.split('[')[0]
        return TypeInfo(obj.__module__, t_name, args=(get_type_name(obj.type, depth+1),))

    if (module_and_name := search_type(obj)):
        return TypeInfo(*module_and_name, type_obj=obj)

    return UnknownTypeInfo


def get_overrides(func: FuncContext) -> Iterator[FunctionType]:
    """Returns each method overridden by the given method.
    
    Args:
        func: A `FunctionType` instance.
    Yields:
        `FunctionType` instances for each method definition overridden by `func`.
    """
    yield func.function_object
    if func.class_object:
        for ancestor in func.class_object.__mro__:
            super_func = getattr(ancestor, func.function_object.__name__, None)
            if super_func: super_func = unwrap(super_func)
            if inspect.isfunction(super_func): yield super_func


def get_override_contexts(func: FuncContext) -> Iterator[CodeType]:
    """Find each code instance overridden by the given method.
    
    Args:
        func: An optional `FunctionType` instance.
        code: An optional backup `CodeType` instance.
    Yields:
        `CodeType` instances for each method definition overridden by `func`.
        If `func` is `None`, then code is yielded.
    """
    for override in get_overrides(func):
        yield override.__code__


def _is_instance(obj: object, types: tuple[type, ...]) -> type|None:
    """Like isinstance(), but returns the type matched."""
    for t in types:
        if isinstance(obj, t):
            return t

    return None


def unwrap(method: FunctionType|classmethod|None) -> FunctionType|None:
    """Follows a chain of `__wrapped__` attributes to find the original function."""

    visited = set()         # there shouldn't be a loop, but just in case...
    while hasattr(method, "__wrapped__"):
        if method in visited:
            return None
        visited.add(method)

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
) -> FuncContext | None:
    """Attempts to map back from a code object to the function that uses it."""

    parts = code.co_qualname.split('.')

    def find_in(namespace: dict|MappingProxyType, index: int=0) -> FuncContext | None:
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
                    return FuncContext(obj, None)

                if type(obj) is dict:
                    return find_in(obj, index+1)
                elif isinstance(obj, type):
                    return find_in(obj.__dict__, index+1)

        # print("==========")
        # print(code.co_qualname)
        # print(code.co_filename)
        # sys.stdout.buffer.write(code.co_code)
        return None


    if '<locals>' in parts:
        # Python re-creates the function object dynamically with each invocation;
        # look for it on the stack.
        if caller_frame.f_back:
            after_locals = len(parts) - parts[::-1].index('<locals>')
            parts = parts[after_locals:]
            return find_in(caller_frame.f_back.f_locals)

    else:
        # look for it in its module
        if (m := src2module(code.co_filename)):
            return find_in(m.__dict__)

    # print("----------")
    # print(code)
    return None


class PostponedIteratorArg:
    """Type used to postpone evaluating generator-based iterators"""


def get_value_type(
    value: Any,
    *,
    container_sample_limit: int,
    use_jaxtyping: bool,
    depth: int = 0
) -> TypeInfo:
    """
    get_value_type takes a value (an instance) as input and returns a string representing its type.

    If the value is a collection, it randomly selects an element (or key-value pair) and determines their types.
    If the value is a tuple, it determines the types of all elements in the tuple.

    For other types, it returns the name of the type.
    """
    if depth > 255:
        # We have likely fallen into an infinite recursion; fail gracefully
        print(f"Warning: RightTyper failed to compute the type of {value}.")
        return UnknownTypeInfo

    t: type|None
    args: tuple[TypeInfo|str|ellipsis, ...]


    def type_for_generator(
        obj: GeneratorType|AsyncGeneratorType|CoroutineType,
        type_obj: type,
        frame: FrameType,
        code: CodeType,
    ) -> TypeInfo:
        if (f := find_function(frame, code)):
            try:
                hints = get_type_hints(f.function_object)
                if 'return' in hints:
                    return hint2type(hints['return']).replace(code_id=CodeId(id(code)))
            except:
                pass

        return TypeInfo.from_type(type_obj, module="typing", code_id=CodeId(id(code)))


    def recurse(v: Any) -> TypeInfo:
        return get_value_type(
            v,
            use_jaxtyping=use_jaxtyping,
            container_sample_limit=container_sample_limit,
            depth=depth+1
        )


    def random_item[T](container: abc.Collection[T]) -> T:
        """Randomly samples from a container."""
        # Unbounded, islice's running time seems to be O(N); we arbitrarily bound to 1,000 items
        # to keep the overhead low (similar to list's O(1), in fact)
        n = random.randint(0, min(container_sample_limit, len(container)-1))
        return next(itertools.islice(container, n, None))


    def first_referent() -> object|None:
        """Returns the first object 'value' refers to, if any."""
        import gc
        ref = gc.get_referents(value)
        return ref[0] if len(ref) else None


    # using getattr or hasattr here can lead to problems when __getattr__ is overridden
    if (orig := inspect.getattr_static(value, "__orig_class__", None)):
        assert type(orig) is GenericAlias
        return hint2type(orig)

    t = type(value)
    if t is RandomDict:
        args = (TypeInfo("typing", "Never"), TypeInfo("typing", "Never"))
        try:
            if value:
                el = value.random_item()
                args = tuple(recurse(fld) for fld in el)
        except Exception:
            pass
        return TypeInfo.from_type(dict, module='', args=args)
    elif t in (
        dict, collections.defaultdict, collections.OrderedDict, collections.ChainMap,
        MappingProxyType
    ):
        if value:
            # it's more efficient to sample a key and then use it than to build .items()
            el = random_item(value)
            args = (recurse(el), recurse(value[el]))
        else:
            args = (TypeInfo("typing", "Never"), TypeInfo("typing", "Never"))

        if t is MappingProxyType:
            return TypeInfo(name='MappingProxyType', module='types', type_obj=t, args=args)
        return TypeInfo.from_type(t, t.__module__ if t.__module__ != 'builtins' else '', args=args)
    elif t is list:
        if value:
            el = value[random.randint(0, len(value)-1)] # this is O(1), much faster than islice()
            args = (recurse(el),)
        else:
            args = (TypeInfo("typing", "Never"),)
        return TypeInfo.from_type(t, module='', args=args)
    elif t in (set, frozenset, collections.Counter, collections.deque):
        # note that deque is-a Sequence, but its integer indexing is O(N)
        if value:
            el = random_item(value)
            args = (recurse(el),)
        else:
            args = (TypeInfo("typing", "Never"),)
        return TypeInfo.from_type(t, module='', args=args)
    elif t is tuple:
        if value:
            args = tuple(recurse(fld) for fld in value)
        else:
            args = tuple()  # FIXME this should yield "tuple[()]"
        return TypeInfo.from_type(t, module='', args=args)
    elif t is re.Pattern:
        return TypeInfo.from_type(t, args=(recurse(value.pattern),))
    elif t is re.Match:
        return TypeInfo.from_type(t, args=(recurse(value.group()),))
    elif isinstance(value, (FunctionType, MethodType)):
        return type_from_annotations(value)
    elif isinstance(value, GeneratorType):
        return type_for_generator(value, abc.Generator, value.gi_frame, value.gi_code)
    elif isinstance(value, AsyncGeneratorType):
        return type_for_generator(value, abc.AsyncGenerator, value.ag_frame, value.ag_code)
    elif isinstance(value, CoroutineType):
        return type_for_generator(value, abc.Coroutine, value.cr_frame, value.cr_code)
    elif isinstance(value, type) and value is not type:
        return TypeInfo("", "type", args=(get_type_name(value, depth+1),), type_obj=t)
    elif t.__module__ == "builtins":
        if t is NoneType:
            return NoneTypeInfo
        elif t in (RANGE_ITER, LONGRANGE_ITER, BYTES_ITER, BYTEARRAY_ITER):
            return TypeInfo("typing", "Iterator", args=(TypeInfo.from_type(int, module=""),))
        elif (t is DICT_KEYITER and (d := first_referent()) is not None and type(d) is dict):
            return TypeInfo("typing", "Iterator", args=(recurse(d).args[0],))
        elif (t is DICT_VALUEITER and (d := first_referent()) is not None and type(d) is dict):
            return TypeInfo("typing", "Iterator", args=(recurse(d).args[1],))
        elif (t is DICT_ITEMITER and (d := first_referent()) is not None and type(d) is dict):
            return TypeInfo("typing", "Iterator", args=(
                       TypeInfo.from_type(tuple, module="", args=recurse(d).args),)
                   )
        elif (
            t in (LIST_ITER, LIST_REVERSED_ITER)
            and (l := first_referent()) is not None
            and type(l) is list
        ):
            return TypeInfo("typing", "Iterator", args=recurse(l).args)
        elif (t is SET_ITER and (l := first_referent()) is not None and type(l) is set):
            return TypeInfo("typing", "Iterator", args=recurse(l).args)
        elif t is STR_ITER:
            return TypeInfo("typing", "Iterator", args=(TypeInfo.from_type(str, module=""),))
        elif (t is TUPLE_ITER and (l := first_referent()) is not None and type(l) is tuple):
            if l:
                el = l[random.randint(0, len(l)-1)] # this is O(1), much faster than islice()
                args = (recurse(el),)
            else:
                args = (TypeInfo("typing", "Never"),)
            return TypeInfo("typing", "Iterator", args=args)
        elif (
            t is GETITEM_ITER
            and (l := first_referent()) is not None
            and (getitem := getattr(type(l), "__getitem__", None)) is not None
        ):
            # If 'getitem' is a Python (non-native) function, we can intercept it;
            # add a postponed evaluation entry.
            if type(getitem) in (FunctionType, MethodType):
                src = recurse(getitem)
                assert src.type_obj is abc.Callable
                return TypeInfo("typing", "Iterator", args=(
                        TypeInfo.from_type(PostponedIteratorArg, args=(src,)),
                    )
                )

            if (l_t := type(l)).__module__ == 'numpy' and l_t.__qualname__ == 'ndarray' and getattr(l, "size", 0) > 0:
                # Use __getitem__, as l.dtype contains classes from numpy.dtypes;
                # check size, as using typing.Never for size=0 leads to mypy errors
                return TypeInfo("typing", "Iterator", args=(get_type_name(type(getitem(l, 0)), depth+1),))

            return TypeInfo("typing", "Iterator")
        elif (
            t is zip
            and (l := first_referent()) is not None
            and type(l) is tuple
            and all(isinstance(s, abc.Iterator) for s in l)  # note a generator also IS-A abc.Iterator
        ):
            zip_sources = tuple(recurse(s) for s in l)
            args = (
                # TODO it's unclear how to generate a typing.Iterator with 0 args, but happened for Emery
                TypeInfo.from_type(tuple, module="", args=tuple(
                    (src.args[0] if src.args else UnknownTypeInfo) if src.qualname() == "typing.Iterator"
                    else TypeInfo.from_type(PostponedIteratorArg, args=(src,))
                    for src in zip_sources
                )),
            )
            return TypeInfo("typing", "Iterator", args=args)
        elif (t is enumerate and (l := first_referent()) is not None and isinstance(l, abc.Iterator)):
            src = recurse(l)
            args = (
                # TODO it's unclear how to generate a typing.Iterator with 0 args, but happened for Emery
                ((src.args[0] if src.args else UnknownTypeInfo),) if src.qualname() == "typing.Iterator"
                else (TypeInfo.from_type(PostponedIteratorArg, args=(src,)),)
            )

            return TypeInfo.from_type(t, module="", args=args)
        elif in_builtins_import(cast(type, t)):
            return TypeInfo.from_type(t, module="") # these are "well known", so no module name needed
        elif (name := from_types_import(cast(type, t))):
            return TypeInfo(module="types", name=name, type_obj=t)
        elif (view := _is_instance(value, (abc.KeysView, abc.ValuesView))):
            # no name in "builtins" or "types" modules, so use abc protocol
            if value:
                el = random_item(value)
                args = (recurse(el),)
            else:
                args = (TypeInfo("typing", "Never"),)
            return TypeInfo("typing", view.__qualname__, args=args)
        elif isinstance(value, abc.ItemsView):
            # no name in "builtins" or "types" modules, so use abc protocol
            if value:
                el = random_item(value)
                args = (recurse(el[0]), recurse(el[1]))
            else:
                args = (TypeInfo("typing", "Never"), TypeInfo("typing", "Never"))
            return TypeInfo("typing", "ItemsView", args=args)
        elif t.__name__ == 'async_generator_wrapped_value':
            if (l := first_referent()) is not None:
                return recurse(l)

            return UnknownTypeInfo


    if use_jaxtyping and hasattr(value, "dtype") and hasattr(value, "shape"):
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
