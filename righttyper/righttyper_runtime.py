import inspect
import random
import re
import sys

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
    GenericAlias
)
from typing import Any, cast, TypeAlias
from pathlib import Path

from righttyper.random_dict import RandomDict
from righttyper.righttyper_types import (
    CodeId,
    Filename,
    FunctionName,
    FuncId,
    T,
    TypeInfo,
    NoneTypeInfo,
    AnyTypeInfo
)
from righttyper.righttyper_utils import skip_this_file, get_main_module_fqn


def sample_from_collection(value: abc.Collection[T]|abc.Iterator[T], depth = 0) -> T:
    """Samples from a collection, or from an interator/generator whose state we don't mind changing."""
    MAX_ELEMENTS = 10   # to keep this O(1)

    if isinstance(value, abc.Collection):
        n = random.randint(0, min(MAX_ELEMENTS, len(value) - 1))
        return next(itertools.islice(value, n, n + 1))

    n = random.randint(1, MAX_ELEMENTS)
    return list(itertools.islice(value, n))[-1]


JX_DTYPES = None
def jx_dtype(value: Any) -> str|None:
    global JX_DTYPES

    if JX_DTYPES is None:
        # we lazy load jaxtyping to avoid "PytestAssertRewriteWarning"s
        try:
            import jaxtyping as jx
            jx_dtype_type: TypeAlias = jx._array_types._MetaAbstractDtype
            JX_DTYPES = cast(list[jx_dtype_type], [
                jx.UInt4, jx.UInt8, jx.UInt16, jx.UInt32, jx.UInt64,
                jx.Int4, jx.Int8, jx.Int16, jx.Int32, jx.Int64,
                jx.BFloat16, jx.Float16, jx.Float32, jx.Float64,
                jx.Complex64, jx.Complex128,
                jx.Bool, jx.UInt, jx.Int, jx.Integer,
                jx.Float, jx.Complex, jx.Inexact, jx.Real,
                jx.Num, jx.Shaped, jx.Key,
            ])
        except ImportError:
            JX_DTYPES = []

    t = type(value)
    for dtype in JX_DTYPES:
        if isinstance(value, dtype[t, "..."]):
            return dtype.__qualname__
    return None


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


def type_from_annotations(func: abc.Callable) -> TypeInfo:
    try:
        signature = inspect.signature(func)
    except ValueError:
        signature = None

    args: tuple

    # Do we have an annotation?
    if (
        signature and (
            any(p.annotation is not p.empty for p in signature.parameters.values())
            or signature.return_annotation is not inspect.Signature.empty
        )
    ):
        # Extract argument types, default to Any if no annotation provided
        arg_types = [
            (param.annotation if param.annotation is not param.empty else Any)
            for name, param in signature.parameters.items()
        ]

        # Extract return type, default to Any if no annotation provided
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            return_type = Any

        def format_arg(arg) -> str:
            if isinstance(arg, str):
                # If types are quoted while using "from __future__ import annotations",
                # strings may appear double quoted
                if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ["'",'"']:
                    arg = arg[1:-1]

                return arg

            if isinstance(arg, GenericAlias):
                return str(arg)

            if isinstance(arg, type):
                return str(get_type_name(arg))

            return repr(arg)

        # Format the result
        args = (
            f"[{', '.join([format_arg(arg) for arg in arg_types])}]",
            format_arg(return_type)
        )
    else:
        # no annotation: default to just "Callable"
        args = tuple()

    # Construct the Callable type string
    return TypeInfo("typing", "Callable",
        args=args,
        code_id=CodeId(id(func.__code__)),
        is_bound=isinstance(func, MethodType)
    )


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


@cache
def lookup_type_module(t: type) -> str:
    parts = t.__qualname__.split('.')

    def is_defined_in_module(namespace: dict, index: int=0) -> bool:
        if index<len(parts) and (obj := namespace.get(parts[index])):
            if obj is t:
                return True

            if isinstance(obj, dict):
                return is_defined_in_module(obj, index+1)

        return False

    # Is it defined where it claims to be?
    if (m := sys.modules.get(t.__module__)):
        if is_defined_in_module(m.__dict__):
            return normalize_module_name(t.__module__)

    # Can we find it some submodule?
    # FIXME this search could be more exhaustive and/or more principled
    module_prefix = f"{t.__module__}."
    for name, mod in sys.modules.items():
        if name.startswith(module_prefix) and is_defined_in_module(mod.__dict__):
            return normalize_module_name(name)

    # Keep it as a last resort, to facilitate diagnostics
    return normalize_module_name(t.__module__)


RANGE_ITER_TYPE = type(iter(range(1)))

def get_type_name(obj: type, depth: int = 0) -> TypeInfo:
    """Returns a type's name as a TypeInfo."""

    if depth > 255:
        # We have likely fallen into an infinite recursion.
        # Fail gracefully to return "Never" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {obj}.")
        return TypeInfo("typing", "Never")

    # Some builtin types are available from the "builtins" module,
    # some from the "types" module, but others still, such as
    # "list_iterator", aren't known by any particular name.
    if obj.__module__ == "builtins":
        if obj is NoneType:
            return NoneTypeInfo
        elif in_builtins_import(obj):
            return TypeInfo("", obj.__name__, type_obj=obj) # these are "well known", so no module name needed
        elif (name := from_types_import(obj)):
            return TypeInfo("types", name, type_obj=obj)
        elif obj is RANGE_ITER_TYPE:
            return TypeInfo("typing", "Iterator", args=(TypeInfo("", "int", type_obj=int),))
        # TODO match other ABC from collections.abc based on interface
        elif issubclass(obj, abc.Iterator):
            return TypeInfo("typing", "Iterator", args=(AnyTypeInfo,))
        else:
            # fall back to its name, just so we can tell where it came from.
            return TypeInfo.from_type(obj)

    # Certain dtype types' __qualname__ doesn't include a fully qualified name of their inner type
    if obj.__module__ == 'numpy' and 'dtype[' in obj.__name__ and hasattr(obj, "type"):
        t_name = obj.__qualname__.split('[')[0]
        return TypeInfo(obj.__module__, t_name, args=(get_type_name(obj.type, depth+1),))

    # Disabled for now: passing types using aliases at this point can lead to
    # confusion, as there can be an alias that conflicts with a module's fully
    # qualified name.  For example, "import x.y as ast" would conflict with "ast.If"
    if False:
        # Look for a local alias for the type
        # FIXME this only checks for globally known names, and doesn't match inner classes (e.g., Foo.Bar).
        # FIXME that name may be locally bound to something different, hiding the global
        if caller_frame := find_caller_frame():
            current_namespace = caller_frame.f_globals
            if current_namespace.get(obj.__name__) is obj:
                return obj.__name__

            # Check if the type is accessible from a module in the current namespace
            for name, mod in current_namespace.items():
                if (
                    inspect.ismodule(mod)
                    and hasattr(mod, obj.__name__)
                    and getattr(mod, obj.__name__) is obj
                ):
                    return f"{name}.{obj.__name__}"

    return TypeInfo(lookup_type_module(obj), obj.__qualname__, type_obj=obj)


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


def find_function(
    caller_frame: FrameType,
    code: CodeType
) -> abc.Callable|None:
    """Attempts to map back from a code object to the function that uses it."""

    visited = set()

    def find_in_class(class_obj: object) -> abc.Callable|None:
        if class_obj in visited:
            return None
        visited.add(class_obj)

        for obj in class_obj.__dict__.values():
            if isinstance(obj, (FunctionType, classmethod)):
                if (obj := unwrap(obj)) and getattr(obj, "__code__", None) is code:
                    return obj

            elif inspect.isclass(obj):
                if (f := find_in_class(obj)):
                    return f

        return None

    dicts: abc.Iterable[Any] = caller_frame.f_globals.values()
    if caller_frame.f_back:
        dicts = itertools.chain(caller_frame.f_back.f_locals.values(), dicts)

    for obj in dicts:
        if isinstance(obj, FunctionType):
            if (obj := unwrap(obj)) and getattr(obj, "__code__", None) is code:
                return obj

        elif inspect.isclass(obj):
            if (f := find_in_class(obj)):
                return f

    return None


def get_value_type(value: Any, *, use_jaxtyping: bool = False, depth: int = 0) -> TypeInfo:
    """
    get_value_type takes a value (an instance) as input and returns a string representing its type.

    If the value is a collection, it randomly selects an element (or key-value pair) and determines their types.
    If the value is a tuple, it determines the types of all elements in the tuple.

    For other types, it returns the name of the type.
    """
    if depth > 255:
        # We have likely fallen into an infinite recursion.
        # Fail gracefully to return "Never" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {value}.")
        return TypeInfo("typing", "Never")

    t: type|None
    args: tuple[TypeInfo, ...]


    def type_for_generator(
        obj: GeneratorType|AsyncGeneratorType|CoroutineType,
        frame: FrameType|None,
        code: CodeType,
        name: str
    ) -> TypeInfo:
        # FIXME We can't yet retrieve types from annotations because all Callable
        # arguments returned by type_from_annotations are strings
        #if (f := find_function(frame, code)):
        #    ann = type_from_annotations(f)
        #    (now use ann_type.args[-1], the return value)

        return TypeInfo("typing", name, code_id=CodeId(id(code)))


    def recurse(v: Any) -> TypeInfo:
        return get_value_type(v, use_jaxtyping=use_jaxtyping, depth=depth+1)


    if isinstance(value, dict):
        t = dict if isinstance(value, RandomDict) else type(value)
        args = (TypeInfo("typing", "Never"), TypeInfo("typing", "Never"))
        try:
            if value:
                el = value.random_item() if isinstance(value, RandomDict) else sample_from_collection(value.items())
                args = tuple(recurse(fld) for fld in el)
        except Exception:
            pass
        return TypeInfo(lookup_type_module(t), t.__qualname__, args=args)
    elif isinstance(value, (list, set)):
        t = type(value)
        args = (TypeInfo("typing", "Never"),)
        try:
            if value:
                el = sample_from_collection(value)
                args = (recurse(el),)
        except Exception:
            pass
        return TypeInfo(lookup_type_module(t), t.__qualname__, args=args)
    elif (t := _is_instance(value, (abc.KeysView, abc.ValuesView))):
        args = (TypeInfo("typing", "Never"),)
        try:
            if value:
                el = sample_from_collection(value)
                args = (recurse(el),)
        except Exception:
            pass
        return TypeInfo("typing", t.__qualname__, args=args)
    elif isinstance(value, abc.ItemsView):
        args = (TypeInfo("typing", "Never"), TypeInfo("typing", "Never"))
        try:
            if value:
                el = sample_from_collection(value)
                args = tuple(recurse(fld) for fld in el)
        except Exception:
            pass
        return TypeInfo("typing", "ItemsView", args=args)
    elif isinstance(value, tuple):
        if isinstance_namedtuple(value):
            t = type(value)
            return TypeInfo(lookup_type_module(t), t.__qualname__)
        else:
            args = tuple()
            try:
                if value:
                    args = tuple(recurse(fld) for fld in value)
            except Exception:
                pass
            return TypeInfo("", "tuple", args=args)
    elif isinstance(value, (FunctionType, MethodType)):
        return type_from_annotations(value)
    elif isinstance(value, GeneratorType):
        return type_for_generator(value, value.gi_frame, value.gi_code, "Generator")
    elif isinstance(value, AsyncGeneratorType):
        return type_for_generator(value, value.ag_frame, value.ag_code, "AsyncGenerator")
    elif isinstance(value, CoroutineType):
        return type_for_generator(value, value.cr_frame, value.cr_code, "Coroutine")
    elif isinstance(value, type) and value is not type:
        return TypeInfo("", "type", args=(get_type_name(value, depth+1),))
    elif type(value).__name__ == 'async_generator_wrapped_value' and type(value).__module__ == 'builtins':
        import righttyper.traverse as tr
        if (len(v := tr.traverse(value)) == 1):
            return recurse(v[0])
        else:
            # something went wrong with the 'traverse' workaround
            return AnyTypeInfo


    if use_jaxtyping and hasattr(value, "dtype") and hasattr(value, "shape"):
        if (dtype := jx_dtype(value)) is not None:
            shape = " ".join(str(d) for d in value.shape)
            return TypeInfo("jaxtyping", dtype, args=(
                get_type_name(type(value), depth+1),
                f"\"{shape}\""
            ))

    if (t := type(value)).__module__ == 'numpy' and t.__qualname__ == 'ndarray':
        return TypeInfo("numpy", "ndarray", args=(
            AnyTypeInfo,
            get_type_name(type(value.dtype), depth+1)
        ))

    return get_type_name(type(value), depth+1)


def isinstance_namedtuple(obj: object) -> bool:
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )
