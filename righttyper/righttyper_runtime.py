import inspect
import os
import random
import sys
import collections.abc as abc
from functools import cache
from itertools import islice
from types import CodeType, FrameType, NoneType, FunctionType, MethodType, GenericAlias
from typing import Any, cast, TypeAlias
from pathlib import Path

from righttyper.random_dict import RandomDict
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    FuncInfo,
    T,
    Typename,
    TypenameSet,
)
from righttyper.righttyper_utils import skip_this_file


def sample_from_collection(value: abc.Collection[T]|abc.Iterator[T], depth = 0) -> T:
    """Samples from a collection, or from an interator/generator whose state we don't mind changing."""
    MAX_ELEMENTS = 10   # to keep this O(1)

    if isinstance(value, abc.Collection):
        n = random.randint(0, min(MAX_ELEMENTS, len(value) - 1))
        return next(islice(value, n, n + 1))

    n = random.randint(1, MAX_ELEMENTS)
    return list(islice(value, n))[-1]


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
    include_files_regex: str,
) -> bool:
    if (
        code.co_name.startswith("<")
        or skip_this_file(
            code.co_filename,
            script_dir,
            include_all,
            include_files_regex,
        )
        or "righttyper" + os.sep in code.co_filename
    ):
        return True
    if not (code.co_flags & 0x2):
        import dis

        assert dis.COMPILER_FLAG_NAMES[0x2] == "NEWLOCALS"
        return True
    return False


def type_from_annotations(func: FunctionType | MethodType) -> str:
    # Get the signature of the function
    signature = inspect.signature(func)
    #print(f"{func=} {signature=}")

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

        if arg is None:
            return "None"

        return get_type_name(arg)

    # Format the result
    arg_types_str = ", ".join([format_arg(arg) for arg in arg_types])
    return_type_str = format_arg(return_type)

    # Construct the Callable type string
    return f"typing.Callable[[{arg_types_str}], {return_type_str}]"


def find_caller_frame() -> FrameType|None:
    """Attempts to find the stack frame which from which we were called. A bit brittle!"""
    from pathlib import Path

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

    if (m := sys.modules.get(t.__module__)):
        if is_defined_in_module(m.__dict__):
            return t.__module__

    module_prefix = f"{t.__module__}."
    for name, mod in sys.modules.items():
        if name.startswith(module_prefix) and is_defined_in_module(mod.__dict__):
            return name

    # it's not in the module, but keep it as a last resort, to facilitate diagnostics
    return t.__module__


RANGE_ITER_TYPE = type(iter(range(1)))

def get_type_name(obj: type, depth: int = 0) -> str:
    """Returns a type's name as a string."""

    if depth > 255:
        # We have likely fallen into an infinite recursion.
        # Fail gracefully to return "Never" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {obj}.")
        return "typing.Never"

    # Some builtin types are available from the "builtins" module,
    # some from the "types" module, but others still, such as
    # "list_iterator", aren't known by any particular name.
    if obj.__module__ == "builtins":
        if obj is NoneType:
            return "None"
        elif in_builtins_import(obj):
            return obj.__name__ # these are "well known", so no module name needed
        elif (name := from_types_import(obj)):
            return f"types.{name}"
        elif obj is RANGE_ITER_TYPE:
            return "typing.Iterator[int]"
        # TODO match other ABC from collections.abc based on interface
        elif issubclass(obj, abc.Iterator):
            return "typing.Iterator[typing.Any]"
        else:
            # fall back to its name, just so we can tell where it came from.
            return f"builtins.{obj.__name__}"

    # Certain dtype types' __qualname__ doesn't include a fully qualified name of their inner type
    if obj.__module__ == 'numpy' and 'dtype[' in obj.__name__ and hasattr(obj, "type"):
        t_name = obj.__qualname__.split('[')[0]
        return f"{obj.__module__}.{t_name}[{get_type_name(obj.type, depth+1)}]"

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

    if obj.__module__ == "__main__":    # TODO merge this into lookup_type_module
        return f"{get_main_module_fqn()}.{obj.__qualname__}"

    return f"{lookup_type_module(obj)}.{obj.__qualname__}"


def _is_instance(obj: object, types: tuple[type, ...]) -> type|None:
    """Like isinstance(), but returns the type matched."""
    for t in types:
        if isinstance(obj, t):
            return t

    return None


def get_full_type(value: Any, /, use_jaxtyping: bool = False, depth: int = 0) -> str:
    """
    get_full_type takes a value (an instance) as input and returns a string representing its type.

    If the value is a collection, it randomly selects an element (or key-value pair) and determines their types.
    If the value is a tuple, it determines the types of all elements in the tuple.

    For other types, it returns the name of the type.
    """
    if depth > 255:
        # We have likely fallen into an infinite recursion.
        # Fail gracefully to return "Never" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {value}.")
        return "typing.Never"

    t: type|None

    if isinstance(value, dict):
        t = type(value)
        module = "" if t.__module__ == "builtins" else f"{t.__module__}."
        if value:
            el = value.random_item() if isinstance(value, RandomDict) else sample_from_collection(value.items())
            return f"{module}{t.__qualname__}[{', '.join(get_full_type(fld, depth=depth+1) for fld in el)}]"
        else:
            return f"{module}{t.__qualname__}[typing.Never, typing.Never]"
    elif isinstance(value, (list, set)):
        t = type(value)
        module = "" if t.__module__ == "builtins" else f"{t.__module__}."
        if value:
            el = sample_from_collection(value)
            return f"{module}{t.__qualname__}[{get_full_type(el, depth=depth+1)}]"
        else:
            return f"{module}{t.__qualname__}[typing.Never]"
    elif (t := _is_instance(value, (abc.KeysView, abc.ValuesView))):
        if value:
            el = sample_from_collection(value)
            return f"typing.{t.__qualname__}[{get_full_type(el, depth=depth+1)}]"
        else:
            return f"typing.{t.__qualname__}[typing.Never]"
    elif isinstance(value, abc.ItemsView):
        if value:
            el = sample_from_collection(value)
            return f"typing.ItemsView[{get_full_type(el[0], depth=depth+1)}, {get_full_type(el[1], depth=depth+1)}]"
        else:
            return "typing.ItemsView[typing.Never, typing.Never]"
    elif isinstance(value, tuple):
        if isinstance_namedtuple(value):
            return f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        else:
            if value:
                return f"tuple[{', '.join(get_full_type(elem, depth=depth+1) for elem in value)}]"
            else:
                return "tuple"
    elif isinstance(value, (FunctionType, MethodType)):
        return type_from_annotations(value)
    elif isinstance(value, abc.Generator):
        return "typing.Generator[typing.Any, typing.Any, typing.Any]"  # FIXME needs yield / send / return types
    elif isinstance(value, abc.AsyncGenerator):
        return "typing.AsyncGenerator[typing.Any, typing.Any]"  # FIXME needs yield / send types
    elif isinstance(value, abc.Coroutine):
        return "typing.Coroutine[typing.Any, typing.Any, typing.Any]"  # FIXME needs yield / send / return types


    if use_jaxtyping and hasattr(value, "dtype") and hasattr(value, "shape"):
        if (dtype := jx_dtype(value)) is not None:
            shape = " ".join(str(d) for d in value.shape)
            return f'jaxtyping.{dtype}[{get_type_name(type(value), depth+1)}, "{shape}"]'

    if (t := type(value)).__module__ == 'numpy' and t.__qualname__ == 'ndarray':
        return f"{get_type_name(t, depth+1)}[typing.Any, {get_type_name(type(value.dtype), depth+1)}]"

    return get_type_name(type(value), depth+1)


def isinstance_namedtuple(obj: object) -> bool:
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )


def update_argtypes(
    argtypes: list[ArgInfo],
    index: tuple[FuncInfo, ArgumentName],
    arg_values: Any,
    argument_name: str,
    /,
    is_vararg: bool,
    is_kwarg: bool,
    use_jaxtyping: bool
) -> None:

    def add_arg_info(
        values: Any,
    ) -> None:
        types = TypenameSet([
            Typename(get_full_type(val, use_jaxtyping=use_jaxtyping))
            for val in values
        ])
        argtypes.append(
            ArgInfo(
                ArgumentName(argument_name),
                types,
            )
        )

    if is_vararg:
        add_arg_info(arg_values[0])
    elif is_kwarg:
        add_arg_info(arg_values[0].values())
    else:
        add_arg_info(arg_values)


def _source_relative_to_pkg(file: Path) -> Path|None:
    """Returns a Python source file's path relative to its package"""
    if not file.is_absolute():
        file = file.resolve()

    parents = list(file.parents)

    for d in sys.path:
        path = Path(d)
        if not path.is_absolute():
            path = path.resolve()

        for p in parents:
            if p == path:
                return file.relative_to(p)

    return None


def source_to_module_fqn(file: Path) -> str|None:
    """Returns a source file's fully qualified package name, if possible."""
    if not (path := _source_relative_to_pkg(file)):
        return None

    path = path.parent if path.name == '__init__.py' else path.parent / path.stem
    return '.'.join(path.parts)


@cache
def get_main_module_fqn() -> str:
    main = sys.modules['__main__']
    if hasattr(main, "__file__") and main.__file__:
        if fqn := source_to_module_fqn(Path(main.__file__)):
            return fqn

    return "__main__"
