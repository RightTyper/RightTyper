import inspect
import os
import random
import sys
import typing
from collections.abc import Generator, AsyncGenerator
from functools import cache
from itertools import islice
from types import CodeType, ModuleType, FrameType
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from righttyper.random_dict import RandomDict
from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    FuncInfo,
    T,
    Typename,
    TypenameFrequency,
    TypenameSet,
)
from righttyper.righttyper_utils import skip_this_file


def get_random_element_from_dict(value: Dict[Any, Any]) -> Any:
    if isinstance(value, RandomDict):
        # If it's a RandomDict, use its built-in random_item method
        return value.random_item()
    else:
        # For a regular dict, use islice to select a random element
        # We limit the range to the first few elements to keep this O(1).
        MAX_ELEMENTS = 10
        n = random.randint(0, min(MAX_ELEMENTS, len(value) - 1))
        return next(islice(value.items(), n, n + 1))


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


def get_class_type_from_stack(
    max_depth: int = 5,
) -> Optional[Type]:
    # Initialize the current frame
    current_frame = inspect.currentframe()
    try:
        # Move up in the stack frame by frame
        depth = 0
        while current_frame and depth < max_depth:
            # Check if 'self' is in the local variables of the frame
            if "self" in current_frame.f_locals:
                instance = current_frame.f_locals["self"]
                return instance.__class__
            # Move one level up in the stack
            current_frame = current_frame.f_back
            depth += 1
    finally:
        # Delete reference to the current frame to avoid reference cycles
        del current_frame
    return None


def get_mypy_type_fn(func: Any) -> str:
    # Get the signature of the function
    signature = inspect.signature(func)

    # Extract argument types, default to Any if no annotation provided
    arg_types = [
        (param.annotation if param.annotation is not param.empty else Any)
        for name, param in signature.parameters.items()
    ]

    # Extract return type, default to Any if no annotation provided
    return_type = signature.return_annotation
    if return_type is inspect.Signature.empty:
        return_type = Any

    # Format the result
    arg_types_str = ", ".join([get_type_name(arg) for arg in arg_types])
    return_type_str = get_type_name(return_type)

    # Construct the Callable type string
    return f"Callable[[{arg_types_str}], {return_type_str}]"


def find_caller_frame() -> Optional[FrameType]:
    """Attempts to find the stack frame which from which we were called. A bit brittle!"""
    from pathlib import Path

    pkg_path = Path(find_caller_frame.__code__.co_filename).parent

    frame: FrameType|None = sys._getframe(1)

    while (frame is not None and frame.f_code is not None and
           pkg_path in Path(frame.f_code.co_filename).parents):
        frame = frame.f_back

    return frame


def get_type_name(obj: object, depth: int = 0) -> str:
    orig_value = obj

    if inspect.ismodule(obj):
        return "types.ModuleType"

    if not inspect.isclass(obj):
        obj = type(obj)

    # Handle numpy and other libraries using dtype
    try:  # workaround failure in Pelican
        if hasattr(orig_value, "dtype"):
            dtype = getattr(orig_value, "dtype")
            t_dtype = type(dtype)
            if hasattr(dtype, "type") and '[' in t_dtype.__name__:
                t_name = t_dtype.__name__.split('[')[0]
                dtype_name = f"{t_dtype.__module__}.{t_name}[{dtype.type.__module__}.{dtype.type.__qualname__}]"
            else:
                dtype_name = f"{t_dtype.__module__}.{t_dtype.__name__}"

            return f"{obj.__module__}.{obj.__name__}[Any, {dtype_name}]"
    except RuntimeError:
        pass

    # Handle built-in types (like list, tuple, NoneType, etc.)
    if obj.__module__ == "builtins":
        if obj.__name__ == "frame":
            return "FrameType"
        elif obj.__name__ == "function":
            return get_mypy_type_fn(orig_value)
        elif obj.__name__ == "NoneType":
            return "None"
        elif obj.__name__ in {"list", "tuple"}:
            return get_full_type(orig_value, depth + 1)
        elif obj.__name__ == "code":
            return "types.CodeType"
        elif obj.__name__ == "range":
            return "Iterable[int]"
        elif obj.__name__ == "range_iterator":
            return "Iterator[int]"
        elif obj.__name__ == "enumerate":
            return "Iterator[Tuple[int, Any]]"
        elif obj.__name__ in (
            "list_iterator", "list_reverseiterator", "set_iterator",
            "dict_keyiterator", "dict_valueiterator", "dict_itemiterator",
            "filter", "map", "zip"
        ):
            return "Iterator[Any]"  # FIXME needs element type
        elif obj.__name__ in (
            "dict_keys", "dict_values"
        ):
            try:
                # FIXME sample rather than use 1st
                el = next(iter(cast(typing.Iterable, orig_value)))
                return f"Iterable[{get_type_name(el, depth+1)}]"
            except StopIteration:
                return "Iterable[Never]"
        elif obj.__name__ == "dict_items":
            try:
                # FIXME sample rather than use 1st
                el = next(iter(cast(typing.Iterable, orig_value)))
                return f"Iterable[Tuple[{get_type_name(el[0], depth+1)}, {get_type_name(el[1], depth+1)}]]"
            except StopIteration:
                return "Iterable[Tuple[Never, Never]]"
        elif obj.__name__ == "coroutine":
            # FIXME need yield / send / return type
            return "Coroutine[Any, Any, Any]"
        else:
            return obj.__name__

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

    # Handle generic types (like List, Dict, etc.)
    origin = typing.get_origin(obj)
    if origin:
        type_name = f"{origin.__module__}.{origin.__name__}"
        type_params = ", ".join(
            get_full_type(arg, depth + 1) for arg in typing.get_args(obj)
        )
        return f"{type_name}[{type_params}]"

    # Handle all other types with fully qualified names
    if obj.__module__ and obj.__module__ != "__main__":
        return f"{obj.__module__}.{obj.__qualname__}"

    return obj.__qualname__


def get_full_type(value: Any, depth: int = 0) -> str:
    """
    get_full_type takes a value as input and returns a string representing the type of the value.

    If the value is of type dictionary, it randomly selects a pair of key and value from the dictionary
    and recursively determines their types.

    If the value is a list or a set, it randomly selects an item and determines its type recursively.

    If the value is a tuple, it determines the types of all elements in the tuple.

    For other types, it returns the name of the type.
    """
    if depth > 255:
        # We have likely fallen into an infinite recursion.
        # Fail gracefully to return "Never" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {value}.")
        return "Never"
    if isinstance(value, dict):
        # Checking if the value is a dictionary
        if value:
            key, val = get_random_element_from_dict(value)
            return (
                f"Dict[{get_full_type(key, depth + 1)},"
                f" {get_full_type(val, depth + 1)}]"
            )
        else:
            return "Dict[Never, Never]"
    elif isinstance(value, list):
        # Checking if the value is a list
        if value:
            # If the list is non-empty
            # we sample one of its elements randomly
            n = random.randint(0, len(value) - 1)
            elem = value[n]
            # We return the type of the list as 'list[element_type]'
            return f"List[{get_full_type(elem, depth + 1)}]"
        else:
            return "List[Never]"
    elif isinstance(value, tuple):
        if isinstance_namedtuple(value):
            return f"{value.__class__.__name__}"
        else:
            # Here we are returning the types of all elements in the tuple
            if len(value) == 0:
                tuple_str = "Tuple"
            else:
                tuple_str = f"Tuple[{', '.join(get_full_type(elem, depth + 1) for elem in value)}]"
            return tuple_str
    elif inspect.ismethod(value):
        return get_mypy_type_fn(value)
    elif isinstance(value, set):
        if value:
            n = random.randint(0, len(value) - 1)
            elem = next(islice(value, n, n + 1))
            return f"Set[{get_full_type(elem, depth + 1)}]"
        else:
            return "Set[Never]"
    elif isinstance(value, Generator):
        # FIXME DISABLED FOR NOW
        # (q, g) = peek(value)
        # value = g
        # return f"Generator[{get_full_type(q)}, None, None]" # FIXME
        return "Generator[Any, Any, Any]"  # FIXME
    elif isinstance(value, AsyncGenerator):
        return "AsyncGenerator[Any, Any]"  # FIXME needs argument types
    else:
        # If the value passed is not a dictionary, list, set, or tuple,
        # we return the type of the value as a string
        return get_type_name(value, depth + 1)


def get_adjusted_full_type(value: Any, class_type: Optional[Type]=None) -> str:
    #print(f"{type(value)=} {class_type=}")
    if type(value) == class_type:
        return "Self"

    return get_full_type(value)


def isinstance_namedtuple(obj: object) -> bool:
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_asdict")
        and hasattr(obj, "_fields")
    )


def peek(
    generator: Generator[T, Any, Any],
) -> Tuple[T, Generator[T, Any, Any]]:
    def wrapped_generator() -> Generator[T, Any, Any]:
        # Yield the peeked value first, then yield from the original generator
        yield peeked_value
        yield from generator

    # Get the next value from the original generator
    peeked_value = next(generator)

    # Return the peeked value and the new generator
    return peeked_value, wrapped_generator()


def update_argtypes(
    argtypes: List[ArgInfo],
    arg_types: Dict[
        Tuple[FuncInfo, ArgumentName],
        ArgumentType,
    ],
    index: Tuple[FuncInfo, ArgumentName],
    arg_values: Any,
    class_type: Optional[Type],
    arg: str,
    varargs: Optional[str],
    varkw: Optional[str],
) -> None:

    def add_arg_info(
        argument_name: str,
        arg_type: type,
        values: Any,
        arg_type_enum: ArgumentType,
    ) -> None:
        if not all(v is None for v in values):
            types = TypenameSet(
                {
                    TypenameFrequency(
                        Typename(get_adjusted_full_type(val, class_type)),
                        1,
                    )
                    for val in values
                }
            )
            argtypes.append(
                ArgInfo(
                    ArgumentName(argument_name),
                    arg_type,
                    types,
                )
            )
            arg_types[index] = arg_type_enum

    if arg == varargs:
        assert varargs
        add_arg_info(
            varargs,
            tuple,
            arg_values[0],
            ArgumentType.vararg,
        )
    elif arg == varkw:
        assert varkw
        add_arg_info(
            varkw,
            dict,
            arg_values[0].values(),
            ArgumentType.kwarg,
        )
    else:
        add_arg_info(
            arg,
            type(arg_values[0]),
            arg_values,
            ArgumentType.positional,
        )


def format_annotation(annotation: Any) -> str:
    """Format an annotation (type hint) as a string."""
    if isinstance(annotation, type):
        return annotation.__name__
    elif hasattr(annotation, "_name") and annotation._name is not None:
        return str(annotation._name)
    elif (
        hasattr(annotation, "__origin__") and annotation.__origin__ is not None
    ):
        origin = format_annotation(annotation.__origin__)
        args = ", ".join(
            [format_annotation(arg) for arg in annotation.__args__]
        )
        return f"{origin}[{args}]"
    else:
        return str(annotation)


def format_function_definition(
    func_name: str,
    arg_names: List[str],
    type_hints: Dict[str, Any],
) -> str:
    """Format the function definition based on its name, argument names, and type hints."""
    params = []
    for arg in arg_names:
        type_hint = type_hints.get(arg)
        if type_hint:
            params.append(f"{arg}: {format_annotation(type_hint)}")
        else:
            params.append(arg)

    return_annotation = ""
    if "return" in type_hints:
        return_annotation = f" -> {format_annotation(type_hints['return'])}"

    params_str = ", ".join(params)
    function_definition = f"def {func_name}({params_str}){return_annotation}:"
    return function_definition


def get_class_source_file(cls: Type[Any]) -> str:
    module_name = cls.__module__

    # Check if the class is built-in
    if module_name == "builtins":
        return ""  # Built-in classes do not have source files

    try:
        # Try to get the module from sys.modules
        module = sys.modules[module_name]
        # Try to get the __file__ attribute
        file_path = getattr(module, "__file__", None)
        if file_path:
            return str(file_path)
        # If __file__ is not available, use inspect to get the source file
        import inspect

        file_path = inspect.getfile(cls)
        return file_path
    except (KeyError, TypeError, AttributeError):
        pass

    # Derive the file path from the module name
    try:
        # Assuming the module is part of the standard package structure
        import os

        module_parts = module_name.split(".")
        file_path = os.path.join(*module_parts)
        return file_path

    except Exception:
        pass

    return ""
