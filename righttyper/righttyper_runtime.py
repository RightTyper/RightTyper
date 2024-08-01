import inspect
import random
import sys
from collections.abc import Generator
from functools import cache
from itertools import islice
from types import CodeType, ModuleType

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

from righttyper.righttyper_types import (
    ArgInfo,
    ArgumentName,
    ArgumentType,
    Filename,
    FunctionName,
    T,
    Typename,
    TypenameFrequency,
    TypenameSet,
)
from righttyper.righttyper_utils import (
    skip_this_file,
)


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
        or "righttyper" in code.co_filename
    ):
        return True
    if not (code.co_flags & 0x2):
        import dis

        assert dis.COMPILER_FLAG_NAMES[0x2] == "NEWLOCALS"
        return True
    return False


def get_class_name_from_stack(
    max_depth: int = 5,
) -> Optional[str]:
    # Initialize the current frame
    current_frame = inspect.currentframe()
    try:
        # Move up in the stack frame by frame
        depth = 0
        while current_frame and depth < max_depth:
            # Check if 'self' is in the local variables of the frame
            if "self" in current_frame.f_locals:
                instance = current_frame.f_locals["self"]
                return str(
                    instance.__class__.__name__
                )  # Return class name of the instance
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

    # Helper function to format the type names correctly
    def format_type(t: Any) -> Any:
        if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
            return f"{t.__module__}.{t.__qualname__}"
        if hasattr(t, "__qualname__"):
            return t.__qualname__
        if hasattr(t, "__name__"):
            return t.__name__
        if hasattr(t, "_name"):  # Handling cases like Union
            return t._name
        if hasattr(t, "__origin__"):  # For generics
            return format_type(t.__origin__)
        return str(t)

    # Format the result
    arg_types_str = ", ".join([get_type_name(arg) for arg in arg_types])
    return_type_str = get_type_name(return_type)

    # Construct the Callable type string
    return f"Callable[[{arg_types_str}], {return_type_str}]"


def get_type_name(value: Any, depth: int = 0) -> str:
    # print(f"get_type_name {value}")
    orig_value = value
    if inspect.ismodule(value):
        # For modules, use their __name__ directly
        if getattr(value, "__name__", None):
            return value.__name__
        else:
            return str(type(value))
    elif not inspect.isclass(value):
        # If the value is an instance, get its class
        value = type(value)

    # print(f"HERE FOR {value}")
    # For types, use their __module__ and __name__
    module_name = value.__module__
    type_name = value.__name__

    if module_name == "builtins":
        if type_name == "frame":
            return "FrameType"
        if type_name == "function":
            return get_mypy_type_fn(orig_value)
        if type_name == "NoneType":
            return "None"
        if type_name == "list":
            return get_full_type(orig_value)
        if type_name == "tuple":
            return get_full_type(orig_value)
        if type_name == "Module":
            return get_full_type(orig_value)
        if type_name == "code":
            return "types.CodeType"
        return str(type_name)

    # Get the current global namespace
    current_namespace = sys._getframe(depth + 2).f_globals

    # Check if the type is accessible directly from the current namespace
    if (
        type_name in current_namespace
        and current_namespace[type_name] is value
    ):
        return str(type_name)

    # Check if the type is accessible from a module in the current namespace
    for name, obj in current_namespace.items():
        if (
            inspect.ismodule(obj)
            and hasattr(obj, type_name)
            and getattr(obj, type_name) is value
        ):
            return f"{name}.{type_name}"

    if module_name and module_name != "__main__":
        return f"{module_name}.{type_name}"
    else:
        return str(type_name)


def get_full_type(value: Any, depth: int = 0) -> str:
    """
    get_full_type function takes a value as input and returns a string representing the type of the value.

    If the value is of type dictionary, it randomly selects a pair of key and value from the dictionary
    and recursively determines their types.

    If the value is a list or a set, it randomly selects an item and determines its type recursively.

    If the value is a tuple, it determines the types of all elements in the tuple.

    For other types, it returns the name of the type.
    """
    if isinstance(value, dict):
        # Checking if the value is a dictionary
        if value:
            # If the dictionary is non-empty
            # we sample one of its items randomly.
            n = random.randint(0, len(value) - 1)
            # Here we are using islice with a starting position n and stopping at n + 1
            # to get a random key-value pair from the dictionary
            # FIXME: this is potentially costly and we should cap the range
            key, val = next(islice(value.items(), n, n + 1))
            # We return the type of the dictionary as 'dict[key_type: value_type]'
            return (
                f"Dict[{get_full_type(key, depth + 1)},"
                f" {get_full_type(val, depth + 1)}]"
            )
        else:
            # If the dictionary is empty, we just return 'dict' as the type
            return "Dict[Any]"
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
            # If the list is empty, we return 'list'
            return "List[Any]"
    elif isinstance(value, tuple):
        if isinstance_namedtuple(value):
            return f"{value.__class__.__name__}"
        else:
            # Here we are returning the types of all elements in the tuple
            if len(value) == 0:
                tuple_str = "Tuple"
            else:
                tuple_str = (
                    f"Tuple[{', '.join(get_full_type(elem, depth + 1) for elem in value)}]"
                )
            return tuple_str
    elif inspect.ismethod(value):
        return get_mypy_type_fn(value)
    elif isinstance(value, set):
        if value:
            n = random.randint(0, len(value) - 1)
            elem = next(islice(value, n, n + 1))
            return f"Set[{get_full_type(elem, depth + 1)}]"
        else:
            return "Set[Any]"
    elif isinstance(value, Generator):
        # FIXME DISABLED FOR NOW
        # (q, g) = peek(value)
        # value = g
        # return f"Generator[{get_full_type(q)}, None, None]" # FIXME
        return "Generator[Any, None, None]"  # FIXME
    else:
        # If the value passed is not a dictionary, list, set, or tuple,
        # we return the type of the value as a string
        retval = get_type_name(value, depth)
        return retval


def get_adjusted_full_type(value: Any, class_name: Optional[str]) -> str:
    # Determine the type name of the return value
    if value is None:
        typename = "None"
    elif type(value) in (bool, float, int):
        typename = type(value).__name__
    else:
        typename = get_full_type(value)
        # print(f"typename = {typename}")
        # print(f"class name = {class_name}")
        if typename == class_name:
            typename = "Self"
        # print(f"typename now = {typename}")
    return typename


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


def get_method_signature(method: Any) -> str:
    # try:
    #    type_hints = get_type_hints(method)
    # except Exception:
    #    type_hints = None
    callable_signature = get_mypy_type_fn(method)
    return callable_signature


def update_argtypes(
    argtypes: List[ArgInfo],
    arg_types: Dict[
        Tuple[Filename, FunctionName, ArgumentName],
        ArgumentType,
    ],
    index: Tuple[Filename, FunctionName, ArgumentName],
    the_values: Dict[str, Any],
    class_name: Optional[str],
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
        types = TypenameSet({
            TypenameFrequency(
                Typename(get_adjusted_full_type(val, class_name)),
                1,
            )
            for val in values
        })
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
            the_values[arg],
            ArgumentType.vararg,
        )
    elif arg == varkw:
        assert varkw
        add_arg_info(
            varkw,
            dict,
            the_values[arg].values(),
            ArgumentType.kwarg,
        )
    else:
        add_arg_info(
            arg,
            type(the_values[arg]),
            [the_values[arg]],
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


def old_requires_import(obj: object) -> bool:
    # Get the class of the object
    cls = obj.__class__

    # Get the module where the class is defined
    cls_module = cls.__module__

    # Get the name of the current module
    current_module = sys._getframe(1).f_globals["__name__"]

    # If the class is defined in a different module, an import is needed
    if cls_module != current_module:
        return True
    return False


def requires_import(obj: object) -> bool:
    # Get the class of the object
    cls = type(obj)

    # Get the module where the class is defined
    cls_module = getattr(cls, "__module__", None)

    # If obj is a module or function from an imported module, use its module directly
    if isinstance(obj, ModuleType):
        cls_module = obj.__name__
    elif callable(obj):
        cls_module = getattr(obj, "__module__", None)

    # Handle cases where cls_module might be None
    if cls_module is None:
        return False

    # Get the name of the current module
    current_module = sys._getframe(4).f_globals.get("__name__", "")

    # Debug prints to see what's happening
    # print(f"Object: {obj}")
    # print(f"Class: {cls}")
    # print(f"Class module: {cls_module}")
    # print(f"Current module: {current_module}")

    # If the class is defined in a different module and not in builtins, an import is needed
    if cls_module != current_module and cls_module != "builtins":
        return True
    return False
