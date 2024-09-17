import inspect
import random
import sys
import typing
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
    FuncInfo,
    T,
    Typename,
    TypenameFrequency,
    TypenameSet,
)
from righttyper.righttyper_utils import (
    skip_this_file,
)

from righttyper.random_dict import (
    RandomDict,
)

def get_random_element_from_dict(value: Dict[Any, Any]) -> Any:
    if isinstance(value, RandomDict):
        # If it's a RandomDict, use its built-in random_item method
        # print("RandomDict")
        return value.random_item()
    else:
        # For a regular dict, use islice to select a random element
        # We limit the range to the first few elements to keep this O(1).
        # print("ordinary dict")
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


def get_type_name(obj: object, depth: int = 0) -> str:
    retval = get_type_name_helper(obj, depth)
    return retval

        
def get_type_name_helper(obj: object, depth: int = 0) -> str:
    orig_value = obj

    # Handle module types
    if inspect.ismodule(obj):
        if getattr(obj, "__name__", None):
            return obj.__name__
        else:
            return str(type(obj))

    # Handle class instances by retrieving their type
    if not inspect.isclass(obj):
        obj = type(obj)

    # Handle numpy and other libraries using dtype
    try: # workaround failure in Pelican
        if hasattr(orig_value, 'dtype'):
            dtype = getattr(orig_value, 'dtype')
            # Use type(dtype).__module__ and type(dtype).__name__ to get the fully qualified name for the dtype
            retval = f"{obj.__module__}.{obj.__name__}[Any, {type(dtype).__module__}.{type(dtype).__name__}]"
            # Forcibly strip builtins, which are somehow getting in there
            # retval = retval.replace("builtins.", "")
            return retval
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
        else:
            return obj.__name__

    # Check if the type is accessible from the current global namespace
    current_namespace = sys._getframe(depth + 2).f_globals
    if obj.__name__ in current_namespace and current_namespace[obj.__name__] is obj:
        return obj.__name__

    # Check if the type is accessible from a module in the current namespace
    for name, mod in current_namespace.items():
        if inspect.ismodule(mod) and hasattr(mod, obj.__name__) and getattr(mod, obj.__name__) is obj:
            return f"{name}.{obj.__name__}"

    # Handle generic types (like List, Dict, etc.)
    origin = typing.get_origin(obj)
    if origin:
        type_name = f"{origin.__module__}.{origin.__name__}"
        type_params = ", ".join(get_full_type(arg, depth + 1) for arg in typing.get_args(obj))
        return f"{type_name}[{type_params}]"

    # Handle all other types with fully qualified names
    if obj.__module__ and obj.__module__ != "__main__":
        return f"{obj.__module__}.{obj.__name__}"

    return obj.__name__


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
        # Fail gracefully to return "Any" while reporting the warning.
        print(f"Warning: RightTyper failed to compute the type of {value}.")
        return "Any"
    if isinstance(value, dict):
        # Checking if the value is a dictionary
        if value:
            key, val = get_random_element_from_dict(value)
            return (
                f"Dict[{get_full_type(key, depth + 1)},"
                f" {get_full_type(val, depth + 1)}]"
            )
        else:
            # If the dictionary is empty, we just return a generic dict as the type
            return "Dict[Any, Any]"
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
        retval = get_type_name(value, depth + 1)
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
        Tuple[FuncInfo, ArgumentName],
        ArgumentType,
    ],
    index: Tuple[FuncInfo, ArgumentName],
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
        types = TypenameSet(
            {
                TypenameFrequency(
                    Typename(get_adjusted_full_type(val, class_name)),
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
