import inspect
import logging
import sys
from functools import lru_cache
from types import ModuleType
from typing import Any, List

from righttyper.righttyper_types import ImportDetails

import libcst as cst

logger = logging.getLogger("righttyper")


# @lru_cache
def get_import_details(
    obj: Any,
) -> ImportDetails:
    """
    Get details about the import of an object, including its name, aliases, module name, and module aliases.

    Parameters:
    obj (Any): The imported object (e.g., a class, function, or module) for which to retrieve import details.

    Returns:
    ImportDetails:
        - The name of the object.
        - A set of aliases used for the object in the current global scope.
        - The name of the module from which the object was imported.
        - A set of aliases used for the module in the current global scope.

    Example:
    from rich.console import Console as Banana
    import numpy as np
    from some.thing import thingy as barnacle

    print(get_import_details(Banana))     # ImportDetails("Console", ["Banana"], "rich.console", [])
    print(get_import_details(np.ndarray)) # ImportDetails("ndarray", [], "numpy", ["np"])
    print(get_import_details(np))         # ImportDetails("numpy", ["np"], "numpy", ["np"])
    print(get_import_details(barnacle))   # ImportDetails("thingy", ["barnacle"], "some.thing", [])
    """
    obj_name = obj.__name__ if hasattr(obj, "__name__") else type(obj).__name__
    module_name = (
        obj.__module__
        if hasattr(obj, "__module__")
        else obj.__class__.__module__
    )

    # Check if the object is built-in
    if module_name == "builtins":
        return ImportDetails(
            obj_name,
            frozenset(),
            "builtins",
            frozenset(),
        )

    module_aliases = set()
    obj_aliases = set()
    frame = inspect.currentframe()
    frame = getattr(frame, "f_back", None)

    while frame:

        # Get set of function parameters which we will exclude as names
        local_scope_exclusions = set(
            frame.f_code.co_varnames[: frame.f_code.co_argcount]
        )

        # Check both globals and locals of the current frame
        for scope in [
            frame.f_globals,
            frame.f_locals,
        ]:
            module_obj = sys.modules.get(module_name)
            for (
                alias_name,
                imported_obj,
            ) in scope.items():
                if (
                    isinstance(imported_obj, ModuleType)
                    and imported_obj is module_obj
                ):
                    if (
                        alias_name != module_name
                        and alias_name not in module_aliases
                        and alias_name not in local_scope_exclusions
                    ):
                        module_aliases.add(alias_name)
                elif (
                    imported_obj is obj
                    and alias_name != obj_name
                    and alias_name not in obj_aliases
                    and alias_name not in local_scope_exclusions
                ):
                    obj_aliases.add(alias_name)

        frame = frame.f_back
    tup = ImportDetails(
        obj_name,
        frozenset(), # temporarily disabling this: frozenset(obj_aliases) - this was intended to capture import aliases but is not reliable
        module_name,
        frozenset(module_aliases),
    )

    return tup


def testme() -> None:
    import numpy
    import numpy as np
    from rich.console import Console as ApplePie
    from rich.console import Console as Banana

    # Test the functions
    assert get_import_details(numpy.ndarray) == ImportDetails(
        "ndarray",
        frozenset({}),
        "numpy",
        frozenset({"np"}),
    )
    assert get_import_details(np.ndarray) == ImportDetails(
        "ndarray",
        frozenset({}),
        "numpy",
        frozenset({"np"}),
    )
    assert get_import_details(Banana) == ImportDetails(
        "Console",
        frozenset({"ApplePie", "Banana"}),
        "rich.console",
        frozenset({}),
    )
    # print_possible_imports(get_import_details(Banana))
    assert get_import_details(ApplePie) == ImportDetails(
        "Console",
        frozenset({"ApplePie", "Banana"}),
        "rich.console",
        frozenset({}),
    )
    assert get_import_details(lru_cache) == ImportDetails(
        "lru_cache",
        frozenset({}),
        "functools",
        frozenset({}),
    )


def print_possible_imports(
    details: ImportDetails,
) -> None:
    """
    Print every possible import statement that might be needed based on the details provided.

    Parameters:
    details (ImportDetails):
        The output from `get_import_details` function, containing:
        - The name of the object.
        - A set of aliases used for the object in the current global scope.
        - The name of the module from which the object was imported.
        - A set of aliases used for the module in the current global scope.
    """

    # Print import statement for the object
    print(f"from {details.import_module_name} import {details.object_name}")
    for alias in details.object_aliases:
        print(
            f"from {details.import_module_name} import {details.object_name} as {alias}"
        )

    # Print import statement for the module
    print(f"import {details.import_module_name}")
    for alias in details.module_aliases:
        print(f"import {details.import_module_name} as {alias}")


# Example usage with the provided test data
# details = get_import_details(Banana)
# print_possible_imports(details)


def generate_import_nodes(
    details: ImportDetails,
) -> List[cst.Import | cst.ImportFrom | cst.EmptyLine]:
    """
    Generate import nodes for libcst based on the details provided.

    Parameters:
    details (Tuple[str, List[str], str, List[str]]):
        The output from `get_import_details` function, containing:
        - The name of the object.
        - A set of aliases used for the object in the current global scope.
        - The name of the module from which the object was imported.
        - A set of aliases used for the module in the current global scope.

    Returns:
    List[cst.CSTNode]: A list of libcst import nodes.
    """

    import_nodes: List[cst.Import | cst.ImportFrom | cst.EmptyLine] = []

    # Helper function to create a dotted name
    def create_dotted_name(
        name: str,
    ) -> cst.Attribute | cst.Name:
        parts = name.split(".")
        if len(parts) == 1:
            return cst.Name(parts[0])
        base: cst.Name | cst.Attribute = cst.Name(parts[0])
        for part in parts[1:]:
            base = cst.Attribute(value=base, attr=cst.Name(part))
        return base

    # Import statement for the object
    try:

        import_nodes.append(
            cst.ImportFrom(
                module=create_dotted_name(details.import_module_name),
                names=[cst.ImportAlias(name=cst.Name(details.object_name))],
            )
        )
        import_nodes.append(cst.EmptyLine())
    except cst.CSTValidationError:
        logger.warning(
            f"Failed to add from {details.import_module_name=} import {details.object_name=}"
        )

    for alias in details.object_aliases:
        try:
            import_from_stmt = cst.ImportFrom(
                module=create_dotted_name(details.import_module_name),
                names=[
                    cst.ImportAlias(
                        name=cst.Name(details.object_name),
                        asname=cst.AsName(name=cst.Name(alias)),
                    )
                ],
            )
            import_nodes.append(import_from_stmt)
            import_nodes.append(cst.EmptyLine())
        except cst.CSTValidationError:
            logger.warning(
                f"Failed to add import {details.object_name=} as {alias=}"
            )

    # Import statement for the module
    stmt = cst.Import(
        names=[
            cst.ImportAlias(
                name=create_dotted_name(details.import_module_name)
            )
        ]
    )
    import_nodes.append(stmt)
    import_nodes.append(cst.EmptyLine())

    for alias in details.module_aliases:
        import_stmt = cst.Import(
            names=[
                cst.ImportAlias(
                    name=create_dotted_name(details.import_module_name),
                    asname=cst.AsName(name=cst.Name(alias)),
                )
            ]
        )
        import_nodes.append(import_stmt)
        import_nodes.append(cst.EmptyLine())

    return import_nodes


# Example usage with the provided test data
# details = get_import_details(Banana)
# import_nodes = generate_import_nodes(details)
# for node in import_nodes:
#    print(cst.Module([]).code_for_node(node))


if __name__ == "__main__":
    testme()
