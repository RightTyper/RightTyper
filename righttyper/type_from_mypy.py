"""
mypy_stub_type_hints.py

A small helper module that gives you a typing.get_type_hints-like API,
but using mypy's view of types from .pyi stubs (including Typeshed).

Core entry points:

    get_type_hints_from_stub_file("foo.pyi", "C.m")
    get_type_hints_from_typeshed_module("collections.abc", "Iterable")

The result is a mapping:

    { name: TypeRepr, ..., "return": TypeRepr }

where TypeRepr is a structured representation with separate
(module, qualname) fields for instance-like types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mypy.build import build, BuildResult
from mypy.modulefinder import BuildSource
from mypy.options import Options
from mypy.nodes import (
    MypyFile,
    FuncDef,
    OverloadedFuncDef,
    TypeInfo,
    Var,
    SymbolTableNode,
    SymbolTable,
)
from mypy.types import (
    Type,
    CallableType,
    Overloaded,
    Instance,
    AnyType,
    NoneType,
    UnboundType,
    TypeVarType,
    ParamSpecType,
    TypeVarTupleType,
    UnpackType,
    UnionType,
    TupleType,
    TypeType,
    LiteralType,
    get_proper_type,
)


class StubTypeError(RuntimeError):
    """Generic error while trying to read type hints from a stub or typeshed module."""


# ---------------------------------------------------------------------------
# Structured type representation
# ---------------------------------------------------------------------------

@dataclass
class TypeRepr:
    """
    A simple, structured representation of a mypy Type.

    Examples:

        # list[str]
        TypeRepr(
            kind="instance",
            module="builtins",
            qualname="list",
            args=[TypeRepr(kind="instance", module="builtins", qualname="str")]
        )

        # Iterable[str] | None
        TypeRepr(
            kind="union",
            args=[
                TypeRepr(kind="instance", module="collections.abc", qualname="Iterable", args=[...]),
                TypeRepr(kind="none"),
            ],
        )

    Fields:
        kind:     string descriptor of the type kind ("instance", "union", "any", etc.)
        module:   module name for instance-like things (e.g. "builtins", "collections.abc")
        qualname: qualified name within the module or type variable / literal name
        args:     sub-components (e.g., generic arguments, union members, tuple elements)
    """
    kind: str
    module: Optional[str] = None
    qualname: Optional[str] = None
    args: List["TypeRepr"] = field(default_factory=list)


def _typeinfo_module_and_qualname(ti: TypeInfo) -> Tuple[Optional[str], str]:
    """
    Return (module_name, qualname_without_module) for a TypeInfo.

    We try to use .fullname when available, but fall back gracefully when
    _fullname isn't initialized.
    """
    module = ti.module_name or None
    try:
        full = ti.fullname  # may raise AttributeError if _fullname not set
    except AttributeError:
        # Best-effort fallback
        return module, ti.name

    if module and full.startswith(module + "."):
        qual = full[len(module) + 1 :]
    else:
        # Either no module_name or fullname isn't prefixed by it;
        # just treat fullname as the "qualified" name.
        qual = full
    return module, qual


def _type_to_repr(typ: Type) -> TypeRepr:
    """
    Convert a mypy.types.Type into a TypeRepr, without using Type.accept()
    (which expects a mypyc-compiled TypeVisitor).
    """
    t = get_proper_type(typ)

    # Instance: like list[str], collections.abc.Iterable[int], etc.
    if isinstance(t, Instance):
        module, qual = _typeinfo_module_and_qualname(t.type)
        args = [_type_to_repr(arg) for arg in t.args]
        return TypeRepr(kind="instance", module=module, qualname=qual, args=args)

    # Simple scalar-ish things
    if isinstance(t, AnyType):
        return TypeRepr(kind="any")

    if isinstance(t, NoneType):
        return TypeRepr(kind="none")

    if isinstance(t, UnboundType):
        return TypeRepr(kind="unbound")

    if isinstance(t, TypeVarType):
        # E.g., T, _T_co, etc.
        return TypeRepr(kind="typevar", qualname=t.name)

    if isinstance(t, ParamSpecType):
        return TypeRepr(kind="paramspec", qualname=t.name)

    if isinstance(t, TypeVarTupleType):
        return TypeRepr(kind="typevartuple", qualname=t.name)

    if isinstance(t, UnpackType):
        return TypeRepr(kind="unpack", args=[_type_to_repr(t.type)])

    # Composite types
    if isinstance(t, UnionType):
        return TypeRepr(kind="union", args=[_type_to_repr(item) for item in t.items])

    if isinstance(t, TupleType):
        return TypeRepr(kind="tuple", args=[_type_to_repr(item) for item in t.items])

    if isinstance(t, CallableType):
        # For fields whose type is Callable[…], not function signatures (those
        # are handled separately in _callable_type_hints).
        arg_reprs = [_type_to_repr(arg_t) for arg_t in t.arg_types]
        ret_repr = _type_to_repr(t.ret_type)
        return TypeRepr(kind="callable", args=arg_reprs + [ret_repr])

    if isinstance(t, TypeType):
        return TypeRepr(kind="type", args=[_type_to_repr(t.item)])

    if isinstance(t, LiteralType):
        # Represent as Literal[value] with underlying fallback as the arg.
        underlying = _type_to_repr(t.fallback)
        return TypeRepr(
            kind="literal",
            qualname=repr(t.value),
            args=[underlying],
        )

    if isinstance(t, Overloaded):
        # You typically won't see Overloaded here; we generally peel it off
        # earlier for functions. But keep it safe.
        return TypeRepr(
            kind="overloaded",
            args=[_type_to_repr(item) for item in t.items()],
        )

    # Fallback: at least give a name so you see what you hit.
    return TypeRepr(kind=t.__class__.__name__.lower())

# ---------------------------------------------------------------------------
# Common mypy helpers
# ---------------------------------------------------------------------------

def _make_mypy_options(python_version: Tuple[int, int] = (3, 12)) -> Options:
    """Create a reasonably configured Options object for stub analysis."""
    opts = Options()
    opts.incremental = False
    opts.show_traceback = True
    opts.export_types = True  # keep inferred types
    opts.namespace_packages = True
    opts.python_version = python_version
    # You can tweak search_path / mypy_path here if needed.
    return opts


def _build_for_sources(
    sources: List[BuildSource],
    module_name: str,
    python_version: Tuple[int, int] = (3, 12),
) -> Tuple[BuildResult, MypyFile]:
    """
    Run mypy on the given sources and return (BuildResult, MypyFile).

    `module_name` must be the logical module name (e.g. "typing", "mymod").
    """
    opts = _make_mypy_options(python_version)
    result = build(sources=sources, options=opts)

    if result.errors:
        raise StubTypeError("mypy reported errors:\n" + "\n".join(result.errors))

    state = result.graph.get(module_name)
    if state is None:
        raise StubTypeError(f"Module {module_name!r} not found in mypy graph")

    state.load_tree()
    if state.tree is None:
        raise StubTypeError(f"No AST tree for module {module_name!r}")

    tree: MypyFile = state.tree  # type: ignore[assignment]
    return result, tree


# ---------------------------------------------------------------------------
# Qualname resolution inside a MypyFile
# ---------------------------------------------------------------------------

def _lookup_qualname(tree: MypyFile, qualname: str):
    """
    Resolve a qualified name like 'C.m' or 'f' within a MypyFile.

    Returns the underlying mypy.nodes.Node (FuncDef, OverloadedFuncDef,
    TypeInfo, Var, etc.), or raises StubTypeError if not found.
    """
    parts = qualname.split(".")
    node: MypyFile | TypeInfo = tree

    for i, part in enumerate(parts):
        names: Optional[SymbolTable] = getattr(node, "names", None)
        if names is None:
            raise StubTypeError(
                f"Cannot descend into {type(node).__name__!r} "
                f"when resolving {qualname!r}"
            )

        sym: Optional[SymbolTableNode] = names.get(part)
        if sym is None or sym.node is None:
            raise StubTypeError(
                f"Name {part!r} not found while resolving {qualname!r}"
            )

        obj = sym.node

        # If this is the final segment, return object
        if i == len(parts) - 1:
            return obj

        # Otherwise, we expect a class (TypeInfo) so we can keep walking
        if not isinstance(obj, TypeInfo):
            raise StubTypeError(
                f"Non-class object {type(obj).__name__!r} encountered "
                f"while resolving {qualname!r}"
            )
        node = obj  # continue into nested class

    # Should be unreachable
    raise StubTypeError(f"Failed to resolve {qualname!r}")


# ---------------------------------------------------------------------------
# Converting mypy types to "get_type_hints-like" dicts
# ---------------------------------------------------------------------------

def _callable_type_hints(typ: CallableType) -> Dict[str, TypeRepr]:
    """
    Turn a mypy CallableType into a get_type_hints-style dict, but with
    TypeRepr objects instead of strings:

        {param_name: TypeRepr, ..., "return": TypeRepr}

    Note: Unlike typing.get_type_hints, we do *not* attempt to hide the
    first parameter for methods (self/cls). You can add that logic if
    you want to emulate get_type_hints on bound methods.
    """
    hints: Dict[str, TypeRepr] = {}

    for arg_name, arg_type in zip(typ.arg_names, typ.arg_types):
        if arg_name is None:
            # positional-only arg without a name; skip like get_type_hints
            continue
        hints[arg_name] = _type_to_repr(arg_type)

    hints["return"] = _type_to_repr(typ.ret_type)
    return hints


def _func_type_hints(node: FuncDef | OverloadedFuncDef) -> Dict[str, TypeRepr]:
    """
    Extract type hints for a function or method.

    Overloaded functions are simplified to the first alternative; you can
    adjust this to merge or expose all overloads if desired.
    """
    if node.type is None:
        raise StubTypeError(f"Function {node.name!r} has no inferred type")

    typ = get_proper_type(node.type)

    if isinstance(typ, Overloaded):
        # Pick the first alternative for simplicity
        first_alt = typ.items()[0]
        return _callable_type_hints(first_alt)

    if isinstance(typ, CallableType):
        return _callable_type_hints(typ)

    raise StubTypeError(f"Unexpected function type for {node.name!r}: {typ}")


def _class_type_hints(info: TypeInfo) -> Dict[str, TypeRepr]:
    """
    Extract class-level attribute annotations, similar to typing.get_type_hints(C).

    - We *do not* return method signatures here (matching typing.get_type_hints).
    - Only attributes with a known type (Var with .type) are returned.
    """
    hints: Dict[str, TypeRepr] = {}

    for name, sym in info.names.items():
        node = sym.node
        if isinstance(node, Var) and node.type is not None:
            hints[name] = _type_to_repr(node.type)

    return hints


def _node_type_hints(node, qualname: str) -> Dict[str, TypeRepr]:
    """
    Dispatch on node kind and return a get_type_hints-like dict.

    Supported:
        - FuncDef / OverloadedFuncDef  → param + return annotations
        - TypeInfo                     → class attribute annotations
        - Var                          → single-entry mapping for that name
    """
    if isinstance(node, (FuncDef, OverloadedFuncDef)):
        return _func_type_hints(node)

    if isinstance(node, TypeInfo):
        return _class_type_hints(node)

    if isinstance(node, Var) and node.type is not None:
        return {qualname: _type_to_repr(node.type)}

    raise StubTypeError(
        f"Unsupported node type for {qualname!r}: {type(node).__name__}"
    )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def get_type_hints_from_stub_file(
    stub_path: str | Path,
    qualname: str,
    *,
    module_name: str | None = None,
    python_version: Tuple[int, int] = (3, 12),
) -> Dict[str, TypeRepr]:
    """
    Rough equivalent of typing.get_type_hints(), but reading types from a .pyi file.

    Parameters
    ----------
    stub_path:
        Path to the .pyi file you want to analyze.
    qualname:
        Qualified name of the target inside the stub:
            "f"           for a top-level function
            "C"           for a class
            "C.m"         for a method
            etc.
    module_name:
        Logical module name to give this stub to mypy. Defaults to the
        file's stem (e.g. 'foo' for 'foo.pyi').
    python_version:
        Python version tuple passed to mypy (e.g. (3, 11)). Defaults to (3, 12).

    Returns
    -------
    Dict[str, TypeRepr]
        A mapping name → TypeRepr. For callables, includes "return".
    """
    path = Path(stub_path)
    if module_name is None:
        module_name = path.stem

    src = BuildSource(str(path), module_name, None)
    _result, tree = _build_for_sources([src], module_name, python_version=python_version)

    node = _lookup_qualname(tree, qualname)
    return _node_type_hints(node, qualname)


def get_type_hints_from_typeshed_module(
    module_name: str,
    qualname: str,
    *,
    python_version: Tuple[int, int] = (3, 12),
) -> Dict[str, TypeRepr]:
    """
    Rough equivalent of typing.get_type_hints(), but for modules loaded
    via mypy, which means:

        - standard library modules are read from Typeshed
        - third-party modules with .pyi are read from their stubs
        - plain .py modules are type-checked by mypy

    Parameters
    ----------
    module_name:
        Logical module name as Python would import it, e.g.:
            "typing"
            "collections.abc"
            "asyncio"
            "numpy"
    qualname:
        Qualified name inside that module:
            "f"
            "C"
            "C.m"
            etc.
    python_version:
        Python version tuple passed to mypy (e.g. (3, 11)). Defaults to (3, 12).

    Returns
    -------
    Dict[str, TypeRepr]
        A mapping name → TypeRepr. For callables, includes "return".
    """
    # BuildSource with file=None → mypy finds the module in sys.path / typeshed
    src = BuildSource(None, module_name, None)
    _result, tree = _build_for_sources([src], module_name, python_version=python_version)

    node = _lookup_qualname(tree, qualname)
    print(dir(node))
    return _node_type_hints(node, qualname)
