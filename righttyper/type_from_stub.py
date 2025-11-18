from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import typeshed_client as typeshed

import libcst as cst
import libcst.matchers as cstm
from libcst.metadata import MetadataWrapper, QualifiedNameProvider, QualifiedName


@dataclass(frozen=True)
class ResolvedType:
    """
    Representation of a type reference like `typing.List[int]`.

    - module: 'typing', 'collections.abc', 'mymodule', etc.
    - qualname: 'List', 'Mapping', 'MyClass', ...
    - args: generic arguments (for List[int], args = (ResolvedType(int),)).
    """
    module: Optional[str]
    qualname: str
    args: Tuple["ResolvedType", ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        base = f"{self.module}.{self.qualname}" if self.module else self.qualname
        if not self.args:
            return base
        inner = ", ".join(str(a) for a in self.args)
        return f"{base}[{inner}]"


@dataclass
class FunctionTypeHints:
    module: str                      # e.g., 'collections.abc'
    qualname: str                    # e.g., 'Iterable.__contains__'
    params: Dict[str, ResolvedType]  # param name -> type
    return_type: Optional[ResolvedType]


class StubTypeHintsCollector(cst.CSTVisitor):
    """
    Collects type hints from a stub file, roughly analogous to typing.get_type_hints.

    The result is a mapping from qualname within the module (e.g. "f", "C.m")
    to FunctionTypeHints.
    """
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.scope_stack: List[str] = []
        self.results: Dict[str, FunctionTypeHints] = {}

    # ----- Helpers for qualified names and type resolution -----

    def _current_qualname(self, name: str) -> str:
        if not self.scope_stack:
            return name
        return ".".join([*self.scope_stack, name])

    def _resolve_qname_of_expr(
        self, expr: cst.BaseExpression
    ) -> Optional[QualifiedName]:
        """
        Ask QualifiedNameProvider for the fully-qualified name of a Name/Attribute.
        Returns one QualifiedName (picks an arbitrary one if there are several).
        """
        try:
            qnames = self.get_metadata(QualifiedNameProvider, expr)  # type: ignore[arg-type]
        except KeyError:
            return None

        if not qnames:
            return None
        # In practice you might want to filter by source (LOCAL, IMPORT, BUILTIN, etc.)
        return next(iter(qnames))

    def _resolved_from_namelike(self, expr: cst.BaseExpression) -> ResolvedType:
        """
        Resolve a bare Name or Attribute to (module, qualname).
        Falls back to the stub's module_name when unsure.
        """
        qn = self._resolve_qname_of_expr(expr)
        if qn is None:
            # Best-effort fallback: just use the textual code.
            if isinstance(expr, cst.Name):
                return ResolvedType(self.module_name, expr.value)
            if isinstance(expr, cst.Attribute):
                return ResolvedType(self.module_name, expr.attr.value)
            return ResolvedType(None, cst.Module([]).code_for_node(expr))

        full = qn.name  # e.g. "typing.List", "builtins.int", "mypkg.mymod.MyClass"
        # Split into module and qualname (everything after the first component)
        parts = full.split(".")
        if len(parts) == 1:
            return ResolvedType(None, parts[0])

        module = parts[0]
        qualname = ".".join(parts[1:])
        return ResolvedType(module, qualname)

    def _resolve_type(self, expr: cst.BaseExpression) -> ResolvedType:
        """
        Convert an annotation expression into a ResolvedType tree.

        Handles:
          - Names / Attributes
          - Subscript (generics): List[int], dict[str, int], etc.
          - PEP 604 unions: int | str
          - String annotations: "Foo[int]"
          - Tuples inside subscription: dict[str, int]
        """
        print(f"{expr=}")
        # String-quoted annotation: "Foo[int]"
        if isinstance(expr, cst.SimpleString):
            # Strip quotes and parse as an expression again
            raw = expr.evaluated_value  # type: ignore[attr-defined]
            inner = cst.parse_expression(raw)
            return self._resolve_type(inner)

        # PEP 604 union: A | B
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.BitOr):
            left = self._resolve_type(expr.left)
            right = self._resolve_type(expr.right)
            # Encode as typing.Union[left, right]
            return ResolvedType("typing", "Union", args=(left, right))

        # Subscript: List[int], dict[str, int], etc.
        if isinstance(expr, cst.Subscript):
            base = self._resolve_type(expr.value)
            # Extract slice expressions
            slice_exprs: List[cst.BaseExpression] = []
            for sl in expr.slice:
                if isinstance(sl, cst.SubscriptElement) and isinstance(sl.slice, cst.Index):
                    val = sl.slice.value
                    # dict[str, int] uses a Tuple inside the single Index
                    if isinstance(val, cst.Tuple):
                        for el in val.elements:
                            if el.value is not None:
                                slice_exprs.append(el.value)
                    else:
                        slice_exprs.append(val)
            args = tuple(self._resolve_type(s) for s in slice_exprs)
            return ResolvedType(base.module, base.qualname, args=args)

        # Name or Attribute: plain type like int, MyClass, or module.Type
        if isinstance(expr, (cst.Name, cst.Attribute)):
            return self._resolved_from_namelike(expr)

        # Parenthesized type
        if isinstance(expr, cst.RemovalSentinel):
            raise AssertionError("Unexpected RemovalSentinel in annotation")

        if isinstance(expr, cst.UnaryOperation):
            # Unusual in type positions; fall back to textual name
            return ResolvedType(None, cst.Module([]).code_for_node(expr))

        if isinstance(expr, cst.Tuple):
            # For tuple annotations like tuple[int, str]
            args = tuple(self._resolve_type(e.value) for e in expr.elements if e.value)
            return ResolvedType("builtins", "tuple", args=args)

        # Fallback: just keep the source text as a "qualname".
        return ResolvedType(None, cst.Module([]).code_for_node(expr))

    # ----- CSTVisitor methods -----

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        qualname = self._current_qualname(node.name.value)

        params: Dict[str, ResolvedType] = {}

        def add_param(p: cst.Param) -> None:
            if p.annotation is not None:
                params[p.name.value] = self._resolve_type(p.annotation.annotation)

        for param in cstm.findall(node, cstm.Param()):
            add_param(param)

        # Return type
        if node.returns is not None:
            return_type = self._resolve_type(node.returns.annotation)
        else:
            return_type = None

        self.results[qualname] = FunctionTypeHints(
            module=self.module_name,
            qualname=qualname,
            params=params,
            return_type=return_type,
        )


def get_stub_type_hints(module_name: str) -> Dict[str, FunctionTypeHints]:
    """
    Parse a stub (or stub-like) source string and return a mapping
    qualname -> FunctionTypeHints.

    module_name should be the importable module name for this stub,
    e.g. 'collections.abc', 'mymod', etc.
    """

    path = typeshed.get_stub_file(module_name)
    module = cst.parse_module(path.read_text(encoding='utf-8'))
    wrapper = MetadataWrapper(module)
    collector = StubTypeHintsCollector(module_name)
    wrapper.visit(collector)
    return collector.results


def get_stub_type_hints_for(
    module_name: str,
    qualname: str,
) -> Optional[FunctionTypeHints]:
    """
    Convenience wrapper that returns type hints only for a single function/method
    identified by its qualname (e.g. 'f', 'C.m').
    """
    all_hints = get_stub_type_hints(module_name)
    return all_hints.get(qualname)
