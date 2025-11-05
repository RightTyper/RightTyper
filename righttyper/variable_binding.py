import libcst as cst
from typing import Any, cast
from libcst.metadata import BatchableMetadataProvider, QualifiedNameProvider
from libcst.helpers import get_full_name_for_node
import libcst.matchers as cstm
from dataclasses import dataclass, field
import collections.abc as abc


@dataclass
class VarBindings:
    """Results from VariableBindingProvider"""
    qualname: tuple[str, ...]                               # elements of (fully qualified) scope name
    defines_vars: set[str] = field(default_factory=set)     # set of variables first defined by this node
    self_name: str|None = None                              # local name of 'self', if any
    defines_attrs: set[str] = field(default_factory=set)    # set of attributes first defined by this node


@dataclass
class _AttrScope:
    known_attributes: set[str] = field(default_factory=set)


@dataclass
class _VarScope:
    known_variables: set[str] = field(default_factory=set)
    self_name: str|None = None
    not_locals: set[str] = field(default_factory=set)


def _iter_binder_names(target: cst.CSTNode) -> abc.Iterator[cst.Name|cst.Attribute]:
    if isinstance(target, (cst.Name, cst.Attribute)):
        yield target
    elif isinstance(target, cst.StarredElement):
        yield from _iter_binder_names(target.value)
    elif isinstance(target, (cst.Tuple, cst.List)):
        for e in target.elements:
            yield from _iter_binder_names(e.value)


class VariableBindingProvider(BatchableMetadataProvider[VarBindings]):
    """Computes the statement which binds each variable or object attribute."""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self) -> None:
        super().__init__()

        self._qualname_stack: list[str] = []             # elements of the fully qualified name
        self._scope_stack: list[_VarScope] = [_VarScope()] # variable scope
        self._attr_scope_stack: list[_AttrScope] = []     # (object) attribute scope
        self._node2result: dict[cst.CSTNode, VarBindings] = {}


    def _get_bindings(self, node: cst.CSTNode) -> VarBindings:
        if node not in self._node2result:
            bindings = self._node2result[node] = VarBindings(
                qualname=tuple(self._qualname_stack),
                self_name=self._scope_stack[-1].self_name
            )
            self.set_metadata(node, bindings)
            return bindings

        return self._node2result[node]


    def _decorator_in(self, decorator: cst.Decorator, names: set[str]):
        try:
            return bool(names & {
                qn.name
                for qn in self.get_metadata(QualifiedNameProvider, decorator)
            })
        except NameError:
            return False


    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._record_name(node, node.name)
        self._qualname_stack.append(node.name.value)
        self._attr_scope_stack.append(_AttrScope())
        self._scope_stack.append(_VarScope())

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._scope_stack.pop()
        self._attr_scope_stack.pop()
        self._qualname_stack.pop()


    def _record_name(self, node: cst.CSTNode, target: cst.CSTNode) -> None:
        if isinstance(target, cst.Name):
            name = target.value
            scope = self._scope_stack[-1]

            if name in scope.not_locals:
                return

            if name == scope.self_name:
                scope.self_name = None  # masks 'self'

            if name not in scope.known_variables:
                b = self._get_bindings(node)
                b.defines_vars.add(name)
                scope.known_variables.add(name)

        elif isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
            if target.value.value == self._scope_stack[-1].self_name:
                name = target.attr.value
                if name not in self._attr_scope_stack[-1].known_attributes:
                    b = self._get_bindings(node)
                    b.defines_attrs.add(name)
                    self._attr_scope_stack[-1].known_attributes.add(name)


    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        is_method = (
            bool(self._qualname_stack)
            and self._qualname_stack[-1] != '<locals>'
            and not any(
                self._decorator_in(decorator, {'builtins.staticmethod', 'builtins.classmethod'})
                for decorator in node.decorators
            )
        )

        parameters = cast(list[cst.Param], cstm.findall(node, cstm.Param()))

        # "inherit" self_name unless masked by a variable or parameter
        scope = _VarScope(self_name=self._scope_stack[-1].self_name)

        self._qualname_stack.append(node.name.value)
        self._qualname_stack.append('<locals>')
        self._scope_stack.append(scope)

        for param in parameters:
            self._record_name(param, param.name)

        if is_method and parameters:
            scope.self_name = parameters[0].name.value

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_stack.pop()
        self._qualname_stack.pop()
        self._qualname_stack.pop()


    def visit_Lambda(self, node: cst.Lambda) -> None:
        self._qualname_stack.append('<lambda>')
        self._scope_stack.append(_VarScope())

    def leave_Lambda(self, node: cst.Lambda) -> None:
        self._scope_stack.pop()
        self._qualname_stack.pop()


    def visit_Assign(self, node: cst.Assign) -> None:
        for t in node.targets:
            for n in _iter_binder_names(t.target):
                self._record_name(node, n)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._record_name(node, node.target)

    def visit_TypeAlias(self, node: cst.TypeAlias) -> None:
        for n in _iter_binder_names(node.name):
            self._record_name(node, n)

    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:
        self._record_name(node, cast(cst.BaseAssignTargetExpression, node.target))


    def visit_For(self, node: cst.For) -> None:
        for n in _iter_binder_names(node.target):
            self._record_name(node, n)

    def visit_With(self, node: cst.With) -> None:
        for item in node.items:
            if item.asname and isinstance(item.asname.name, cst.Name):
                self._record_name(node, item.asname.name)

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:
        if node.name and isinstance(node.name, cst.AsName):
            self._record_name(node, node.name.name)

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if alias.asname is not None:
                self._record_name(node, alias.asname.name)
            elif isinstance(alias.name, cst.Attribute):
                toplevel = cast(list[cst.Attribute], cstm.findall(alias.name, cstm.Attribute(value=cstm.Name())))
                if toplevel:
                    self._record_name(node, toplevel[0].value)
            else:
                self._record_name(node, alias.name)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, abc.Sequence):    # not cst.ImportStar (not supported)
            for alias in node.names:
                if alias.asname is not None:
                    self._record_name(node, alias.asname.name)
                else:
                    self._record_name(node, alias.name)

    def visit_MatchCase(self, node: cst.MatchCase) -> None:
        for p in cast(list[cst.MatchAs|cst.MatchStar|cst.MatchMapping],
                      cstm.findall(node.pattern,
                                   cstm.OneOf(cstm.MatchAs(), cstm.MatchStar(),
                                   cstm.MatchMapping()))
        ):
            if isinstance(p, cst.MatchMapping):
                if p.rest:
                    self._record_name(node, p.rest)
            elif p.name:
                self._record_name(node, p.name)

    def visit_Global(self, node: cst.Global) -> None:
        if self._qualname_stack:
            self._scope_stack[-1].not_locals |= {
                name.name.value
                for name in node.names
            }

    def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
        self._scope_stack[-1].not_locals |= {
            name.name.value
            for name in node.names
        }
