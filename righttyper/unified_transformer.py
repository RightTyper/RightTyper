import typing
import builtins
import collections.abc as abc
from dataclasses import dataclass
import libcst as cst
import libcst.matchers as cstm
from libcst.metadata import MetadataWrapper, PositionProvider, QualifiedNameProvider
from righttyper.variable_binding import VariableBindingProvider
import re

from righttyper.typeinfo import TypeInfo
from righttyper.righttyper_types import Filename, CodeId, FunctionName
from righttyper.annotation import FuncAnnotation, ModuleVars


ChangeStmtList: typing.TypeAlias = typing.Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement | cst.FunctionDef]

@dataclass(frozen=True)
class Change:
    """Describes a change made to the source code."""
    scope: str
    before: ChangeStmtList
    after: ChangeStmtList


    @staticmethod
    def _format_stmts(stmts: ChangeStmtList) -> str:
        """Formats change statements."""

        class FuncBodyRemover(cst.CSTTransformer):
            def leave_FunctionDef(
                self,
                original_node: cst.FunctionDef,
                updated_node: cst.FunctionDef
            ) -> cst.FunctionDef:
                return updated_node.with_changes(
                    body=cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])
                )

            def leave_Decorator(
                self,
                original_node: cst.Decorator,
                updated_node: cst.Decorator
            ) -> cst.RemovalSentinel:
                return cst.RemoveFromParent()

        bodyless = [
            typing.cast(cst.FunctionDef, it.visit(FuncBodyRemover())) if isinstance(it, cst.FunctionDef) else it
            for it in stmts
        ]

        # remove " ..." from the FunctionDef body deletion, if present
        return cst.Module(bodyless).code.strip().removesuffix(" ...")

    def format(self) -> tuple[str, str, str]:
        return (
            self.scope,
            Change._format_stmts(self.before),
            Change._format_stmts(self.after)
        )


_BUILTIN_TYPES : frozenset[str] = frozenset({
    t for t in (
        "None",
        *(name for name, value in builtins.__dict__.items() if isinstance(value, type))
    )
})

# FIXME this prevents us from missing out on well-known "typing." types,
# but is risky... change to receiving fully qualified names and simplifying
# them in context.
_TYPING_TYPES : frozenset[str] = frozenset({
    t for t in typing.__all__
})


# generated from https://docs.python.org/3/library/typing.html#deprecated-aliases
_DEPRECATED_TYPING_TYPES: typing.Final[tuple[tuple[str, str], ...]] = (
    ("typing.List",         "list"),
    ("typing.Dict",         "dict"),
    ("typing.Set",          "set"),
    ("typing.FrozenSet",    "frozenset"),
    ("typing.Tuple",        "tuple"),
    ("typing.Type",         "type"),
    ("typing.Text",         "str"),

    ("typing.DefaultDict",  "collections.defaultdict"),
    ("typing.OrderedDict",  "collections.OrderedDict"),
    ("typing.ChainMap",     "collections.ChainMap"),
    ("typing.Counter",      "collections.Counter"),
    ("typing.Deque",        "collections.deque"),

    ("typing.Pattern",      "re.Pattern"),
    ("typing.Match",        "re.Match"),

    ("typing.IO",           "io.IOBase"),
    ("typing.TextIO",       "io.TextIOBase"),
    ("typing.BinaryIO",     "io.BufferedIOBase"),

    ("typing.AbstractSet",   "collections.abc.Set"),
    ("typing.ByteString",    "collections.abc.ByteString"),
    ("typing.Collection",    "collections.abc.Collection"),
    ("typing.Container",     "collections.abc.Container"),
    ("typing.ItemsView",     "collections.abc.ItemsView"),
    ("typing.KeysView",      "collections.abc.KeysView"),
    ("typing.Mapping",       "collections.abc.Mapping"),
    ("typing.MappingView",   "collections.abc.MappingView"),
    ("typing.MutableMapping","collections.abc.MutableMapping"),
    ("typing.MutableSequence","collections.abc.MutableSequence"),
    ("typing.MutableSet",    "collections.abc.MutableSet"),
    ("typing.Sequence",      "collections.abc.Sequence"),
    ("typing.ValuesView",    "collections.abc.ValuesView"),
    ("typing.Coroutine",     "collections.abc.Coroutine"),
    ("typing.AsyncGenerator","collections.abc.AsyncGenerator"),
    ("typing.AsyncIterable", "collections.abc.AsyncIterable"),
    ("typing.AsyncIterator", "collections.abc.AsyncIterator"),
    ("typing.Awaitable",     "collections.abc.Awaitable"),
    ("typing.Iterable",      "collections.abc.Iterable"),
    ("typing.Iterator",      "collections.abc.Iterator"),
    ("typing.Callable",      "collections.abc.Callable"),
    ("typing.Generator",     "collections.abc.Generator"),
    ("typing.Hashable",      "collections.abc.Hashable"),
    ("typing.Reversible",    "collections.abc.Reversible"),
    ("typing.Sized",         "collections.abc.Sized"),
)

def _dotted_name_to_nodes(name: str) -> cst.Attribute | cst.Name:
    """Creates Attribute/Name to build a module name, dotted or not."""
    parts = name.split(".")
    if len(parts) == 1:
        return cst.Name(parts[0])
    base: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
        base = cst.Attribute(value=base, attr=cst.Name(part))
    return base

def _nodes_to_dotted_name(node: cst.Attribute|cst.Name|cst.BaseExpression) -> str:
    """Extracts a module name from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        return f"{_nodes_to_dotted_name(node.value)}.{_nodes_to_dotted_name(node.attr)}"

    assert isinstance(node, cst.Name), f"{node=}"
    return node.value

def _nodes_to_top_level_name(node: cst.Attribute|cst.Name|cst.BaseExpression) -> str:
    """Extracts the top-level name (e.g., 'foo' in 'foo.bar') from CST Attribute/Name nodes."""
    while isinstance(node, cst.Attribute):
        node = node.value

    assert isinstance(node, cst.Name), f"{node=}"
    return node.value

def _nodes_to_all_dotted_names(node: cst.Attribute|cst.Name|cst.BaseExpression) -> list[str]:
    """Extracts the list of all module and parent module names from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        names = _nodes_to_all_dotted_names(node.value)
        return [*names, f"{names[-1]}.{node.attr.value}"]

    assert isinstance(node, cst.Name), f"{node=}"
    return [node.value]

def _quote(s: str) -> str:
    s = s.replace('\\', '\\\\')
    return '"' + s.replace('"', '\\"') + '"'


class UnifiedTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider, QualifiedNameProvider, VariableBindingProvider)

    def __init__(
        self,
        filename: str,
        type_annotations: dict[CodeId, FuncAnnotation],
        module_variables: ModuleVars | None,
        module_name: str,
        *,
        override_annotations: bool = True,
        only_update_annotations: bool = False,
        inline_generics: bool = True,
    ) -> None:
        self.filename = filename
        self.type_annotations = type_annotations
        self.module_variables: dict[str, TypeInfo] = {}
        if module_variables:
            self.module_variables = {
                var_name: var_type
                for var_name, var_type in module_variables.variables
            }
        self.module_name = module_name
        self.override_annotations = override_annotations
        self.only_update_annotations = only_update_annotations
        self.inline_generics = inline_generics
        self.has_future_annotations = False
        self.change_list: list[Change] = []

        # TODO Ideally we'd use TypeInfo.module and avoid this as well as _module_for
        def iter_types(t: TypeInfo):
            yield t
            for arg in t.args:
                if type(arg) is TypeInfo:
                    yield from iter_types(arg)

        self.name2module: dict[str, str] = {
            t.fullname(): t.module
            for ann in type_annotations.values()
            for root in [arg[1] for arg in ann.args] + [ann.retval] + [var[1] for var in ann.variables]
            for t in iter_types(root)
        } | {
            t.fullname(): t.module
            for t in self.module_variables.values()
        }

    def _module_for(self, name: str) -> tuple[str, str]:
        """Splits a dot name in its module and qualified name parts."""
        if name.startswith("builtins."):
            return 'builtins', name[9:]

        if (m := self.name2module.get(name)):
            return m, name[len(m)+1:]

        return '', name

    def _rename_types(self, annotation: TypeInfo) -> TypeInfo:
        """Renames types in an annotation based on the module name and on any aliases."""

        class Renamer(TypeInfo.Transformer):
            def visit(tt, node: TypeInfo) -> TypeInfo:
                node = super().visit(node)

                if node.module in ('', 'builtins') and node.name in _BUILTIN_TYPES:
                    if node.name not in self.used_names[-1]:
                        return node.replace(module='')  # don't need the module

                    # somebody overrode a builtin name(!), qualify it
                    node = node.replace(module='builtins') 

                # use local names where possible (shorter)
                if a := self.aliases.get(node.fullname()):
                    return node.replace(module='', name=a)

                if node.module:
                    if node.module == self.module_name:
                        # Use local name if possible; when <locals> are
                        # involved, this may be the only way to refer to it
                        name_parts = node.name.split('.')
                        if name_parts[:-1] == self.name_stack:
                            return node.replace(module='', name=name_parts[-1])

                        return node.replace(module='')  # it's us, refer by (global) name

                    # use local names for the module where possible (shorter)
                    if a := self.aliases.get(node.module):
                        return node.replace(module=a)

                    # does the module's first part conflict with other definitions?
                    if node.module.split('.')[0] in self.used_names[-1]:
                        alias = "_rt_" + "_".join(node.module.split("."))
                        self.if_checking_aliases[node.module] = alias
                        return node.replace(module=alias)

                return node

        return Renamer().visit(annotation)


    def _process_generics(
        self,
        ann: FuncAnnotation,
        used_inline_names: set[str]
    ) -> tuple[FuncAnnotation, dict[int, TypeInfo]]:
        """Transforms an annotation, defining type variables and using them."""
        used_inline_names = set(used_inline_names)

        class RenameGenericsTransformer(TypeInfo.Transformer):
            def __init__(vself):
                vself.generics = {}
            
            def visit(vself, node: TypeInfo) -> TypeInfo:
                if not node.typevar_index:
                    return super().visit(node)

                if node.typevar_index not in vself.generics:
                    if self.inline_generics:
                        i = 0
                        while (name := f"T{node.typevar_index + i}") in used_inline_names:
                            i += 1
                        used_inline_names.add(name)
                    else:
                        while (name := f"rt_T{self.module_generic_index}") in self.used_names[-1]:
                            self.module_generic_index += 1
                        self.module_generic_index += 1

                    vself.generics[node.typevar_index] = node.replace(typevar_name=name)

                return vself.generics[node.typevar_index]

        tr = RenameGenericsTransformer()
        updated_ann = FuncAnnotation(
            [(name, tr.visit(ti)) for name, ti in ann.args],
            tr.visit(ann.retval),
            varargs=ann.varargs, kwargs=ann.kwargs,
            variables=[(name, tr.visit(ti)) for name, ti in ann.variables],
        )
        
        return (updated_ann, tr.generics)
                    
    def _qualified_name_in(self, decorator: cst.CSTNode, names: set[str]):
        try:
            return bool(names & {
                qn.name
                for qn in self.get_metadata(QualifiedNameProvider, decorator)
            })
        except NameError:
            return False

    def visit_Module(self, node: cst.Module) -> bool:
        # Initialize mutable members here, just in case transformer gets reused

        # names used somewhere in a module or class scope
        self.used_names: list[set[str]] = [used_names(node)]

        # stack of known names
        self.known_names: list[set[str]] = [set(_BUILTIN_TYPES)]

        # global aliases from 'from .. import ..' and 'import .. as ..';
        # maps from qualified name to local name
        self.aliases: dict[str, str] = {
            # we will add any such "typing." names, so add them as aliases
            f"typing.{t}": t
            for t in _TYPING_TYPES
        }

        # We automatically import 'typing' types, so for now it is convenient
        # to continue using 'typing', by default (unless imported)
        for old, new in _DEPRECATED_TYPING_TYPES:
            if old.startswith('typing.') and new.startswith('collections.abc'):
                self.aliases[new] = old[7:]

        # "import ... as ..." within "if TYPE_CHECKING:".
        # TODO read those in as well so as not to duplicate imports
        self.if_checking_aliases: dict[str, str] = dict()

        # modules imported under their own names
        self.imported_modules: set[str] = set()

        self.unknown_types : set[str] = set()
        self.name_stack : list[str] = []

        self.has_future_annotations = any(
            True for match in cstm.findall(
                node,
                cstm.ImportFrom(
                    module=cstm.Name("__future__"),
                    names=[
                        cstm.ImportAlias(
                            name=cstm.Name("annotations")
                        )
                    ]
                )
            )
        )

        self.module_generic_index = 1

        # Since overloads must be consecutive, we don't need to keep track of
        # multiple concurrent overloaded functions within each namespace.

        # The current list of overloads that have been collected in each scope.
        # Note that this is a stack since namespaces can be nested
        # (e.g. classes).
        self.overload_stack: list[list[cst.FunctionDef]] = [[]]
        # The current list of overloaded function names that are in each scope.
        self.overload_name_stack: list[str] = [""]

        # The annotation for the current function, if any
        self.func_ann_stack: list[FuncAnnotation|None] = [None]

        # Whether to annotate variables in this scope
        self.annotate_vars_stack: list[bool] = [True]

        # A map indicating whether classes seen in this module inherit from enum.Enum
        self.class_is_enum: dict[str, bool] = dict()

        # The set of variable assignments we're modifying
        self.modified_assignments: set[cst.CSTNode] = set()
        return True


    def visit_If(self, node: cst.If) -> bool:
        if cstm.matches(node, cstm.If(test=cstm.Name("TYPE_CHECKING"))):
            return False    # to ignore imports within TYPE_CHECKING

        return True


    def visit_Import(self, node: cst.Import) -> bool:
        if not self.name_stack: # for now, we only handle global imports
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    if alias.asname is not None:
                        self.known_names[-1].add(_nodes_to_top_level_name(alias.asname.name))
                        self.aliases[_nodes_to_dotted_name(alias.name)] = _nodes_to_dotted_name(alias.asname.name)
                    else:
                        self.imported_modules |= set(_nodes_to_all_dotted_names(alias.name))
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        if not self.name_stack: # for now, we only handle global imports
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    self.known_names[-1].add(
                        _nodes_to_top_level_name(
                            alias.asname.name if alias.asname is not None else alias.name
                        )
                    )

                    source: list[str] = []

                    if node.relative:
                        if self.module_name is None:
                            # TODO warn: cannot handle import: module name unknown
                            return False

                        source = self.module_name.split('.')[:-len(node.relative)]
                        if not source:
                            # TODO warn: invalid import (goes out of package space)
                            return False

                    source += _nodes_to_dotted_name(node.module).split('.') if node.module else []

                    self.aliases[f"{'.'.join(source)}.{_nodes_to_dotted_name(alias.name)}"] = \
                        _nodes_to_dotted_name(
                            alias.asname.name if alias.asname is not None else alias.name
                        )

        return False


    def _get_var(self, target: cst.Name|cst.Attribute) -> TypeInfo | None:
        if (func_scope_index := list_rindex(self.name_stack, '<locals>')):
            func_scope_index = len(self.name_stack) + func_scope_index + 1

        qualname = ".".join(self.name_stack[func_scope_index:] + [_nodes_to_dotted_name(target)])
        if func_scope_index:
            if (ann := self.func_ann_stack[-1]):
                return next(
                    (
                        var_type
                        for var_name, var_type in ann.variables
                        if var_name == qualname
                    ), None
                )
        else:
            return self.module_variables.get(qualname)

        return None


    def visit_Assign(self, node: cst.Assign) -> bool:
        for t in node.targets:
            if isinstance(t.target, cst.Name):
                self.known_names[-1].add(t.target.value)
            elif isinstance(t.target, cst.Tuple):
                for el in t.target.elements:
                    if isinstance(el.value, cst.Name):
                        self.known_names[-1].add(el.value.value)
        return True


    def _is_typing_construct(self, assign_value: cst.BaseExpression | None) -> bool:
        if isinstance(assign_value, cst.Call):
            try:
                return any(
                    qn.name in ('collections.namedtuple', 'typing.NamedTuple',
                                'typing.TypeVar', 'typing.ParamSpec', 'typing.TypeVarTuple',
                                'typing.NewType')
                    for qn in self.get_metadata(QualifiedNameProvider, assign_value.func)
                )
            except NameError:
                pass

        elif isinstance(assign_value, cst.Subscript):
            try:
                return any(
                    qn.name in ('typing.Literal', 'typing.Annotated')
                    for qn in self.get_metadata(QualifiedNameProvider, assign_value.value)
                )
            except NameError:
                pass

        return False


    def _record_var_change(self,
        old: cst.Assign|cst.AnnAssign|cst.TypeAlias,
        new: cst.Assign|cst.AnnAssign|cst.TypeAlias
    ) -> None:
        self.change_list.append(Change(
            ".".join(self.name_stack) if self.name_stack else '<module>',
            [cst.SimpleStatementLine([old])],
            [cst.SimpleStatementLine([new])]
        ))
        self.modified_assignments.add(new)


    def _seems_type_like(self, expr: cst.BaseExpression | None) -> bool:
        ALLOWED_NODES = (
            # Base identifiers and attributes
            cst.Name, cst.Attribute,
            # Literals and constants
            cst.SimpleString, cst.Integer, cst.Float, cst.Imaginary, cst.Ellipsis,
            # Subscripted types and tuples
            cst.Subscript, cst.Index, cst.Element, cst.Tuple, cst.StarredElement,
            # Bitwise union / binary ops
            cst.BitOr, cst.BinaryOperation,
            # Expression wrapper
            cst.Expr,
        )

        return isinstance(expr, ALLOWED_NODES) and all(
            self._seems_type_like(child)
            for child in expr.children
            if isinstance(child, cst.BaseExpression)
        )


    def _node_defines(self, node: cst.CSTNode, target: cst.BaseExpression) -> cst.Name|cst.Attribute|None:
        """Returns the target iff the given node is first to define it.
           Note that currently the target must be cstm.OneOf(cstm.Name(), cstm.Atttribute(value=cstm.Name()))"""
        try:
            if isinstance(target, cst.Name):
                binding = self.get_metadata(VariableBindingProvider, node)
                return target if target.value in binding.defines_vars else None

            if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
                binding = self.get_metadata(VariableBindingProvider, node)
                return target if (
                    target.value.value == binding.self_name
                    and target.attr.value in binding.defines_attrs
                ) else None
        except:
            pass

        return None


    def _handle_assign(
        self,
        node: cst.Assign|cst.AnnAssign|cst.TypeAlias,
        node_target: cst.BaseExpression,
        annotation: cst.Annotation|None
    ) -> cst.Assign|cst.AnnAssign|cst.TypeAlias|None:
        if (
            not self.annotate_vars_stack[-1]
            or not (target := self._node_defines(node, node_target))
            or not (var_type := self._get_var(target))
        ):
            return None

        new_node: cst.AnnAssign|cst.Assign|cst.TypeAlias

        if node.value and self._is_typing_construct(node.value):
            new_node = cst.Assign(
                targets=[cst.AssignTarget(target)],
                value=node.value
            )
            self._record_var_change(node, new_node)
            return new_node

        # Type-valued definitions present a special challenge: the 'TypeAlias' annotation
        # introduced in Python 3.10 now leads to mypy (1.18.2) errors, as users are
        # encouraged to move to Python 3.12's 'type' statement.  Even when supported,
        # converting every type-valued assignment to 'type' doesn't work in general,
        # as the resulting TypeAliasType does not support all that the assignment does
        # (for example, using the name as a class base or calling to instantiate an object).
        # The most compatible course of action seems to be to forgo annotating them.
        # In non-global (i.e., class and function) scope, using 'type' or a 'TypeAlias'
        # annotation is required (by mypy) to use the name as a type; we are here just
        # careful not to remove a previous annotation.

        if var_type.fullname() == 'type':
            return None

#        if var_type.fullname() == 'type' and self._seems_type_like(node.value):
#            # Our type map, based only on dynamic analysis, may use local variables
#            # pointing to a type as type names... for that reason, if the assignment
#            # looks type-like, declare it a TypeAlias.
#
#            assert node.value is not None # implicitly checked by seems_type_like
#
#            if self.use_type_keyword and isinstance(target, cst.Name):
#                new_node = cst.TypeAlias(
#                    name=target,
#                    value=node.value,
#                )
#                self._record_var_change(node, new_node)
#                return new_node
#
#            var_type = TypeInfo('typing', 'TypeAlias')

        # Don't throw away (presumably human-generated) "Final" and "ClassVar" annotations
        wrappers = []
        if annotation:
            # TODO this only handles top-level Final/ClassVar
            expr: cst.BaseExpression|None = annotation.annotation
            while (
                (
                    isinstance(name := expr, cst.Name)
                    or (isinstance(expr, cst.Subscript) and isinstance(name := expr.value, cst.Name))
                )
                and self._qualified_name_in(name, {'typing.Final', 'typing.ClassVar'})
            ):
                wrappers.append(name)
                expr = (
                    expr.slice[0].slice.value
                    if (isinstance(expr, cst.Subscript) and isinstance(expr.slice[0].slice, cst.Index))
                    else None
                )

        if not (expr := self._get_annotation_expr(var_type)):
            return None

        for w in reversed(wrappers):
            expr = cst.Subscript(
                value=w,
                slice=[cst.SubscriptElement(cst.Index(expr))]
            )

        new_node = cst.AnnAssign(
            target=target,
            annotation=cst.Annotation(annotation=expr),
            value=node.value
        )
        self._record_var_change(node, new_node)
        return new_node


    def leave_Assign(
        self,
        node: cst.Assign,
        updated_node: cst.Assign
    ) -> cst.Assign|cst.AnnAssign|cst.TypeAlias:
        if (
            len(node.targets) == 1  # no a = b = ...
            and not self.only_update_annotations
            and (new_node := self._handle_assign(node, node.targets[0].target, annotation=None))
        ):
            return new_node

        return updated_node


    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        if isinstance(node.target, cst.Name):
            self.known_names[-1].add(node.target.value)
        return True

    def leave_AnnAssign(
        self,
        node: cst.AnnAssign,
        updated_node: cst.AnnAssign
    ) -> cst.AnnAssign|cst.Assign|cst.TypeAlias:
        if (
            (self.override_annotations or self.only_update_annotations)
            and (new_node := self._handle_assign(node, node.target, node.annotation))
        ):
            return new_node

        return updated_node


    def visit_TypeAlias(self, node: cst.TypeAlias) -> bool:
        if isinstance(node.name, cst.Name):
            self.known_names[-1].add(node.name.value)
        return True

#    def leave_TypeAlias(
#        self,
#        node: cst.TypeAlias,
#        updated_node: cst.TypeAlias
#    ) -> cst.TypeAlias|cst.Assign|cst.AnnAssign:
#        if (new_node := self._handle_assign(node, node.name)):
#            return new_node
#
#        return updated_node


    def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
        if isinstance(node.target, cst.Name):
            self.known_names[-1].add(node.target.value)
        return True


    def leave_SimpleStatementLine(
        self,
        node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        return (
            typing.cast(cst.SimpleStatementLine, updated_node.visit(TypeHintDeleter()))
            if any(stmt in self.modified_assignments for stmt in updated_node.body)
            else updated_node
        )


    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        name_source = list_rindex(self.name_stack, '<locals>') # neg. index of last function, or 0 (for globals)
        self.name_stack.append(node.name.value)
        self.used_names.append(self.used_names[name_source] | used_names(node))
        self.overload_stack.append([])
        self.overload_name_stack.append("")
        self.known_names.append(set(self.known_names[0]))   # current globals ([0]) are also known

        name = '.'.join(self.name_stack)
        is_enum = any(
            qn.name.startswith('enum.') or self.class_is_enum.get(qn.name, False)
            for base in node.bases
            if base.keyword is None
            for qn in self.get_metadata(QualifiedNameProvider, base.value)
        )
        is_dataclass = any(
            self._qualified_name_in(decorator, {'dataclasses.dataclass'})
            for decorator in node.decorators
        )

        self.class_is_enum[name] = is_enum
        self.annotate_vars_stack.append(not (is_enum or is_dataclass))
        return True

    def leave_ClassDef(self, orig_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.annotate_vars_stack.pop()
        # a class is known once its definition is done
        self.known_names.pop()
        self.known_names[-1].add(self.name_stack[-1])
        self.name_stack.pop()
        self.used_names.pop()
        self.overload_stack.pop()
        self.overload_name_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name_source = list_rindex(self.name_stack, '<locals>') # neg. index of last function, or 0 (for globals)
        self.name_stack.extend([node.name.value, "<locals>"])
        self.used_names.append(self.used_names[name_source] | used_names(node))
        self.overload_stack.append([])
        self.overload_name_stack.append("")
        self.known_names.append(set(self.known_names[0]))   # current globals ([0]) are also known

        name = ".".join(self.name_stack[:-1])
        first_line = min(
            self.get_metadata(PositionProvider, node).start.line
            for node in (node, *node.decorators)
        )
        key = CodeId(Filename(self.filename), FunctionName(name), first_line, 0)
        self.func_ann_stack.append(self.type_annotations.get(key))
        self.annotate_vars_stack.append(True)
        return True

    def _get_annotation_expr(self, annotation: TypeInfo) -> cst.BaseExpression | None:
        annotation = self._rename_types(annotation)

        try:
            annotation_expr: cst.BaseExpression = cst.parse_expression(str(annotation))
        except cst.ParserSyntaxError:
            return None # result would be invalid; most likely it contains "<locals>"

        class UnknownTypeExtractor(TypeInfo.Transformer):
            def __init__(me):
                me.types = set()

            def visit(me, node: TypeInfo) -> TypeInfo:
                if (
                    node.typevar_name is None                       # we'll define these
                    and not (node.is_union() or node.is_list())
                    and node.fullname().split('.')[0] not in self.known_names[-1]
                    and node.module not in self.imported_modules
                ):
                    me.types.add(node.fullname())

                return super().visit(node)
        
        unknown = UnknownTypeExtractor()
        unknown.visit(annotation)
        self.unknown_types |= unknown.types 

        if not self.has_future_annotations and (unknown.types - _TYPING_TYPES):
            annotation_expr = cst.SimpleString(_quote(str(annotation)))

        return annotation_expr
    
    def _process_parameter(self, parameter: cst.Param, ann: FuncAnnotation) -> cst.Param:
        """Processes a parameter, either returning an updated parameter or the original one."""
        if self.only_update_annotations and parameter.annotation is None:
            return parameter

        if (
            not (parameter.annotation is None or self.override_annotations)
            or (annotation := next(
                (
                    annotation for arg, annotation in ann.args
                    if arg == parameter.name.value
                ), None)
            ) is None
        ):
            return parameter

        if (
            annotation.fullname() == "typing.Any"
            or not (annotation_expr := self._get_annotation_expr(annotation))
        ):
            new_par = parameter.with_changes(annotation=None)
        else:
            new_par = parameter.with_changes(
                annotation=cst.Annotation(annotation=annotation_expr)
            )

        new_par = typing.cast(cst.Param, new_par.visit(TypeHintDeleter()))
        return new_par

    def leave_FunctionDef(
            self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef | cst.FlattenSentinel | cst.RemovalSentinel:
        func_name = ".".join(self.name_stack[:-1])
        self.annotate_vars_stack.pop()
        self.name_stack.pop()
        self.name_stack.pop()
        self.used_names.pop()
        self.overload_stack.pop()
        self.overload_name_stack.pop()
        self.known_names.pop()

        # If we encounter a function whose name doesn't match the current
        # overload sequence, we discard the overload stack and update the
        # current overload sequence.
        # This also has the side effect of cleaning up orphan overload
        # signatures, which is good.
        if func_name != self.overload_name_stack[-1]:
            self.overload_stack[-1] = []
            self.overload_name_stack[-1] = func_name

        is_overload = any(
            self._qualified_name_in(decorator, {'typing.overload'})
            for decorator in original_node.decorators
        )

        # If our function is an overload signature, we append it to the overload list.
        # NOTE: This check technically misses if @overload is aliased or used as
        # the result of an expression, but not even mypy handles that.
        if is_overload:
            self.overload_stack[-1].append(original_node)
            return cst.RemoveFromParent()

        if ann := typing.cast(FuncAnnotation, self.func_ann_stack.pop()):  # cast to make mypy happy
            pre_function: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
            argmap: dict[str, TypeInfo] = {aname: atype for aname, atype in ann.args}
            overloads = self.overload_stack[-1]
            self.overload_stack[-1] = []

            retained_name_annotations = {   # if a typevar is retained, it'll be in this set
                name.value
                for par in typing.cast(typing.Iterator[cst.Param],
                                       cstm.findall(updated_node.params,
                                                    cstm.Param(annotation=cstm.Annotation())))
                if par.annotation is not None
                for name in typing.cast(typing.Iterator[cst.Name],
                                        cstm.findall(par.annotation, cstm.Name()))
                if not self.override_annotations or par.name.value not in argmap
            }

            if updated_node.returns is not None and (not self.override_annotations or ann.retval is None):
                retained_name_annotations |= {
                    name.value
                    for name in typing.cast(typing.Iterator[cst.Name],
                                            cstm.findall(updated_node.returns, cstm.Name()))
                }

            del argmap

            # We don't yet support updating overloads; if they are present and not being removed, leave the function alone.
            if overloads == [] or self.override_annotations:
                ann, generics = self._process_generics(ann, retained_name_annotations)

                if self.inline_generics:
                    type_params = [
                        tpar
                        for tpar in typing.cast(typing.Iterator[cst.TypeParam],
                                                cstm.findall(updated_node,
                                                             cstm.TypeParam(param=cstm.TypeVar())))
                        if tpar.param.name.value in retained_name_annotations
                    ]

                    # add type parameters
                    for generic in generics.values():
                        assert all(isinstance(arg, TypeInfo) for arg in generic.args)
                        type_params.append(cst.TypeParam(param=cst.TypeVar(
                            name=cst.Name(value=str(generic)),
                            bound=cst.Tuple(elements=[
                                cst.Element(value=expr)
                                for arg in generic.args
                                if isinstance(arg, TypeInfo)
                                if (expr := self._get_annotation_expr(arg))
                            ])
                        )))

                    if type_params:
                        updated_node = updated_node.with_changes(type_parameters=cst.TypeParameters(
                            params=type_params
                        ))
                    else:
                        updated_node = updated_node.with_changes(type_parameters=None)

                else:
                    # define typevars
                    for generic in generics.values():
                        assert generic.typevar_name is not None
                        assert all(isinstance(arg, TypeInfo) for arg in generic.args)
                        pre_function.append(cst.SimpleStatementLine(body=[
                            cst.Assign(targets=[cst.AssignTarget(target=cst.Name(generic.typevar_name))],
                                       value=cst.Call(func=cst.Name("TypeVar"), args=[
                                           cst.Arg(value=cst.SimpleString(_quote(str(generic)))),
                                           *(cst.Arg(value=expr)
                                             for arg in generic.args
                                             if isinstance(arg, TypeInfo)
                                             if (expr := self._get_annotation_expr(arg))
                                            )
                                       ])
                            )
                        ]))
                        self.unknown_types.add("TypeVar")

                # Now update the parameters
                class ParamChanger(cst.CSTTransformer):
                    def leave_Param(vself, node: cst.Param, updated_node: cst.Param) -> cst.Param:
                        return self._process_parameter(updated_node, ann)

                updated_node = updated_node.with_changes(params=updated_node.params.visit(ParamChanger()))

                should_update_ret = (
                    self.override_annotations
                    or (self.only_update_annotations and updated_node.returns is not None)
                    or (not self.only_update_annotations and updated_node.returns is None)
                )
                if should_update_ret:
                    if (
                        ann.retval.fullname() == "typing.Any"
                        or not (annotation_expr := self._get_annotation_expr(ann.retval))
                    ):
                        updated_node = updated_node.with_changes(returns=None)
                    else:
                        updated_node = updated_node.with_changes(
                            returns=cst.Annotation(annotation=annotation_expr),
                        )

                    if updated_node.body.body:
                        # delete any type hint in empty 1st body line
                        updated_node = updated_node.with_changes(
                            body=updated_node.body.with_changes(
                                body=[
                                    updated_node.body.body[0].visit(TypeHintDeleter()),
                                    *updated_node.body.body[1:]
                                ]
                            )
                        )

                if hasattr(updated_node.body, 'header'):
                    # delete any type hint in the same line as the 'def'
                    updated_node = updated_node.with_changes(
                        body=updated_node.body.with_changes(
                            header=updated_node.body.header.visit(TypeHintDeleter())
                        )
                    )

            original_overloads = overloads
            if self.override_annotations:
                overloads = []      # overloads are annotations, so wipe them
                # TODO Generate new overloads.

            pre_function.extend(overloads)

            if pre_function:
                leading_lines = updated_node.leading_lines
                first_comment = next((
                        i
                        for i, el in enumerate(leading_lines)
                        if isinstance(el, cst.EmptyLine) and el.comment is not None
                    ), None)
                if first_comment is not None:
                    pre_function[0] = pre_function[0].with_changes(leading_lines=leading_lines[:first_comment])
                    updated_node = updated_node.with_changes(leading_lines=leading_lines[first_comment:])

            self.change_list.append(Change(
                func_name,
                original_overloads + [original_node],
                pre_function + [updated_node]
            ))
            
            return cst.FlattenSentinel([*pre_function, updated_node]) if pre_function else updated_node

        overloads = self.overload_stack[-1]
        self.overload_stack[-1] = []
        return cst.FlattenSentinel([*overloads, updated_node]) if overloads else updated_node

    # ConstructImportTransformer logic
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        # Collect `from __future__` imports and remove them
        future_imports: list[cst.BaseStatement] = []
        new_body: list[cst.BaseStatement] = []
        stmt: cst.BaseStatement

        from_future = cstm.SimpleStatementLine(
            body=[cstm.ImportFrom(
                module=cstm.Name("__future__")
            )]
        )

        for stmt in updated_node.body:
            if cstm.matches(stmt, from_future):
                # Collect future imports to lift later
                future_imports.append(stmt)
            else:
                new_body.append(stmt)

        missing_modules = {
            mod
            for mod in (
                self._module_for(t)[0]
                for t in self.unknown_types - _TYPING_TYPES
                if '.' in t
                and t.split('.')[0] not in self.known_names[-1]
            )
            if mod != ''
        }

        def stmt_index(body: list[cst.BaseStatement], pattern: cstm.BaseMatcherNode) -> int|None:
            for i, stmt in enumerate(body):
                if cstm.matches(stmt, pattern):
                    return i

            return None

        def find_beginning(body: list[cst.BaseStatement]) -> int:
            for i, stmt in enumerate(body):
                if not cstm.matches(stmt,
                    cstm.OneOf(
                        cstm.EmptyLine(),
                        cstm.SimpleStatementLine(
                            body=[cstm.Expr(cstm.SimpleString())]
                        )
                    )
                ):
                    return i

            return 0

        if_type_checking_position = stmt_index(new_body, cstm.If(
                test=cstm.Name('TYPE_CHECKING'),
                body=cstm.IndentedBlock()
            )
        )

        # Add additional type checking imports if needed
        if missing_modules or self.if_checking_aliases:
            existing_body = [*(typing.cast(cst.If, new_body[if_type_checking_position]).body.body
                               if if_type_checking_position is not None
                               else ())]

            # TODO delete modules already imported

            new_stmt: cst.BaseStatement = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(
                    body=existing_body + [
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(_dotted_name_to_nodes(m))])
                        ])
                        for m in sorted(missing_modules)
                    ] + [
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(
                                name=_dotted_name_to_nodes(m),
                                asname=cst.AsName(cst.Name(a))
                            )])
                        ])
                        for m, a in sorted(self.if_checking_aliases.items())
                    ]
                )
            )

            if if_type_checking_position is not None:
                new_body[if_type_checking_position] = new_stmt
            else:
                if_type_checking_position = find_beginning(new_body)
                new_body.insert(if_type_checking_position, new_stmt)

            if 'TYPE_CHECKING' not in self.known_names[-1]:
                self.unknown_types.add('TYPE_CHECKING')

        # Emit "from typing import ..."
        if (typing_types := (self.unknown_types & _TYPING_TYPES)):
            existing = stmt_index(new_body, cstm.SimpleStatementLine(
                    body=[cstm.ImportFrom(module=cstm.Name("typing"))]
                )
            )

            if existing is not None:
                for alias in cstm.findall(new_body[existing], cstm.ImportAlias()):
                    typing_types.add(_nodes_to_dotted_name(typing.cast(cst.ImportAlias, alias).name))

            new_stmt = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names = [
                            cst.ImportAlias(name=_dotted_name_to_nodes(t))
                            # sort 'TYPE_CHECKING' first just because it seems neater that way
                            for t in sorted(typing_types, key=lambda x: (x != 'TYPE_CHECKING', x))
                        ]
                    )
                ]
            )

            if (existing is not None
                and (if_type_checking_position is None
                     or existing < if_type_checking_position)
            ):
                new_body[existing] = new_stmt
            else: 
                position = if_type_checking_position if if_type_checking_position is not None \
                           else find_beginning(new_body)
                new_body.insert(position, new_stmt)


        b = find_beginning(new_body)
        new_body[b:b] = future_imports

        return updated_node.with_changes(body=new_body)


    def get_changes(self: typing.Self) -> list[tuple[str, str, str]]:
        return [
            change.format()
            for change in self.change_list
        ]


    def transform_code(self: typing.Self, code: cst.Module) -> cst.Module:
        """Applies this transformer to a module."""
        wrapper = MetadataWrapper(code)
        return wrapper.visit(self)


def types_in_annotation(annotation: cst.BaseExpression) -> set[str]:
    """Extracts all type names included in a type annotation."""

    class TypeNameExtractor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Name(self, node: cst.Name) -> bool:
            self.names.add(node.value)
            return False 

        def visit_Attribute(self, node: cst.Attribute) -> bool:
            self.names.add(_nodes_to_dotted_name(node))
            return False 

    extractor = TypeNameExtractor()
    annotation.visit(extractor)
    return extractor.names


def used_names(node: cst.Module|cst.ClassDef|cst.FunctionDef) -> set[str]:
    """Extracts the names in a module or class."""

    # FIXME handle 'global' and 'nonlocal'

    names: set[str] = set()

    class Extractor(cst.CSTVisitor):
        def __init__(self):
            self.in_scope = False

        def visit_Module(self, node: cst.Module) -> bool:
            self.in_scope = True
            return True

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            if self.in_scope:
                names.add(node.name.value)
                return False

            self.in_scope = True
            return True

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            if self.in_scope:
                names.add(node.name.value)
                return False

            self.in_scope = True
            return True

        def visit_Assign(self, node: cst.Assign) -> bool:
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    names.add(t.target.value)
                elif isinstance(t.target, cst.Tuple):
                    for el in t.target.elements:
                        if isinstance(el.value, cst.Name):
                            names.add(el.value.value)
            return True

        def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
            if isinstance(node.target, cst.Name):
                names.add(node.target.value)
            return True

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            # node.names could also be cst.ImportStar
            if isinstance(node.names, abc.Sequence):
                for alias in node.names:
                    if not alias.asname:
                        names.add(_nodes_to_top_level_name(alias.name))
            return True

        def visit_NamedExpr(self, node: cst.NamedExpr) -> bool:
            names.add(_nodes_to_top_level_name(node.target))
            return True

        def visit_AsName(self, node: cst.AsName) -> bool:
            if isinstance(node.name, (cst.Tuple, cst.List)):
                for el in node.name.elements:
                    names.add(_nodes_to_top_level_name(el.value))
            else:
                names.add(_nodes_to_top_level_name(node.name))

            return True

    node.visit(Extractor())
    return names


def list_rindex(lst: list, item: object) -> int:
    """Returns either a negative index for the last occurrence of 'item' on the list,
       or 0 if not found."""
    try:
        return -1 - lst[::-1].index(item)
    except ValueError:
        return 0


class TypeHintDeleter(cst.CSTTransformer):
    """Deletes type hint comments."""

    _TYPE_HINT_COMMENT = re.compile(r"\s*#\s+type:\s.*")


    def leave_EmptyLine(self, node: cst.EmptyLine, updated: cst.EmptyLine) -> cst.EmptyLine|cst.RemovalSentinel:
        if not updated.comment or not self._TYPE_HINT_COMMENT.search(updated.comment.value):
            return updated

        comment = self._TYPE_HINT_COMMENT.sub("", updated.comment.value)
        if not comment:
            return cst.RemoveFromParent()

        return updated.with_changes(
            whitespace=updated.whitespace,
            comment=cst.Comment(comment),
        )


    def leave_TrailingWhitespace(
        self,
        node: cst.TrailingWhitespace,
        updated: cst.TrailingWhitespace
    ) -> cst.TrailingWhitespace:
        if not updated.comment or not self._TYPE_HINT_COMMENT.search(updated.comment.value):
            return updated

        comment = self._TYPE_HINT_COMMENT.sub("", updated.comment.value)
        return updated.with_changes(
            whitespace=updated.whitespace if comment else cst.SimpleWhitespace(''),
            comment=cst.Comment(comment) if comment else None
        )
