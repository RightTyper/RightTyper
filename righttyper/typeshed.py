import libcst as cst
from collections.abc import Sequence
from libcst.metadata import MetadataWrapper, ScopeProvider, QualifiedNameProvider, QualifiedName
from libcst.metadata.scope_provider import (
    ImportAssignment,
    BuiltinAssignment
)
from righttyper.typeinfo import TypeInfo, TypeInfoArg, NoneTypeInfo, UnknownTypeInfo
from righttyper.righttyper_utils import normalize_module_name
import typeshed_client
from righttyper.logger import logger
from ast import literal_eval
import typing
import builtins
from functools import cache
from pathlib import Path


_BUILTINS: typing.Final[dict[str, TypeInfo]] = {
    n: TypeInfo('', n, type_obj=t)  # note we use '' as the module name
    for n, t in builtins.__dict__.items()
    if type(t) is type
}


def get_full_name(node: cst.Attribute|cst.Name|cst.BaseExpression) -> str:
    """Extracts a module name from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        return f"{get_full_name(node.value)}.{get_full_name(node.attr)}"

    assert isinstance(node, cst.Name), f"{node=}"
    return node.value


class FunctionFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self, module_name: str, want: str):
        self._module_name = module_name
        self._name_stack: list[str] = []
        self._want = want.split('.')
        self.result: list[TypeInfo] = []

    def _type_from_name(self, scope, node: cst.Name|cst.Attribute) -> TypeInfo:
        full_name = get_full_name(node)
        if full_name == 'None':
            return NoneTypeInfo

        first_name = full_name.split('.')[0]
#        print(f"scope[{first_name}]={scope[first_name]}")

        # TODO pick "most recent" assignment, perhaps using PositionProvider
        for assignment in scope[first_name]:
            if isinstance(assignment, ImportAssignment):
                if isinstance(assignment.node, cst.Import):
                    module_name = ''
                    for i_alias in assignment.node.names:
                        if (
                            isinstance(i_alias.asname, cst.AsName)
                            and isinstance(i_alias.asname.name, cst.Name)
                            and i_alias.asname.name.value == first_name
                        ):
                            return TypeInfo(
                                get_full_name(i_alias.name),
                                full_name[len(first_name)+1:]
                            )
                        else:
                            name = get_full_name(i_alias.name)
                            if full_name.startswith(name) and full_name[len(name)] == '.':
                                if not module_name or len(name) > len(module_name):
                                    module_name = name

                    if module_name:
                        return TypeInfo(
                            module_name,
                            full_name[len(module_name)+1:]
                        )

                elif (
                    isinstance(assignment.node, cst.ImportFrom)
                    and assignment.node.module
                    and not isinstance(assignment.node.names, cst.ImportStar)
                ):
                    for i_alias in assignment.node.names:
                        if (
                            isinstance(i_alias.asname, cst.AsName)
                            and isinstance(i_alias.asname.name, cst.Name)
                            and i_alias.asname.name.value == first_name
                        ):
                            name = get_full_name(i_alias.name)
                            return TypeInfo(
                                get_full_name(assignment.node.module),
                                name if len(full_name) == len(first_name) else name + '.' + full_name[len(first_name)+1:]
                            )

                    return TypeInfo(
                        get_full_name(assignment.node.module),
                        full_name
                    )

            elif isinstance(assignment, BuiltinAssignment):
                if (t := _BUILTINS.get(full_name)):
                    return t

        return TypeInfo(normalize_module_name(self._module_name), get_full_name(node))


    def _parse_type(self, scope, expr: cst.BaseExpression | None) -> TypeInfo:
        # Name / Attribute leaf
        if isinstance(expr, (cst.Name, cst.Attribute)):
            return self._type_from_name(scope, expr)

        # quoted type (forward reference)
        if isinstance(expr, cst.SimpleString):
            return self._parse_type(scope, cst.parse_expression(literal_eval(expr.value)))
            try:
                return self._parse_type(cst.parse_expression(literal_eval(expr.value)))
            except BaseException as e:
                logger.info(f"Error parsing {expr.value}: {e}")
                return UnknownTypeInfo

        # A | B
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.BitOr):
            return TypeInfo.from_set({self._parse_type(scope, expr.left), self._parse_type(scope, expr.right)})

        # list literal (e.g., within Callable)
        if isinstance(expr, cst.List):
            return TypeInfo.list([
                self._parse_type(scope, el.value)
                for el in expr.elements
                if isinstance(el, cst.Element)
            ])

        def expr2type_arg(expr: cst.BaseExpression) -> TypeInfoArg:
            if isinstance(expr, cst.Ellipsis):
                return ...

            if isinstance(expr, cst.Tuple) and not expr.elements:
                return ()

            if isinstance(expr, cst.SimpleString):
                try:
                    value = literal_eval(expr.value)
                    try:
                        quoted_expr = cst.parse_expression(value)
                        if (quoted_type := self._parse_type(scope, quoted_expr)) is not UnknownTypeInfo:
                            return quoted_type
                    except:
                        return value
                except BaseException as e:
                    logger.info(f"Error evaluating {expr.value}: {e}")
                    return UnknownTypeInfo  # XXX is this the right thing to do here?

            return self._parse_type(scope, expr)

        # Generic: X[T, ...]
        if isinstance(expr, cst.Subscript):
            return self._parse_type(scope, expr.value).replace(args=tuple(
                expr2type_arg(el.slice.value)
                for el in expr.slice
                if isinstance(el, cst.SubscriptElement)
                if isinstance(el.slice, cst.Index)
            ))

        logger.info(f"Unable to convert type expression {expr}")
        return UnknownTypeInfo


    def _parse_annotation(self, a: cst.Annotation | None):
        if a and a.annotation:
            scope = self.get_metadata(ScopeProvider, a)
            return self._parse_type(scope, a.annotation)

        return None


    def _save_results(self, f: cst.FunctionDef) -> None:
        p: typing.Any

        for p in (*f.params.posonly_params, *f.params.params):
            self.result.append(self._parse_annotation(p.annotation))

        if isinstance(p := f.params.star_arg, cst.Param):
            self.result.append(self._parse_annotation(p.annotation))

        for p in f.params.kwonly_params:
            self.result.append(self._parse_annotation(p.annotation))

        if isinstance(p := f.params.star_kwarg, cst.Param):
            self.result.append(self._parse_annotation(p.annotation))

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._name_stack.append(node.name.value)
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self._name_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._name_stack.append(node.name.value)

        if self._name_stack == self._want:
            self._save_results(node)

        self._name_stack.append('<locals>')
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._name_stack.pop()
        self._name_stack.pop()


def get_func_params(code: cst.Module, module_name: str, func_name: str) -> list[TypeInfo]:
    finder = FunctionFinder(module_name, func_name)
    w = MetadataWrapper(code)
    w.visit(finder)
    return finder.result


@cache
def get_typeshed_module(module_name: str) -> cst.Module|None:
    stub_path: Path | None = typeshed_client.get_stub_file(module_name)
    if not stub_path:
        return None
    return cst.parse_module(stub_path.read_text('utf-8'))


def get_typeshed_func_params(module_name: str, qualname: str) -> list[TypeInfo] | None:
    if not (module := get_typeshed_module(module_name)):
        return None

    return get_func_params(module, module_name, qualname)
