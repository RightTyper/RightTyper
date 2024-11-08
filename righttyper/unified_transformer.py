import typing
import builtins
import collections.abc
import types
import libcst as cst
import re

from righttyper.righttyper_types import (
    ArgumentName,
    Filename,
    FuncInfo,
    FunctionName,
    Typename,
)


_BUILTIN_TYPES : set[Typename] = {
    Typename(t) for t in (
        "None",
        *(name for name, value in builtins.__dict__.items()if isinstance(value, type))
    )
}

# FIXME this prevents us from missing out on well-known "typing." types,
# but is risky... change to receiving fully qualified names and simplifying
# them in context.
_TYPING_TYPES : set[Typename] = {
    Typename(t) for t in typing.__all__
}

# Regex for a type hint comment
_TYPE_HINT_COMMENT = re.compile(
    r"#\stype:\s*[^\s]+"
)


# Regex for a function and retval type hint comment
_TYPE_HINT_COMMENT_FUNC = re.compile(
    r"#\stype:\s*\([^\)]*\)\s*->\s*[^\s]+"
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

def _nodes_to_dotted_name(node: cst.CSTNode) -> str:
    """Extracts a module name from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        return f"{_nodes_to_dotted_name(node.value)}.{_nodes_to_dotted_name(node.attr)}"

    assert isinstance(node, cst.Name)
    return node.value

def _nodes_to_name(node: cst.CSTNode) -> str:
    """Extracts the top-level name (e.g., 'foo' in 'foo.bar') from CST Attribute/Name nodes."""
    while isinstance(node, cst.Attribute):
        node = node.value

    assert isinstance(node, cst.Name)
    return node.value

def _nodes_to_all_dotted_names(node: cst.CSTNode) -> list[str]:
    """Extracts the list of all module and parent module names from CST Attribute/Name nodes."""
    if isinstance(node, cst.Attribute):
        names = _nodes_to_all_dotted_names(node.value)
        return [*names, f"{names[-1]}.{node.attr.value}"]

    assert isinstance(node, cst.Name)
    return [node.value]

def _get_str_attr(obj: object, path: str) -> str|None:
    """Looks for a str-valued attribute along the given dot-separated attribute path."""
    for elem in path.split('.'):
        if obj and isinstance(obj, (list, tuple)):
            obj = obj[0]

        if (obj := getattr(obj, elem, None)) is None:
            break

    return obj if isinstance(obj, str) else None

def _namespace_of(t: str) -> str:
    """Returns the dotted name prefix of a name, as in "foo.bar" for "foo.bar.baz"."""
    return '.'.join(t.split('.')[:-1])


class UnifiedTransformer(cst.CSTTransformer):
    def __init__(
        self,
        filename: str,
        type_annotations: dict[
            FuncInfo,
            tuple[
                list[tuple[ArgumentName, Typename]],
                Typename,
            ],
        ],
        not_annotated: dict[FuncInfo, set[ArgumentName]]
    ) -> None:
        # Initialize AnnotateFunctionTransformer data
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.has_future_annotations = False


    def _is_valid(self, annotation: str) -> bool:
        # local names such as foo.<locals>.Bar yield this exception
        try:
            cst.parse_expression(annotation)
            return True
        except cst.ParserSyntaxError:
            return False

    def _should_output_as_string(self, annotation: str) -> bool:
        #print(f"{types_in_annotation(annotation)=}")
        return (
            not self.has_future_annotations
            and any(
                t not in self.known_names
                and _namespace_of(t) not in self.known_names
                and _namespace_of(t) not in self.imported_modules
                for t in types_in_annotation(annotation)
            )
        )

    def visit_Module(self, node: cst.Module) -> bool:
        # Initialize mutable members here, just in case transformer gets reused

        global_imports = _global_imports(node)

        self.known_names : set[Typename] = _BUILTIN_TYPES | _TYPING_TYPES | {
            Typename(_nodes_to_name(alias.asname.name if alias.asname is not None else alias.name))
            for alias, imp in global_imports
        }
        self.known_names.update(set(_global_assigns(node)))
        #print(f"{self.known_names=}")

        self.imported_modules: set[str] = set().union(*(
            set(_nodes_to_all_dotted_names(alias.name))
            for alias, imp in global_imports if alias.asname is None
        ))
        #print(f"{self.imported_modules=}")

        self.used_types : set[Typename] = set()
        self.name_stack : list[str] = []

        self.has_future_annotations = any(
            imp.module.value == "__future__" and alias.name.value == "annotations"
            for alias, imp in global_imports
            if isinstance(imp, cst.ImportFrom)
            and isinstance(imp.module, cst.Name) and isinstance(alias.name, cst.Name)
        )

        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.name_stack.append(node.name.value)
        return True

    def leave_ClassDef(self, orig_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # a class is known once its definition is done
        self.known_names.add(Typename(".".join(self.name_stack)))
        self.name_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.name_stack.extend([node.name.value, "<locals>"])
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        name = ".".join(self.name_stack[:-1])
        self.name_stack.pop()
        self.name_stack.pop()
        key = FuncInfo(Filename(self.filename), FunctionName(name))

        if key in self.type_annotations:
            args, return_type = self.type_annotations[key]

            new_parameters = []
            for parameter in updated_node.params.params:
                for arg, annotation_ in args:
                    if parameter.name.value == arg:
                        if arg not in self.not_annotated.get(key, set()) or not self._is_valid(annotation_):
                            continue

                        # TODO recognize (and use) import aliases, transforming annotation
                        annotation_expr: cst.BaseExpression
                        if self._should_output_as_string(annotation_):
                            new_par = parameter.with_changes(
                                annotation=cst.Annotation(
                                    annotation=cst.SimpleString(f'"{annotation_}"')
                                )
                            )
                        else:
                            new_par = parameter.with_changes(
                                annotation=cst.Annotation(
                                    annotation=cst.parse_expression(annotation_)
                                )
                            )

                        # remove per-parameter type hint comment for non-last parameter
                        if ((comment := _get_str_attr(new_par, "comma.whitespace_after.first_line.comment.value"))
                            and _TYPE_HINT_COMMENT.match(comment)):
                            new_par = new_par.with_changes(
                                comma=new_par.comma.with_changes(   # type: ignore[union-attr]
                                    whitespace_after=new_par.comma.whitespace_after.with_changes( # type: ignore[union-attr]
                                        first_line=cst.TrailingWhitespace()
                                    )
                                )
                            )

                        # remove per-parameter type hint comment for last parameter
                        if ((comment := _get_str_attr(new_par, "whitespace_after_param.first_line.comment.value"))
                            and _TYPE_HINT_COMMENT.match(comment)):
                            new_par = new_par.with_changes(
                                whitespace_after_param=new_par.whitespace_after_param.with_changes(
                                    first_line=cst.TrailingWhitespace()
                                )
                            )

                        new_parameters.append(new_par)
                        self.used_types |= types_in_annotation(annotation_)
                        break
                else:
                    new_parameters.append(parameter)

            updated_node = updated_node.with_changes(
                params=updated_node.params.with_changes(
                    params=new_parameters
                )
            )

            if "return" in self.not_annotated.get(key, set()) and self._is_valid(return_type):
                return_type_expr: cst.BaseExpression
                if self._should_output_as_string(return_type):
                    return_type_expr = cst.SimpleString(f'"{return_type}"')
                else:
                    return_type_expr = cst.parse_expression(return_type)

                updated_node = updated_node.with_changes(
                    returns=cst.Annotation(annotation=return_type_expr),
                )
                self.used_types |= types_in_annotation(return_type)

                # remove "(...) -> retval"-style type hint comment
                if ((comment := _get_str_attr(updated_node, "body.body.leading_lines.comment.value"))
                    and _TYPE_HINT_COMMENT_FUNC.match(comment)):
                    updated_node = updated_node.with_changes(
                        body=updated_node.body.with_changes(
                            body=(updated_node.body.body[0].with_changes(leading_lines=[]),
                                  *updated_node.body.body[1:])
                        )
                    )

            # remove single-line type hint comment in the same line as the 'def'
            if ((comment := _get_str_attr(updated_node, "body.header.comment.value"))
                and _TYPE_HINT_COMMENT_FUNC.match(comment)):
                updated_node = updated_node.with_changes(
                    body=updated_node.body.with_changes(
                        header=cst.TrailingWhitespace()))

        return updated_node

    # ConstructImportTransformer logic
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        # Collect `from __future__` imports and remove them
        future_imports: list[cst.BaseStatement] = []
        new_body: list[cst.BaseStatement] = []
        stmt: cst.BaseStatement
        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                if any(
                    isinstance(imp, cst.ImportFrom)
                    and imp.module
                    and isinstance(imp.module, cst.Name)
                    and imp.module.value == "__future__"
                    for imp in stmt.body
                ):
                    # Collect future imports to lift later
                    future_imports.append(stmt)
                else:
                    new_body.append(stmt)
            else:
                new_body.append(stmt)
            updated_node = updated_node.with_changes(body=new_body)

        # Add additional type checking imports if needed
        unknown_types = self.used_types - self.known_names
        if unknown_types:
            # TODO update any existing "if TYPE_CHECKING"
            unknown_modules = {
                _namespace_of(name) if '.' in name else name
                for name in unknown_types
            }
            unknown_modules -= self.known_names
            unknown_modules -= self.imported_modules
            type_checking_imports = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(
                    body=[
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(_dotted_name_to_nodes(m))])
                        ])
                        for m in sorted(unknown_modules)
                    ]
                ),
            )

            new_body = [type_checking_imports] + list(new_body)
            updated_node = updated_node.with_changes(body=new_body)

        # Emit "from typing import ..."
        if unknown_types or (self.used_types & _TYPING_TYPES):
            # TODO update existing import statement, if any
            typing_import = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names=[
                            cst.ImportAlias(name=cst.Name(value="TYPE_CHECKING"))
                        ] + [
                            cst.ImportAlias(name=cst.Name(value=t))
                            for t in (self.used_types & _TYPING_TYPES)
                        ]
                    )
                ]
            )
            new_body = [typing_import] + list(updated_node.body)
            updated_node = updated_node.with_changes(body=new_body)

        # Determine where to insert the `from __future__` imports
        insertion_index = 0
        for i, stmt in enumerate(new_body):
            if (
                isinstance(stmt, cst.EmptyLine)
                or isinstance(stmt, cst.SimpleStatementLine)
                and isinstance(stmt.body[0], cst.Expr)
                and isinstance(stmt.body[0].value, cst.SimpleString)
            ):
                insertion_index += 1
            else:
                break

        # Insert future imports at the top (after comments and whitespace)
        new_body = (
            new_body[:insertion_index]
            + future_imports
            + new_body[insertion_index:]
        )

        updated_node = updated_node.with_changes(body=new_body)
        return updated_node


def types_in_annotation(annotation: str) -> set[Typename]:
    """Extracts all type names included in a type annotation."""

    class TypeNameExtractor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.names: set[Typename] = set()

        def visit_Name(self, node: cst.Name) -> bool|None:
            self.names.add(Typename(node.value))
            return False 

        def visit_Attribute(self, node: cst.Attribute) -> bool|None:
            self.names.add(Typename(_nodes_to_dotted_name(node)))
            return False 

    parsed_expr = cst.parse_expression(annotation)
    extractor = TypeNameExtractor()
    parsed_expr.visit(extractor)
    return extractor.names


def _global_imports(node: cst.Module) -> list[tuple[cst.ImportAlias, cst.Import|cst.ImportFrom]]:
    """Extracts global imports in a module."""

    imports: list[tuple[cst.ImportAlias, cst.Import|cst.ImportFrom]] = []

    class Extractor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.names: set[Typename] = set()

        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            return False

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            return False

        def visit_Import(self, node: cst.Import) -> bool:
            # node.names could also be cst.ImportStar
            if isinstance(node.names, collections.abc.Sequence):
                for alias in node.names:
                    imports.append((alias, node))
            return False

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
            # node.names could also be cst.ImportStar
            if isinstance(node.names, collections.abc.Sequence):
                for alias in node.names:
                    imports.append((alias, node))
            return False

    node.visit(Extractor())
    return imports


def _global_assigns(node: cst.Module) -> list[Typename]:
    """Extracts global imports in a module."""

    assigns: list[Typename] = []

    class Extractor(cst.CSTVisitor):
        def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
            return False

        def visit_ClassDef(self, node: cst.ClassDef) -> bool:
            return False

        def visit_Assign(self, node: cst.Assign) -> bool:
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    assigns.append(Typename(t.target.value))
                elif isinstance(t.target, cst.Tuple):
                    for el in t.target.elements:
                        if isinstance(el.value, cst.Name):
                            assigns.append(Typename(el.value.value))
            return False

        def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
            if isinstance(node.target, cst.Name):
                assigns.append(Typename(node.target.value))
            return False

    node.visit(Extractor())
    return assigns
