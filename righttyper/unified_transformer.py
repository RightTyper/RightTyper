from typing import Dict, List, Optional, Set, Tuple

import libcst as cst

from righttyper.righttyper_types import (
    ArgumentName,
    Filename,
    FuncInfo,
    FunctionName,
    ImportInfo,
    Typename,
)


_BUILTIN_TYPES : Set[Typename] = {
    Typename(t) for t in [
        "bool",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "None",
        "set",
        "str",
    ]
}

_TYPING_TYPES : Set[Typename] = {
    Typename(t) for t in [
        "Any",
        "Callable",
        "Dict",
        "FrozenSet",
        "Generator",
        "List",
        "Never",    # FIXME requires Python >= 3.11
        "Optional",
        "Set",
        "Tuple",
        "Union",
    ]
}


def _dotted_name(name: str) -> cst.Attribute | cst.Name:
    """Creates CST nodes equivalent to a module name, dotted or not"""
    parts = name.split(".")
    if len(parts) == 1:
        return cst.Name(parts[0])
    base: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
        base = cst.Attribute(value=base, attr=cst.Name(part))
    return base


class UnifiedTransformer(cst.CSTTransformer):
    def __init__(
        self,
        filename: str,
        type_annotations: Dict[
            FuncInfo,
            Tuple[
                List[Tuple[ArgumentName, Typename]],
                Typename,
            ],
        ],
        not_annotated: Dict[FuncInfo, Set[ArgumentName]],
        allowed_types: List[Typename] = [], # TODO delete?
        imports: Set[ImportInfo] = set(), # TODO delete
    ) -> None:
        # Initialize AnnotateFunctionTransformer data
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.known_types : Set[Typename] = set(allowed_types) | _BUILTIN_TYPES | _TYPING_TYPES


    def _should_output_as_string(self, annotation: str) -> bool:
        return any(t not in self.known_types for t in types_in_annotation(annotation))


    def visit_Module(self, node: cst.Module) -> bool:
        # Initialize mutable members here, just in case transformer gets reused
        self.used_types : Set[Typename] = set()
        # TODO modify known_types based on existing imports, so that they're not unnecessarily imported
        return True


    # AnnotateFunctionTransformer logic
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        name = original_node.name.value
        key = FuncInfo(Filename(self.filename), FunctionName(name))

        if key in self.type_annotations:
            args, return_type = self.type_annotations[key]

            new_parameters = []
            for parameter in updated_node.params.params:
                for arg, annotation_ in args:
                    if parameter.name.value == arg:
                        if arg not in self.not_annotated.get(key, set()):
                            continue

                        # TODO recognize (and use) import aliases, transforming annotation
                        annotation_expr: cst.BaseExpression
                        if self._should_output_as_string(annotation_):
                            annotation_expr = cst.SimpleString(
                                f'"{annotation_}"'
                            )
                        else:
                            parsed_expr = cst.parse_expression(annotation_)
                            annotation_expr = parsed_expr
                        new_parameters.append(
                            parameter.with_changes(
                                annotation=cst.Annotation(
                                    annotation=annotation_expr
                                )
                            )
                        )
                        self.used_types |= types_in_annotation(annotation_)
                        break
                else:
                    new_parameters.append(parameter)

            if "return" in self.not_annotated.get(key, set()):
                return_type_expr: cst.BaseExpression
                if self._should_output_as_string(return_type):
                    return_type_expr = cst.SimpleString(f'"{return_type}"')
                else:
                    return_type_expr = cst.parse_expression(return_type)

                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(
                        params=new_parameters
                    ),
                    returns=cst.Annotation(annotation=return_type_expr),
                )
                self.used_types |= types_in_annotation(return_type)
            else:
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(
                        params=new_parameters
                    ),
                )
        return updated_node

    # ConstructImportTransformer logic
    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        # Collect `from __future__` imports and remove them
        future_imports: List[cst.BaseStatement] = []
        new_body: List[cst.BaseStatement] = []
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
        unknown_types = self.used_types - self.known_types
        if unknown_types:
            # TODO update any existing "if TYPE_CHECKING"
            unknown_modules = {
                '.'.join(name.split('.')[:-1]) if '.' in name else name
                for name in unknown_types
            }
            type_checking_imports = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(
                    body=[
                        cst.SimpleStatementLine([
                            cst.Import([cst.ImportAlias(_dotted_name(m))])
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


def types_in_annotation(annotation: str) -> Set[str]:
    """Extracts all type names included in a type annotation."""

    class TypeNameExtractor(cst.CSTVisitor):
        def __init__(self) -> None:
            self.names: Set[str] = set()

        def visit_Name(self, node: cst.Name) -> Optional[bool]:
            self.names.add(node.value)
            return False 

        def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
            full_name = node.attr.value
            current_node = node.value
            while isinstance(current_node, cst.Attribute):
                full_name = f"{current_node.attr.value}.{full_name}"
                current_node = current_node.value
            if isinstance(current_node, cst.Name):
                full_name = f"{current_node.value}.{full_name}"
            self.names.add(full_name)
            return False 

    parsed_expr = cst.parse_expression(annotation)
    extractor = TypeNameExtractor()
    parsed_expr.visit(extractor)
    return extractor.names
