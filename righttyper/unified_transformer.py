from typing import Dict, List, Optional, Set, Tuple

import libcst as cst

from righttyper.get_import_details import generate_import_nodes
from righttyper.righttyper_types import (
    ArgumentName,
    Filename,
    FuncInfo,
    FunctionName,
    ImportInfo,
    Typename,
)


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
        allowed_types: List[Typename] = [],
        imports: Set[ImportInfo] = set(),
    ) -> None:
        # Initialize AnnotateFunctionTransformer data
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.allowed_types = allowed_types or [
            Typename(t)
            for t in [
                "Any",
                "bool",
                "bytes",
                "Callable",
                "complex",
                "Dict",
                "dict",
                "float",
                "FrozenSet",
                "frozenset",
                "Generator",
                "int",
                "List",
                "list",
                "None",
                "Optional",
                "Set",
                "set",
                "str",
                "Tuple",
                "Union",
            ]
        ]

        # Initialize ConstructImportTransformer data
        self.imports = imports

        # Initialize InsertTypingImportTransformer data
        self.has_typing_import = False

        # Track `from __future__` imports
        self.future_imports: List[cst.SimpleStatementLine] = []

    # Helper method from AnnotateFunctionTransformer
    def _should_output_as_string(self, annotation: str) -> bool:
        parsed_expr = cst.parse_expression(annotation)
        extractor = TypeNameExtractor()
        parsed_expr.visit(extractor)
        components = extractor.names
        return not all(comp in self.allowed_types for comp in components)

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
                        annotation_expr : cst.BaseExpression
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
                        break
                else:
                    new_parameters.append(parameter)

            return_type_expr: cst.BaseExpression
            if self._should_output_as_string(return_type):
                return_type_expr = cst.SimpleString(f'"{return_type}"')
            else:
                return_type_expr = cst.parse_expression(return_type)

            if "return" in self.not_annotated.get(key, set()):
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(
                        params=new_parameters
                    ),
                    returns=cst.Annotation(annotation=return_type_expr),
                )
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
        # Step 1: Collect `from __future__` imports and remove them
        new_body : List[cst.BaseStatement] = []
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
                    self.future_imports.append(stmt)
                else:
                    new_body.append(stmt)
            else:
                new_body.append(stmt)
            updated_node = updated_node.with_changes(body=new_body)

        # Add additional imports if needed
        if self.imports:
            new_imports = []
            for imp in self.imports:
                # Make sure not to include imports specific to righttyper
                if "righttyper" not in imp.function_fname:
                    new_imports.extend(
                        generate_import_nodes(imp.import_details)
                    )

            valid_imports = [
                imp
                for imp in new_imports
                if not isinstance(imp, cst.EmptyLine)
            ]

            type_checking_imports = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(
                    body=[
                        cst.SimpleStatementLine([imp]) for imp in valid_imports
                    ]
                ),
            )

            new_body = [type_checking_imports] + list(new_body)
            updated_node = updated_node.with_changes(body=new_body)

        if not self.has_typing_import:
            typing_import = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names=[
                            cst.ImportAlias(name=cst.Name(value="Any")),
                            cst.ImportAlias(name=cst.Name(value="Callable")),
                            cst.ImportAlias(name=cst.Name(value="Dict")),
                            cst.ImportAlias(name=cst.Name(value="FrozenSet")),
                            cst.ImportAlias(name=cst.Name(value="Generator")),
                            cst.ImportAlias(name=cst.Name(value="List")),
                            cst.ImportAlias(name=cst.Name(value="Never")),
                            cst.ImportAlias(name=cst.Name(value="Optional")),
                            cst.ImportAlias(name=cst.Name(value="Set")),
                            cst.ImportAlias(name=cst.Name(value="Tuple")),
                            cst.ImportAlias(
                                name=cst.Name(value="TYPE_CHECKING")
                            ),
                            cst.ImportAlias(name=cst.Name(value="Union")),
                        ],
                    )
                ]
            )
            new_body = [typing_import] + list(updated_node.body)
            updated_node = updated_node.with_changes(body=new_body)
            self.has_typing_import = True

        # Step 2: Determine where to insert the `from __future__` imports
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
            + self.future_imports
            + new_body[insertion_index:]
        )

        updated_node = updated_node.with_changes(body=new_body)

        return updated_node


# TypeNameExtractor helper class remains unchanged
class TypeNameExtractor(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names : Set[str] = set()

    def visit_Name(self, node: cst.Name) -> Optional[bool]:
        self.names.add(node.value)
        return True

    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        full_name = node.attr.value
        current_node = node.value
        while isinstance(current_node, cst.Attribute):
            full_name = f"{current_node.attr.value}.{full_name}"
            current_node = current_node.value
        if isinstance(current_node, cst.Name):
            full_name = f"{current_node.value}.{full_name}"
        self.names.add(full_name)
        return True
