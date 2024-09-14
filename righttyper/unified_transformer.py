import libcst as cst
from typing import Dict, List, Set, Tuple, Optional, Union
from righttyper.righttyper_types import ArgumentName, Filename, FuncInfo, FunctionName, Typename, ImportInfo
from righttyper.get_import_details import generate_import_nodes


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
        not_annotated: Dict[FuncInfo, Set[str]],
        allowed_types: List[str] = [],
        imports: Set[ImportInfo] = set(),
        root_path: str = "",
    ) -> None:
        # Initialize AnnotateFunctionTransformer data
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.class_name: List[str] = []
        self.allowed_types = allowed_types or [
            "Any", "bool", "bytes", "Callable", "complex", "Dict", "dict",
            "float", "FrozenSet", "frozenset", "Generator", "int", "List",
            "list", "None", "Optional", "Set", "set", "str", "Tuple", "Union",
        ]

        # Initialize ConstructImportTransformer data
        self.imports = imports
        self.root_path = root_path

        # Initialize InsertTypingImportTransformer data
        self.has_typing_import = False

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
                        if arg not in self.not_annotated[key]:
                            continue
                        if self._should_output_as_string(annotation_):
                            annotation_expr = cst.SimpleString(f'"{annotation_}"')
                        else:
                            annotation_expr = cst.parse_expression(annotation_)
                        new_parameters.append(
                            parameter.with_changes(annotation=cst.Annotation(annotation=annotation_expr))
                        )
                        break
                else:
                    new_parameters.append(parameter)

            if self._should_output_as_string(return_type):
                return_type_expr = cst.SimpleString(f'"{return_type}"')
            else:
                return_type_expr = cst.parse_expression(return_type)

            if "return" in self.not_annotated[key]:
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(params=new_parameters),
                    returns=cst.Annotation(annotation=return_type_expr),
                )
            else:
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(params=new_parameters),
                )
        return updated_node

    # ConstructImportTransformer logic
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Add imports if any
        if self.imports:
            new_imports = []
            for imp in self.imports:
                new_imports.extend(generate_import_nodes(imp.import_details))

            # Filter valid import statements
            valid_imports = [imp for imp in new_imports if not isinstance(imp, cst.EmptyLine)]
            
            # Create If(TYPE_CHECKING) node
            type_checking_imports = cst.If(
                test=cst.Name("TYPE_CHECKING"),
                body=cst.IndentedBlock(body=[cst.SimpleStatementLine([imp]) for imp in valid_imports]),
            )

            # Add the TYPE_CHECKING block at the beginning of the module
            updated_node = updated_node.with_changes(body=[type_checking_imports] + list(updated_node.body))

        # Insert typing import if needed (InsertTypingImportTransformer logic)
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
                            cst.ImportAlias(name=cst.Name(value="Optional")),
                            cst.ImportAlias(name=cst.Name(value="Set")),
                            cst.ImportAlias(name=cst.Name(value="Tuple")),
                            cst.ImportAlias(name=cst.Name(value="TYPE_CHECKING")),
                            cst.ImportAlias(name=cst.Name(value="Union")),
                        ],
                    )
                ]
            )
            updated_node = updated_node.with_changes(body=[typing_import] + list(updated_node.body))
            self.has_typing_import = True

        return updated_node


# TypeNameExtractor helper class remains unchanged
class TypeNameExtractor(cst.CSTVisitor):
    def __init__(self):
        self.names = set()

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
