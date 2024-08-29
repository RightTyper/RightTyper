from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import libcst as cst

from righttyper.righttyper_types import (
    ArgumentName,
    Filename,
    FuncInfo,
    FunctionName,
    Typename,
)


class TypeNameExtractor(cst.CSTVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node: cst.Name) -> Optional[bool]:
        self.names.add(node.value)
        return True

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> Union[cst.BaseStatement, cst.RemovalSentinel]:
        self.class_name.pop()
        return original_node

    def _should_output_as_string(self, annotation: str) -> bool:
        # Parse the annotation expression and extract all names
        parsed_expr = cst.parse_expression(annotation)
        extractor = TypeNameExtractor()
        parsed_expr.visit(extractor)
        components = extractor.names
        
        # Check if all extracted names are in the allowed types list
        return not all(comp in self.allowed_types for comp in components)
    
    def visit_Attribute(self, node: cst.Attribute) -> Optional[bool]:
        # Handle cases like typing.List or custom_module.MyType
        full_name = node.attr.value
        current_node = node.value
        while isinstance(current_node, cst.Attribute):
            full_name = f"{current_node.attr.value}.{full_name}"
            current_node = current_node.value
        if isinstance(current_node, cst.Name):
            full_name = f"{current_node.value}.{full_name}"
        self.names.add(full_name)
        return True

    
class AnnotateFunctionTransformer(cst.CSTTransformer):
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
        allowed_types: List[str] = None,
        # generate_annotations_as_strings: bool = False,
    ) -> None:
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.class_name: List[str] = []
        # Initialize allowed types or use a default list
        self.allowed_types = allowed_types or [
            "Any",
            "bool",
            "bytes",
            "complex",
            "Dict",
            "dict",
            "float",
            "FrozenSet",
            "frozenset",
            "int",
            "List",
            "list",
            "None",
            "Optional",
            "Set",
            "set",
            "str",
            "Tuple",
            "Union"
        ]

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.class_name.append(node.name.value)
        return None

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> Union[cst.BaseStatement, cst.RemovalSentinel]:
        self.class_name.pop()
        return original_node


    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> Union[cst.BaseStatement, cst.RemovalSentinel]:
        self.class_name.pop()
        return original_node

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> Union[cst.BaseStatement, cst.RemovalSentinel]:
        self.class_name.pop()
        return original_node

    def _should_output_as_string(self, annotation: str) -> bool:
        # Parse the annotation expression and extract all names
        parsed_expr = cst.parse_expression(annotation)
        extractor = TypeNameExtractor()
        parsed_expr.visit(extractor)
        components = extractor.names
        
        # Check if all extracted names are in the allowed types list
        return not all(comp in self.allowed_types for comp in components)

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        name = original_node.name.value

        key = FuncInfo(
            Filename(self.filename),
            FunctionName(name),
        )
        if key in self.type_annotations:
            args, return_type = self.type_annotations[key]

            new_parameters = []
            for parameter in updated_node.params.params:
                for arg, annotation_ in args:
                    if parameter.name.value == arg:
                        if arg not in self.not_annotated[key]:
                            continue
                        if self._should_output_as_string(annotation_):
                            annotation_expr = cst.SimpleString(
                                f'"{annotation_}"'
                            )
                        else:
                            annotation_expr = cst.parse_expression(annotation_)
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

            if self._should_output_as_string(return_type):
                return_type_expr = cst.SimpleString(f'"{return_type}"')
            else:
                return_type_expr = cst.parse_expression(return_type)

            if "return" in self.not_annotated[key]:
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

