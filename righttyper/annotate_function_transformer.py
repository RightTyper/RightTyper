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
    ) -> None:
        self.filename = filename
        self.type_annotations = type_annotations
        self.not_annotated = not_annotated
        self.class_name: List[str] = []

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

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        # real_name = ".".join(self.class_name + [original_node.name.value])
        name = original_node.name.value

        # TODO: the current handling will lead to a bug if a function
        # appears inside and outside a class definition in the same
        # file. We need to handle this case.

        key = FuncInfo(
            Filename(self.filename),
            FunctionName(name),
        )  # original_node.name.value))
        if key in self.type_annotations:
            args, return_type = self.type_annotations[key]
            # print(f"{args=}, {return_type=}")
            # Update arguments with type annotations
            new_parameters = []
            for parameter in updated_node.params.params:
                for arg, annotation_ in args:
                    if parameter.name.value == arg:
                        if arg not in self.not_annotated[key]:
                            continue
                        parsed_annotation = cst.parse_expression(annotation_)
                        new_parameters.append(
                            parameter.with_changes(
                                annotation=cst.Annotation(
                                    annotation=parsed_annotation
                                )
                            )
                        )
                        break
                else:
                    new_parameters.append(parameter)
            parsed_return_type = cst.parse_expression(return_type)
            if "return" in self.not_annotated[key]:
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(
                        params=new_parameters
                    ),
                    returns=cst.Annotation(annotation=parsed_return_type),
                )
            else:
                updated_node = updated_node.with_changes(
                    params=updated_node.params.with_changes(
                        params=new_parameters
                    ),
                )
        return updated_node
