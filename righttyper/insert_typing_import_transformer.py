from typing import Optional

import libcst as cst


class InsertTypingImportTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.has_typing_import = False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        # Check if there is already an import from 'typing' with '*'
        if (
            isinstance(node.module, cst.Name)
            and node.module.value == "typing"
            and isinstance(node.names, cst.ImportStar)
        ):
            self.has_typing_import = True
        return None

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        if not self.has_typing_import:
            # Create the import statement
            typing_import_star = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names=cst.ImportStar(),
                    )
                ]
            )
            typing_import = cst.SimpleStatementLine(
                body=[
                    cst.ImportFrom(
                        module=cst.Name(value="typing"),
                        names=[
                            cst.ImportAlias(name=cst.Name(value="Any")),
                            cst.ImportAlias(name=cst.Name(value="Dict")),
                            cst.ImportAlias(name=cst.Name(value="FrozenSet")),
                            cst.ImportAlias(name=cst.Name(value="List")),
                            cst.ImportAlias(name=cst.Name(value="None")),
                            cst.ImportAlias(name=cst.Name(value="Optional")),
                            cst.ImportAlias(name=cst.Name(value="Set")),
                            cst.ImportAlias(name=cst.Name(value="Tuple")),
                            cst.ImportAlias(name=cst.Name(value="Union")),
                        ],
                    )
                ]
            )
            # Insert the import at the top of the module
            new_body = [typing_import] + list(updated_node.body)
            self.has_typing_import = (
                True  # Ensure we don't add the import more than once
            )
            return updated_node.with_changes(body=new_body)
        return updated_node
