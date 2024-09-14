import libcst as cst


class InsertTypingImportTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.has_typing_import = False

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
            # Insert the import at the top of the module
            new_body = [typing_import] + list(updated_node.body)
            self.has_typing_import = (
                True  # Ensure we don't add the import more than once
            )
            return updated_node.with_changes(body=new_body)
        return updated_node
