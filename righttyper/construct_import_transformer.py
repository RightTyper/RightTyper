from typing import Set

import libcst as cst

from righttyper.get_import_details import (
    generate_import_nodes,
)
from righttyper.righttyper_types import ImportInfo


class ConstructImportTransformer(cst.CSTTransformer):
    def __init__(
        self,
        imports: Set[ImportInfo],
        root_path: str,
    ) -> None:
        self.imports = imports
        self.root_path = root_path

    def leave_Module(
        self,
        original_node: cst.Module,
        updated_node: cst.Module,
    ) -> cst.Module:
        new_imports = []
        for imp in self.imports:
            q = generate_import_nodes(imp.import_details)
            new_imports.extend(q)


        # Ensure that we only add valid statements, not EmptyLine nodes
        valid_imports = [imp for imp in new_imports if not isinstance(imp, cst.EmptyLine)]
            
        # Create an If(TYPE_CHECKING) node
        type_checking_imports = cst.If(
            test=cst.Name("TYPE_CHECKING"),
            body=cst.IndentedBlock(body=[cst.SimpleStatementLine([imp]) for imp in valid_imports]),
        )

        # Add the TYPE_CHECKING block at the beginning of the module
        return updated_node.with_changes(
            body=[type_checking_imports] + list(updated_node.body)
        )
