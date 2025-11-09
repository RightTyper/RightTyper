from dataclasses import dataclass
from typing import NewType
import types


Filename = NewType("Filename", str)
ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)
FunctionName = NewType("FunctionName", str)

@dataclass(eq=True, order=True, frozen=True)
class CodeId:
    file_name: Filename
    func_name: FunctionName
    first_code_line: int

    # if a <genexpr> or such, hash(code) to be able to differentiate same-line objects
    code_hash: int


    @staticmethod
    def from_code(code: types.CodeType) -> "CodeId":
        return CodeId(
            Filename(code.co_filename),
            FunctionName(code.co_qualname),
            code.co_firstlineno,
            hash(code) if code.co_qualname.endswith('>') else 0
        )


def cast_not_None[T](x: T | None) -> T:
    """Small utility to just cast off None from x's type"""
    return x    # type: ignore[return-value]
