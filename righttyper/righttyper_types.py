from dataclasses import dataclass
from typing import NewType
import types
from righttyper.typeinfo import TypeInfo, NoneTypeInfo


ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)

Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)

@dataclass(eq=True, order=True, frozen=True)
class CodeId:
    file_name: Filename
    first_code_line: int
    func_name: FunctionName


    @staticmethod
    def from_code(code: types.CodeType) -> "CodeId":
        return CodeId(
            Filename(code.co_filename),
            code.co_firstlineno,
            FunctionName(code.co_qualname),
        )


@dataclass(eq=True, frozen=True)
class FuncAnnotation:
    args: list[tuple[ArgumentName, TypeInfo]]   # TODO: make me a map?
    retval: TypeInfo
    varargs: str|None
    kwargs: str|None
    variables: list[tuple[VariableName, TypeInfo]]


@dataclass(eq=True, frozen=True)
class ModuleVars:
    variables: list[tuple[VariableName, TypeInfo]]


type CallTrace = tuple[TypeInfo, ...]


def cast_not_None[T](x: T | None) -> T:
    """Small utility to just cast off None from x's type"""
    return x    # type: ignore[return-value]
