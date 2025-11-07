from dataclasses import dataclass
from typing import NewType
from righttyper.typeinfo import TypeInfo, NoneTypeInfo


ArgumentName = NewType("ArgumentName", str)
VariableName = NewType("VariableName", str)

Filename = NewType("Filename", str)
FunctionName = NewType("FunctionName", str)

CodeId = NewType("CodeId", int)     # obtained from id(code) where code is-a CodeType
FrameId = NewType("FrameId", int)   # similarly from id(frame)


@dataclass(eq=True, order=True, frozen=True)
class FuncId:
    file_name: Filename
    first_code_line: int
    func_name: FunctionName


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
