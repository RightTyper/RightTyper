from dataclasses import dataclass
from righttyper.righttyper_types import ArgumentName, VariableName
from righttyper.typeinfo import TypeInfo


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



