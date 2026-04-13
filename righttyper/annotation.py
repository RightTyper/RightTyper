from dataclasses import dataclass
from righttyper.righttyper_types import ArgumentName, VariableName
from righttyper.typeinfo import TypeInfo


@dataclass(eq=True)
class FuncAnnotation:
    args: dict[ArgumentName, TypeInfo]
    retval: TypeInfo
    varargs: str|None
    kwargs: str|None
    variables: dict[VariableName, TypeInfo]


@dataclass(eq=True, frozen=True)
class ModuleVars:
    variables: dict[VariableName, TypeInfo]


@dataclass
class TraceDistribution:
    """A single observed call trace with its frequency."""
    args: dict[str, str]  # arg_name -> type_string
    retval: str           # return type string
    pct: float            # percentage of total observations
