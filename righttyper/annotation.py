from dataclasses import dataclass, field
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


@dataclass
class TypeDistributions:
    """Per-argument/return type frequency distributions for a function or variable."""
    # Maps arg name (or "return") to list of (type_string, percentage), sorted descending
    distributions: dict[str, list[tuple[str, float]]] = field(default_factory=dict)



