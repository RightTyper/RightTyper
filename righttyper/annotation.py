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
class TraceDistribution:
    """A single observed call trace with its frequency."""
    args: dict[str, str]  # arg_name -> type_string
    retval: str           # return type string
    pct: float            # percentage of total observations


@dataclass
class TypeDistributions:
    """Observed type distributions for a function or variable."""
    # Function traces, sorted descending by percentage. Only populated when >1 distinct trace.
    traces: list[TraceDistribution] = field(default_factory=list)
    # Variable type lists (no frequency data). Maps var name to list of type strings.
    variable_types: dict[str, list[str]] = field(default_factory=dict)



